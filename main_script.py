#!/usr/bin/env python3
"""
AI4SE 2025 - IF Statement Prediction Pipeline
Complete end-to-end implementation with checkpoint-based resumability

Key improvements over failing implementation:
1. Proper IF statement masking using line-based approach (not regex)
2. Checkpoint system - each step can be skipped if already complete
3. Robust tree-sitter integration for accurate AST parsing
4. Proper data splits with sufficient fine-tuning examples
5. Clean tokenizer integration with T5
"""

import os
import re
import csv
import json
import hashlib
import random
import requests
import zipfile
import io
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass

# Tree-sitter for Python AST parsing
import tree_sitter_languages
from tree_sitter import Parser

# Tokenization
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.processors import TemplateProcessing

# Transformers
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# Data handling
import pandas as pd
from datasets import Dataset
import torch

# ===================== CONFIGURATION =====================
class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Data collection
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # Set via environment variable
    MAX_REPOS = 1000
    MIN_STARS = 10
    
    # Dataset targets
    TOTAL_FUNCTIONS_TARGET = 500_000
    PRETRAIN_TARGET = 350_000
    FINETUNE_TARGET = 150_000
    
    # Model configuration
    VOCAB_SIZE = 32_000
    MODEL_DIM = 512
    NUM_HEADS = 8
    NUM_LAYERS = 6
    FFN_DIM = 2048
    MAX_LENGTH = 512
    
    # Training
    PRETRAIN_EPOCHS = 3
    PRETRAIN_BATCH_SIZE = 32
    FINETUNE_EPOCHS = 5
    FINETUNE_BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    
    # Checkpoints
    RAW_FUNCTIONS_FILE = DATA_DIR / "raw_functions.jsonl"
    PRETRAIN_DATA_FILE = DATA_DIR / "pretrain_data.jsonl"
    FINETUNE_TRAIN_FILE = DATA_DIR / "finetune_train.jsonl"
    FINETUNE_VAL_FILE = DATA_DIR / "finetune_val.jsonl"
    FINETUNE_TEST_FILE = DATA_DIR / "finetune_test.jsonl"
    TOKENIZER_DIR = MODELS_DIR / "tokenizer"
    PRETRAINED_MODEL_DIR = MODELS_DIR / "pretrained"
    FINETUNED_MODEL_DIR = MODELS_DIR / "finetuned"
    BENCHMARK_FILE = DATA_DIR / "benchmark_if_only_2_.csv"
    
    # Special tokens
    MASK_TOKEN = "<extra_id_0>"
    PAD_TOKEN = "<pad>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"
    
    def __init__(self):
        # Create directories
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


config = Config()


# ===================== TREE-SITTER SETUP =====================
def get_python_parser():
    """Initialize tree-sitter parser for Python"""
    try:
        py_lang = tree_sitter_languages.get_language('python')
        parser = Parser()
        parser.set_language(py_lang)
        return parser, py_lang
    except Exception as e:
        print(f" Parser initialization error: {e}")
        return None, None


PARSER, PYTHON_LANG = get_python_parser()


@dataclass
class IFStatement:
    """Represents an extracted IF statement"""
    condition: str
    line: int
    depth: int


# ===================== STEP 1: DATA COLLECTION =====================
class DataCollector:
    """Collects Python functions with IF statements from GitHub"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"token {config.GITHUB_TOKEN}"
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
    
    def should_skip(self) -> bool:
        """Check if data collection can be skipped"""
        if config.RAW_FUNCTIONS_FILE.exists():
            with open(config.RAW_FUNCTIONS_FILE, 'r') as f:
                count = sum(1 for _ in f)
            if count >= config.TOTAL_FUNCTIONS_TARGET:
                print(f" Data collection: SKIPPED ({count:,} functions already collected)")
                return True
        return False
    
    def discover_repos(self) -> List[Dict]:
        """Discover Python repositories on GitHub"""
        print(f" Discovering up to {config.MAX_REPOS} repositories...")
        repos = []
        page = 1
        
        with tqdm(total=config.MAX_REPOS, desc="Discovering repos") as pbar:
            while len(repos) < config.MAX_REPOS:
                try:
                    url = "https://api.github.com/search/repositories"
                    params = {
                        "q": f"language:python stars:>={config.MIN_STARS} fork:false archived:false",
                        "sort": "stars",
                        "order": "desc",
                        "per_page": 100,
                        "page": page
                    }
                    
                    response = self.session.get(url, params=params, timeout=30)
                    
                    if response.status_code == 403:
                        print("\n Rate limit hit, waiting 60s...")
                        import time
                        time.sleep(60)
                        continue
                    
                    if response.status_code != 200:
                        break
                    
                    items = response.json().get('items', [])
                    if not items:
                        break
                    
                    for item in items:
                        if len(repos) >= config.MAX_REPOS:
                            break
                        
                        license_info = item.get('license')
                        if not license_info or license_info.get('spdx_id') not in [
                            'MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause', 'ISC'
                        ]:
                            continue
                        
                        repos.append({
                            'owner': item['owner']['login'],
                            'repo': item['name'],
                            'html_url': item['html_url'],
                            'license': license_info['spdx_id'],
                            'default_branch': item.get('default_branch', 'main')
                        })
                        pbar.update(1)
                    
                    page += 1
                    import time
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    print(f"\n Error discovering repos: {e}")
                    break
        
        return repos
    
    def extract_if_statements(self, code: str) -> List[IFStatement]:
        """Extract IF statements from Python code using tree-sitter"""
        if not PARSER or not PYTHON_LANG:
            return []
        
        try:
            tree = PARSER.parse(bytes(code, "utf8"))
            if_statements = []
            
            def visit_node(node, depth=0):
                if node.type == 'if_statement':
                    # Get the condition node
                    for child in node.children:
                        if child.type not in ['if', 'elif', ':', 'block']:
                            # This is the condition
                            condition = child.text.decode('utf8').strip()
                            line_num = node.start_point[0] + 1
                            
                            if_statements.append(IFStatement(
                                condition=condition,
                                line=line_num,
                                depth=depth
                            ))
                            break
                
                for child in node.children:
                    visit_node(child, depth + 1)
            
            visit_node(tree.root_node)
            return if_statements
            
        except Exception:
            return []
    
    def extract_functions_from_code(self, code: str) -> List[Dict]:
        """Extract function definitions with IF statements"""
        if not PARSER or not PYTHON_LANG:
            return []
        
        try:
            tree = PARSER.parse(bytes(code, "utf8"))
            functions = []
            
            # Query for function definitions
            query = PYTHON_LANG.query("(function_definition) @func")
            captures = query.captures(tree.root_node)
            
            for node, _ in captures:
                func_code = node.text.decode("utf8")
                
                # Extract IF statements from this function
                if_stmts = self.extract_if_statements(func_code)
                
                if if_stmts:
                    # Convert IFStatement objects to dicts
                    if_stmts_dict = [
                        {'condition': stmt.condition, 'line': stmt.line, 'depth': stmt.depth}
                        for stmt in if_stmts
                    ]
                    
                    functions.append({
                        'code': func_code,
                        'if_statements': if_stmts_dict,
                        'num_ifs': len(if_stmts),
                        'lines': func_code.count('\n') + 1
                    })
            
            return functions
            
        except Exception:
            return []
    
    def harvest_repo(self, repo_info: Dict) -> List[Dict]:
        """Download and parse a single repository"""
        owner = repo_info['owner']
        repo = repo_info['repo']
        branch = repo_info['default_branch']
        
        try:
            # Download repo as zipball
            zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{branch}"
            response = self.session.get(zip_url, timeout=60)
            
            if response.status_code != 200:
                return []
            
            functions = []
            
            # Extract and parse Python files
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                for file_info in zip_ref.filelist:
                    if file_info.filename.endswith('.py'):
                        try:
                            with zip_ref.open(file_info) as f:
                                code = f.read().decode('utf-8', errors='ignore')
                                funcs = self.extract_functions_from_code(code)
                                functions.extend(funcs)
                        except Exception:
                            continue
            
            return functions
            
        except Exception:
            return []
    
    def collect(self):
        """Main data collection pipeline"""
        if self.should_skip():
            return
        
        print("\n" + "="*80)
        print("STEP 1: DATA COLLECTION")
        print("="*80)
        
        # Discover repositories
        repos = self.discover_repos()
        print(f"Found {len(repos)} repositories")
        
        # Harvest functions
        all_functions = []
        print(f"\n Harvesting functions from {len(repos)} repositories...")
        
        with tqdm(total=len(repos), desc="Harvesting repos") as pbar:
            for repo in repos:
                if len(all_functions) >= config.TOTAL_FUNCTIONS_TARGET:
                    break
                
                funcs = self.harvest_repo(repo)
                all_functions.extend(funcs)
                pbar.update(1)
                pbar.set_postfix({'functions': len(all_functions)})
        
        # Add unique IDs
        for i, func in enumerate(all_functions):
            func['id'] = hashlib.sha256(f"{i}:{func['code'][:100]}".encode()).hexdigest()
        
        # Save to file
        print(f"\n Saving {len(all_functions):,} functions...")
        with open(config.RAW_FUNCTIONS_FILE, 'w') as f:
            for func in all_functions:
                f.write(json.dumps(func) + '\n')
        
        print(f" Data collection complete: {len(all_functions):,} functions")


# ===================== STEP 2: TOKENIZER TRAINING =====================
class TokenizerTrainer:
    """Trains custom BPE tokenizer from scratch"""
    
    def should_skip(self) -> bool:
        """Check if tokenizer training can be skipped"""
        if (config.TOKENIZER_DIR / "tokenizer.json").exists():
            print(" Tokenizer training: SKIPPED (already exists)")
            return True
        return False
    
    def train(self):
        """Train custom tokenizer"""
        if self.should_skip():
            return
        
        print("\n" + "="*80)
        print("STEP 2: TOKENIZER TRAINING")
        print("="*80)
        
        # Load code corpus
        print(" Loading code corpus...")
        corpus = []
        with open(config.RAW_FUNCTIONS_FILE, 'r') as f:
            for line in f:
                func = json.loads(line)
                corpus.append(func['code'])
        
        print(f" Loaded {len(corpus):,} functions")
        
        # Initialize tokenizer
        print("\n Training BPE tokenizer...")
        tokenizer = Tokenizer(models.BPE(unk_token=config.UNK_TOKEN))
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Train
        trainer = trainers.BpeTrainer(
            vocab_size=config.VOCAB_SIZE,
            special_tokens=[
                config.PAD_TOKEN,
                config.UNK_TOKEN,
                config.EOS_TOKEN,
                config.MASK_TOKEN,
            ] + [f"<extra_id_{i}>" for i in range(1, 100)],
            show_progress=True
        )
        
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        
        # Add post-processor
        tokenizer.post_processor = TemplateProcessing(
            single=f"{config.EOS_TOKEN} $A",
            special_tokens=[(config.EOS_TOKEN, tokenizer.token_to_id(config.EOS_TOKEN))],
        )
        
        # Save
        config.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(config.TOKENIZER_DIR / "tokenizer.json"))
        
        print(f" Tokenizer trained: vocab_size={config.VOCAB_SIZE}")


# ===================== STEP 3: DATA PREPROCESSING =====================
class DataPreprocessor:
    """Prepares data for pre-training and fine-tuning"""
    
    def should_skip(self) -> bool:
        """Check if preprocessing can be skipped"""
        required_files = [
            config.PRETRAIN_DATA_FILE,
            config.FINETUNE_TRAIN_FILE,
            config.FINETUNE_VAL_FILE,
            config.FINETUNE_TEST_FILE
        ]
        if all(f.exists() for f in required_files):
            print(" Data preprocessing: SKIPPED (data files exist)")
            return True
        return False
    
    def create_pretrain_examples(self, functions: List[Dict]) -> List[Dict]:
        """Create pre-training examples with span corruption"""
        examples = []
        
        for func in tqdm(functions, desc="Creating pre-training examples"):
            code = func['code']
            tokens = code.split()
            
            if len(tokens) < 10:
                continue
            
            # Randomly corrupt 15% of tokens
            num_masks = max(1, int(len(tokens) * 0.15))
            mask_positions = random.sample(range(len(tokens)), num_masks)
            mask_positions.sort()
            
            # Create input with masks
            input_tokens = tokens.copy()
            target_tokens = []
            
            for i, pos in enumerate(mask_positions):
                input_tokens[pos] = f"<extra_id_{i}>"
                target_tokens.append(f"<extra_id_{i}>")
                target_tokens.append(tokens[pos])
            
            examples.append({
                'input': ' '.join(input_tokens),
                'target': ' '.join(target_tokens)
            })
        
        return examples
    
    def create_finetune_examples(self, functions: List[Dict]) -> List[Dict]:
        """Create fine-tuning examples with IF masking - CRITICAL FIX"""
        examples = []
        
        for func in tqdm(functions, desc="Creating fine-tuning examples"):
            code = func['code']
            if_stmts = func.get('if_statements', [])
            
            if not if_stmts:
                continue
            
            # Create one example per IF statement
            for if_stmt in if_stmts:
                condition = if_stmt['condition'].strip()
                line_num = if_stmt['line']
                
                # CRITICAL FIX: Use simple line-based masking (not regex)
                lines = code.split('\n')
                if line_num - 1 >= len(lines):
                    continue
                
                target_line = lines[line_num - 1]
                
                # Find 'if' or 'elif' keyword and mask everything until ':'
                for keyword in ['elif', 'if']:  # Check elif first (longer match)
                    if keyword in target_line:
                        # Split on keyword
                        parts = target_line.split(keyword, 1)
                        if len(parts) == 2:
                            before = parts[0]
                            after = parts[1]
                            
                            # Find the colon
                            colon_idx = after.find(':')
                            if colon_idx != -1:
                                # Mask the condition
                                masked_line = before + keyword + ' ' + config.MASK_TOKEN + after[colon_idx:]
                                lines[line_num - 1] = masked_line
                                
                                masked_code = '\n'.join(lines)
                                
                                examples.append({
                                    'id': hashlib.sha256(f"{func['id']}:{line_num}".encode()).hexdigest(),
                                    'input': masked_code,
                                    'target': condition
                                })
                                break
        
        return examples
    
    def preprocess(self):
        """Main preprocessing pipeline"""
        if self.should_skip():
            return
        
        print("\n" + "="*80)
        print("STEP 3: DATA PREPROCESSING")
        print("="*80)
        
        # Load all functions
        print(" Loading functions...")
        functions = []
        with open(config.RAW_FUNCTIONS_FILE, 'r') as f:
            for line in f:
                functions.append(json.loads(line))
        
        print(f" Loaded {len(functions):,} functions")
        
        # Split data
        random.shuffle(functions)
        pretrain_funcs = functions[:config.PRETRAIN_TARGET]
        finetune_funcs = functions[config.PRETRAIN_TARGET:config.PRETRAIN_TARGET + config.FINETUNE_TARGET]
        
        print(f"\n Split: {len(pretrain_funcs):,} pre-train, {len(finetune_funcs):,} fine-tune")
        
        # Create pre-training examples
        pretrain_examples = self.create_pretrain_examples(pretrain_funcs)
        print(f" Created {len(pretrain_examples):,} pre-training examples")
        
        # Create fine-tuning examples
        finetune_examples = self.create_finetune_examples(finetune_funcs)
        print(f" Created {len(finetune_examples):,} fine-tuning examples")
        
        # Split fine-tuning data
        random.shuffle(finetune_examples)
        train_size = int(len(finetune_examples) * 0.8)
        val_size = int(len(finetune_examples) * 0.1)
        
        train_examples = finetune_examples[:train_size]
        val_examples = finetune_examples[train_size:train_size + val_size]
        test_examples = finetune_examples[train_size + val_size:]
        
        print(f"\n Fine-tuning split:")
        print(f"   Train: {len(train_examples):,}")
        print(f"   Val:   {len(val_examples):,}")
        print(f"   Test:  {len(test_examples):,}")
        
        # Save datasets
        print("\n Saving datasets...")
        
        with open(config.PRETRAIN_DATA_FILE, 'w') as f:
            for ex in pretrain_examples:
                f.write(json.dumps(ex) + '\n')
        
        with open(config.FINETUNE_TRAIN_FILE, 'w') as f:
            for ex in train_examples:
                f.write(json.dumps(ex) + '\n')
        
        with open(config.FINETUNE_VAL_FILE, 'w') as f:
            for ex in val_examples:
                f.write(json.dumps(ex) + '\n')
        
        with open(config.FINETUNE_TEST_FILE, 'w') as f:
            for ex in test_examples:
                f.write(json.dumps(ex) + '\n')
        
        print(" Preprocessing complete")


# ===================== STEP 4: MODEL PRE-TRAINING =====================
class ModelPretrainer:
    """Pre-trains T5 model from scratch"""
    
    def should_skip(self) -> bool:
        """Check if pre-training can be skipped"""
        if (config.PRETRAINED_MODEL_DIR / "pytorch_model.bin").exists():
            print(" Pre-training: SKIPPED (model exists)")
            return True
        return False
    
    def pretrain(self):
        """Pre-train T5 model"""
        if self.should_skip():
            return
        
        print("\n" + "="*80)
        print("STEP 4: MODEL PRE-TRAINING")
        print("="*80)
        
        # Load tokenizer
        print(" Loading tokenizer...")
        tokenizer = T5TokenizerFast(tokenizer_file=str(config.TOKENIZER_DIR / "tokenizer.json"))
        tokenizer.pad_token = config.PAD_TOKEN
        tokenizer.eos_token = config.EOS_TOKEN
        
        # Initialize model
        print("\n🔧 Initializing T5 model...")
        model_config = T5Config(
            vocab_size=config.VOCAB_SIZE,
            d_model=config.MODEL_DIM,
            d_ff=config.FFN_DIM,
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            decoder_start_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        model = T5ForConditionalGeneration(model_config)
        print(f" Model: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Load dataset
        print("\n Loading pre-training data...")
        examples = []
        with open(config.PRETRAIN_DATA_FILE, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
        
        print(f" Loaded {len(examples):,} examples")
        
        # Tokenize
        def tokenize_function(examples_batch):
            inputs = tokenizer(
                [ex['input'] for ex in examples_batch],
                max_length=config.MAX_LENGTH,
                truncation=True,
                padding='max_length'
            )
            targets = tokenizer(
                [ex['target'] for ex in examples_batch],
                max_length=config.MAX_LENGTH,
                truncation=True,
                padding='max_length'
            )
            inputs['labels'] = targets['input_ids']
            return inputs
        
        # Create dataset
        dataset = Dataset.from_list(examples)
        dataset = dataset.map(
            lambda x: tokenize_function([x]),
            batched=False,
            remove_columns=dataset.column_names
        )
        
        # Split for validation
        train_val_split = dataset.train_test_split(test_size=0.05)
        train_dataset = train_val_split['train']
        eval_dataset = train_val_split['test']
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(config.PRETRAINED_MODEL_DIR),
            num_train_epochs=config.PRETRAIN_EPOCHS,
            per_device_train_batch_size=config.PRETRAIN_BATCH_SIZE,
            per_device_eval_batch_size=config.PRETRAIN_BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=500,
            eval_steps=2000,
            save_steps=5000,
            evaluation_strategy="steps",
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("\n Starting pre-training...")
        trainer.train()
        
        # Save
        print("\n Saving pre-trained model...")
        trainer.save_model(str(config.PRETRAINED_MODEL_DIR))
        tokenizer.save_pretrained(str(config.PRETRAINED_MODEL_DIR))
        
        print(" Pre-training complete")


# ===================== STEP 5: MODEL FINE-TUNING =====================
class ModelFinetuner:
    """Fine-tunes model on IF prediction task"""
    
    def should_skip(self) -> bool:
        """Check if fine-tuning can be skipped"""
        if (config.FINETUNED_MODEL_DIR / "pytorch_model.bin").exists():
            print(" Fine-tuning: SKIPPED (model exists)")
            return True
        return False
    
    def finetune(self):
        """Fine-tune model"""
        if self.should_skip():
            return
        
        print("\n" + "="*80)
        print("STEP 5: MODEL FINE-TUNING")
        print("="*80)
        
        # Load pre-trained model
        print(" Loading pre-trained model...")
        model = T5ForConditionalGeneration.from_pretrained(str(config.PRETRAINED_MODEL_DIR))
        tokenizer = T5TokenizerFast.from_pretrained(str(config.PRETRAINED_MODEL_DIR))
        
        # Load datasets
        print("\n Loading fine-tuning data...")
        
        def load_dataset_file(filepath):
            examples = []
            with open(filepath, 'r') as f:
                for line in f:
                    examples.append(json.loads(line))
            return examples
        
        train_examples = load_dataset_file(config.FINETUNE_TRAIN_FILE)
        val_examples = load_dataset_file(config.FINETUNE_VAL_FILE)
        
        print(f" Train: {len(train_examples):,}, Val: {len(val_examples):,}")
        
        # Tokenize
        def tokenize_function(examples_batch):
            inputs = tokenizer(
                [ex['input'] for ex in examples_batch],
                max_length=config.MAX_LENGTH,
                truncation=True,
                padding='max_length'
            )
            targets = tokenizer(
                [ex['target'] for ex in examples_batch],
                max_length=config.MAX_LENGTH,
                truncation=True,
                padding='max_length'
            )
            inputs['labels'] = targets['input_ids']
            return inputs
        
        # Create datasets
        train_dataset = Dataset.from_list(train_examples)
        train_dataset = train_dataset.map(
            lambda x: tokenize_function([x]),
            batched=False,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = Dataset.from_list(val_examples)
        val_dataset = val_dataset.map(
            lambda x: tokenize_function([x]),
            batched=False,
            remove_columns=val_dataset.column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(config.FINETUNED_MODEL_DIR),
            num_train_epochs=config.FINETUNE_EPOCHS,
            per_device_train_batch_size=config.FINETUNE_BATCH_SIZE,
            per_device_eval_batch_size=config.FINETUNE_BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=100,
            eval_steps=500,
            save_steps=1000,
            evaluation_strategy="steps",
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("\n Starting fine-tuning...")
        trainer.train()
        
        # Save
        print("\n Saving fine-tuned model...")
        trainer.save_model(str(config.FINETUNED_MODEL_DIR))
        tokenizer.save_pretrained(str(config.FINETUNED_MODEL_DIR))
        
        print(" Fine-tuning complete")


# ===================== STEP 6: EVALUATION =====================
class ModelEvaluator:
    """Evaluates model on test sets"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load fine-tuned model"""
        if self.model is None:
            print(" Loading fine-tuned model...")
            self.model = T5ForConditionalGeneration.from_pretrained(str(config.FINETUNED_MODEL_DIR))
            self.tokenizer = T5TokenizerFast.from_pretrained(str(config.FINETUNED_MODEL_DIR))
            self.model.eval()
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
    
    def predict(self, input_text: str) -> str:
        """Generate prediction for masked IF statement"""
        inputs = self.tokenizer(
            input_text,
            max_length=config.MAX_LENGTH,
            truncation=True,
            return_tensors='pt'
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction.strip()
    
    def evaluate_dataset(self, dataset_file: Path, output_file: Path) -> Dict:
        """Evaluate on a dataset file"""
        self.load_model()
        
        # Load examples
        examples = []
        with open(dataset_file, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
        
        # Generate predictions
        results = []
        metrics = {'exact_match': 0, 'total': 0}
        
        print(f"\n Generating predictions for {len(examples):,} examples...")
        
        for ex in tqdm(examples):
            input_text = ex['input']
            target = ex['target']
            
            prediction = self.predict(input_text)
            
            is_correct = (prediction.strip().lower() == target.strip().lower())
            
            if is_correct:
                metrics['exact_match'] += 1
            metrics['total'] += 1
            
            results.append({
                'input': input_text,
                'target': target,
                'prediction': prediction,
                'correct': is_correct
            })
        
        # Calculate accuracy
        accuracy = metrics['exact_match'] / metrics['total'] if metrics['total'] > 0 else 0
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        
        print(f" Accuracy: {accuracy*100:.2f}% ({metrics['exact_match']}/{metrics['total']})")
        
        return {'accuracy': accuracy, 'exact_match': metrics['exact_match'], 'total': metrics['total']}
    
    def evaluate(self):
        """Main evaluation pipeline"""
        print("\n" + "="*80)
        print("STEP 6: MODEL EVALUATION")
        print("="*80)
        
        # Evaluate on generated test set
        print("\n Evaluating on generated test set...")
        generated_results = self.evaluate_dataset(
            config.FINETUNE_TEST_FILE,
            config.OUTPUT_DIR / "generated-testset.csv"
        )
        
        # Evaluate on provided benchmark
        if config.BENCHMARK_FILE.exists():
            print("\n Evaluating on provided benchmark...")
            
            # Load benchmark
            df_benchmark = pd.read_csv(config.BENCHMARK_FILE)
            
            # Process benchmark (extract IF statements and mask them)
            benchmark_examples = []
            
            for idx, row in df_benchmark.iterrows():
                code = row['code']
                
                # Extract IF statements
                if_stmts = DataCollector().extract_if_statements(code)
                
                if if_stmts:
                    # Take first IF statement
                    if_stmt = if_stmts[0]
                    condition = if_stmt.condition
                    line_num = if_stmt.line
                    
                    # Mask the IF statement
                    lines = code.split('\n')
                    if line_num - 1 < len(lines):
                        target_line = lines[line_num - 1]
                        
                        # Mask it
                        for keyword in ['elif', 'if']:
                            if keyword in target_line:
                                parts = target_line.split(keyword, 1)
                                if len(parts) == 2:
                                    before = parts[0]
                                    after = parts[1]
                                    colon_idx = after.find(':')
                                    if colon_idx != -1:
                                        masked_line = before + keyword + ' ' + config.MASK_TOKEN + after[colon_idx:]
                                        lines[line_num - 1] = masked_line
                                        masked_code = '\n'.join(lines)
                                        
                                        benchmark_examples.append({
                                            'input': masked_code,
                                            'target': condition
                                        })
                                        break
            
            # Save temp file
            temp_benchmark = config.DATA_DIR / "temp_benchmark.jsonl"
            with open(temp_benchmark, 'w') as f:
                for ex in benchmark_examples:
                    f.write(json.dumps(ex) + '\n')
            
            # Evaluate
            benchmark_results = self.evaluate_dataset(
                temp_benchmark,
                config.OUTPUT_DIR / "provided-testset.csv"
            )
            
            # Clean up
            temp_benchmark.unlink()
        else:
            print(" Benchmark file not found")
            benchmark_results = None
        
        # Save metrics
        all_metrics = {
            'generated_testset': generated_results,
            'provided_testset': benchmark_results
        }
        
        with open(config.OUTPUT_DIR / "metrics.json", 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print("\n Evaluation complete!")
        print(f" Results saved to {config.OUTPUT_DIR}")


# ===================== MAIN PIPELINE =====================
def main():
    """Execute complete pipeline with checkpointing"""
    
    print("\n" + "="*80)
    print("AI4SE 2025 - IF STATEMENT PREDICTION PIPELINE")
    print("="*80)
    print(f"Base directory: {config.BASE_DIR}")
    print(f"GPU available: {torch.cuda.is_available()}")
    
    try:
        # Step 1: Data Collection
        collector = DataCollector()
        collector.collect()
        
        # Step 2: Tokenizer Training
        tokenizer_trainer = TokenizerTrainer()
        tokenizer_trainer.train()
        
        # Step 3: Data Preprocessing
        preprocessor = DataPreprocessor()
        preprocessor.preprocess()
        
        # Step 4: Pre-training
        pretrainer = ModelPretrainer()
        pretrainer.pretrain()
        
        # Step 5: Fine-tuning
        finetuner = ModelFinetuner()
        finetuner.finetune()
        
        # Step 6: Evaluation
        evaluator = ModelEvaluator()
        evaluator.evaluate()
        
        print("\n" + "="*80)
        print(" PIPELINE COMPLETE!")
        print("="*80)
        print(f"\nResults available in: {config.OUTPUT_DIR}")
        print(f"  - generated-testset.csv")
        print(f"  - provided-testset.csv")
        print(f"  - metrics.json")
        
    except KeyboardInterrupt:
        print("\n\n Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
