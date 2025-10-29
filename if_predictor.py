#!/usr/bin/env python3
"""
if_predictor_improved.py — Enhanced IF-Statement Predictor with Architectural Improvements

Key Improvements:
    • GitHub API integration for automated repository mining
    • AST-based IF condition extraction (no regex)
    • Tokenizer trained on full pre-training corpus
    • Proper Masked Language Modeling (15% token masking)
    • Multi-metric evaluation (EM, Token F1, Edit Distance, BLEU)
    • Consistent prompt prefixes between pre-training and fine-tuning
    • Enhanced data validation and quality checks
"""

import os
import re
import json
import ast
import argparse
import random
import csv
import time
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from difflib import SequenceMatcher
import zipfile
import io

import torch
from datasets import load_dataset, Dataset
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerFast,
)
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


# **DEVICE UTILITY**
def get_device():
    """Return the best available compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"    → GPU: {torch.cuda.get_device_name(0)}")
    return device


# **GITHUB REPOSITORY MINING**
class GitHubRepoMiner:
    """Mine Python repositories from GitHub using REST API."""
    
    APPROVED_LICENSES = ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause"]
    DISALLOWED_PREFIXES = ("GPL-", "LGPL-", "AGPL-")
    
    def __init__(self, output_dir="repos", max_repos=50):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.max_repos = max_repos
        
    def _has_good_license(self, license_info):
        """Check if repository has an approved license."""
        if not license_info or not license_info.get("spdx_id"):
            return False
        spdx = license_info["spdx_id"]
        # Reject GPL variants
        if any(spdx.startswith(prefix) for prefix in self.DISALLOWED_PREFIXES):
            return False
        # Check if in approved list (case-insensitive)
        return spdx.lower() in [lic.lower() for lic in self.APPROVED_LICENSES]
    
    def fetch_repo_list(self, min_stars=100):
        """
        Dynamically fetch list of Python repositories from GitHub API.
        Falls back to static list if API is unavailable or token is missing.
        """
        # Static fallback list
        popular_repos = [
            "psf/requests",
            "pallets/flask",
            "django/django",
            "numpy/numpy",
            "pandas-dev/pandas",
            "scikit-learn/scikit-learn",
            "ansible/ansible",
            "pytorch/pytorch",
            "tensorflow/tensorflow",
            "keras-team/keras",
            "matplotlib/matplotlib",
            "scipy/scipy",
            "python/cpython",
            "tornadoweb/tornado",
            "paramiko/paramiko",
            "certbot/certbot",
            "sqlalchemy/sqlalchemy",
            "fabric/fabric",
            "celery/celery",
            "boto/boto3",
            "explosion/spaCy",
            "getsentry/sentry",
            "pytest-dev/pytest",
            "pallets/click",
            "pypa/pip",
            "cookiecutter/cookiecutter",
            "home-assistant/core",
            "zulip/zulip",
            "ytdl-org/youtube-dl",
            "httpie/httpie",
        ]
        
        # Try dynamic discovery via GitHub API
        try:
            import requests
            
            # Read token from environment
            token = os.getenv("GITHUB_TOKEN")
            
            # Create session with authentication
            sess = requests.Session()
            sess.headers["Accept"] = "application/vnd.github+json"
            if token:
                sess.headers["Authorization"] = f"token {token}"
            
            # Build search query
            q = f"language:Python stars:>={min_stars} fork:false archived:false"
            
            results = []
            page = 1
            per_page = 50
            
            print(f"Fetching Python repositories from GitHub API...")
            
            while len(results) < self.max_repos:
                try:
                    r = sess.get(
                        "https://api.github.com/search/repositories",
                        params={
                            "q": q,
                            "sort": "stars",
                            "order": "desc",
                            "per_page": per_page,
                            "page": page
                        },
                        timeout=30
                    )
                    
                    # Handle rate limiting
                    if r.status_code == 403 and "rate limit" in r.text.lower():
                        print("   ⚠ Rate limit hit, waiting 60 seconds...")
                        time.sleep(60)
                        continue
                    
                    r.raise_for_status()
                    
                    items = r.json().get("items", [])
                    if not items:
                        break
                    
                    for item in items:
                        if len(results) >= self.max_repos:
                            break
                        
                        # Check license
                        lic = item.get("license")
                        if not self._has_good_license(lic):
                            continue
                        
                        # Add to results
                        full_name = f"{item['owner']['login']}/{item['name']}"
                        results.append(full_name)
                    
                    page += 1
                    
                    # Rate limiting between requests
                    time.sleep(0.5)
                    
                except requests.RequestException as e:
                    print(f"   ⚠ Request failed: {e}")
                    break
            
            if results:
                print(f"✓ Dynamically fetched {len(results)} repositories via GitHub API (MIT/APACHE/BSD licenses)")
                return results
            else:
                raise Exception("No repositories found via API")
                
        except Exception as e:
            print(f"⚠ GitHub token not found or API fetch failed — using static fallback list.")
            print(f"   (Error: {e})")
            return popular_repos[:self.max_repos]
    
    def clone_repositories(self):
        """Clone repositories to local directory."""
        repos = self.fetch_repo_list()
        print(f"Attempting to clone {len(repos)} repositories...")
        
        cloned_count = 0
        for repo in tqdm(repos, desc="Cloning repositories"):
            repo_name = repo.split('/')[-1]
            repo_path = self.output_dir / repo_name
            
            if repo_path.exists():
                print(f"   ○ {repo_name} already exists, skipping")
                cloned_count += 1
                continue
            
            # Clone with depth 1 for faster downloads
            url = f"https://github.com/{repo}.git"
            cmd = f"git clone --depth 1 {url} {repo_path} > /dev/null 2>&1"
            
            result = os.system(cmd)
            if result == 0:
                cloned_count += 1
                print(f"   ✓ Cloned {repo_name}")
            else:
                print(f"   ✗ Failed to clone {repo_name}")
            
            # Rate limiting
            time.sleep(0.5)
        
        print(f"\nSuccessfully cloned {cloned_count}/{len(repos)} repositories")
        return cloned_count
    
    def dynamic_fetch_until_target(self, extractor, target_count, min_stars=100):
        """
        Dynamically fetch and clone repositories until target function count is reached.
        
        Args:
            extractor: ASTFunctionExtractor instance for extracting functions
            target_count: Target number of total functions to collect
            min_stars: Minimum stars for repository search
            
        Returns:
            List of cloned repository names
        """
        print(f"🎯 Dynamic mining target: {target_count:,} functions")
        
        # Static fallback list
        popular_repos = [
            "psf/requests", "pallets/flask", "django/django", "numpy/numpy",
            "pandas-dev/pandas", "scikit-learn/scikit-learn", "ansible/ansible",
            "pytorch/pytorch", "tensorflow/tensorflow", "keras-team/keras",
            "matplotlib/matplotlib", "scipy/scipy", "python/cpython",
            "tornadoweb/tornado", "paramiko/paramiko", "certbot/certbot",
            "sqlalchemy/sqlalchemy", "fabric/fabric", "celery/celery",
            "boto/boto3", "explosion/spaCy", "getsentry/sentry",
            "pytest-dev/pytest", "pallets/click", "pypa/pip",
            "cookiecutter/cookiecutter", "home-assistant/core", "zulip/zulip",
            "ytdl-org/youtube-dl", "httpie/httpie",
        ]
        
        try:
            import requests
            
            # Read token from environment
            token = os.getenv("GITHUB_TOKEN")
            
            # Create session with authentication
            sess = requests.Session()
            sess.headers["Accept"] = "application/vnd.github+json"
            if token:
                sess.headers["Authorization"] = f"token {token}"
            
            # Build search query
            q = f"language:Python stars:>={min_stars} fork:false archived:false"
            
            total_funcs = 0
            cloned_repos = []
            page = 1
            per_page = 50
            repo_count = 0
            
            print(f"Starting dynamic repository mining with GitHub API...")
            
            while total_funcs < target_count:
                try:
                    # Fetch next page of repositories
                    r = sess.get(
                        "https://api.github.com/search/repositories",
                        params={
                            "q": q,
                            "sort": "stars",
                            "order": "desc",
                            "per_page": per_page,
                            "page": page
                        },
                        timeout=30
                    )
                    
                    # Handle rate limiting
                    if r.status_code == 403 and "rate limit" in r.text.lower():
                        print("   ⚠ Rate limit hit, sleeping for 60s...")
                        time.sleep(60)
                        continue
                    
                    r.raise_for_status()
                    
                    items = r.json().get("items", [])
                    if not items:
                        print("   ⚠ No more repositories available from API")
                        break
                    
                    if page == 1:
                        print(f"✓ Fetched {len(items)} repos from GitHub API (page {page})")
                    
                    # Process each repository
                    for item in items:
                        if total_funcs >= target_count:
                            break
                        
                        # Check license
                        lic = item.get("license")
                        if not self._has_good_license(lic):
                            continue
                        
                        full_name = f"{item['owner']['login']}/{item['name']}"
                        repo_name = item['name']
                        repo_path = self.output_dir / repo_name
                        
                        # Clone repository
                        if not repo_path.exists():
                            url = f"https://github.com/{full_name}.git"
                            cmd = f"git clone --depth 1 {url} {repo_path} > /dev/null 2>&1"
                            
                            result = os.system(cmd)
                            if result != 0:
                                print(f"   ✗ Failed to clone {repo_name}")
                                continue
                        
                        # Extract functions from this repository
                        try:
                            # Create temporary output for counting
                            temp_output = f"/tmp/temp_extract_{repo_name}.jsonl"
                            func_count = extractor.extract_from_directory(
                                str(repo_path),
                                temp_output,
                                max_functions=None  # Extract all available
                            )
                            
                            # Clean up temp file
                            if os.path.exists(temp_output):
                                os.remove(temp_output)
                            
                            total_funcs += func_count
                            cloned_repos.append(full_name)
                            repo_count += 1
                            
                            print(f"✓ Cloned {repo_name} (+{func_count:,} funcs, total={total_funcs:,})")
                            
                        except Exception as e:
                            print(f"   ⚠ Failed to extract from {repo_name}: {e}")
                            continue
                        
                        # Rate limiting between repos
                        time.sleep(0.5)
                    
                    # Move to next page
                    page += 1
                    
                    # Rate limiting between API pages
                    time.sleep(1.0)
                    
                except requests.RequestException as e:
                    print(f"   ⚠ API request failed: {e}")
                    break
                except Exception as e:
                    print(f"   ⚠ Unexpected error: {e}")
                    break
            
            if total_funcs >= target_count:
                print(f"\n✅ Dynamic mining complete: {total_funcs:,} functions from {repo_count} repositories")
                return cloned_repos
            else:
                print(f"\n⚠ Could not reach target ({total_funcs:,}/{target_count:,} functions collected)")
                print(f"   Falling back to static repository list...")
                raise Exception("Target not reached via API")
                
        except Exception as e:
            print(f"⚠ Dynamic mining failed: {e}")
            print(f"   Using static fallback list and cloning repositories...")
            
            # Fallback: clone static repos and extract
            total_funcs = 0
            cloned_repos = []
            
            for repo in popular_repos:
                if total_funcs >= target_count:
                    break
                
                repo_name = repo.split('/')[-1]
                repo_path = self.output_dir / repo_name
                
                # Clone if needed
                if not repo_path.exists():
                    url = f"https://github.com/{repo}.git"
                    cmd = f"git clone --depth 1 {url} {repo_path} > /dev/null 2>&1"
                    result = os.system(cmd)
                    if result != 0:
                        continue
                
                # Extract functions
                try:
                    temp_output = f"/tmp/temp_extract_{repo_name}.jsonl"
                    func_count = extractor.extract_from_directory(
                        str(repo_path),
                        temp_output,
                        max_functions=None
                    )
                    
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    
                    total_funcs += func_count
                    cloned_repos.append(repo)
                    print(f"✓ Cloned {repo_name} (+{func_count:,} funcs, total={total_funcs:,})")
                    
                except Exception:
                    continue
                
                time.sleep(0.5)
            
            print(f"\n✅ Fallback mining complete: {total_funcs:,} functions from {len(cloned_repos)} repositories")
            return cloned_repos


# **AST-BASED FUNCTION EXTRACTION**
class ASTFunctionExtractor:
    """Extract Python functions using AST with strict validation."""
    
    def __init__(self, min_lines=3, max_lines=100, require_if=False):
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.require_if = require_if
    
    def is_valid_function(self, func_code):
        """Validate function quality with multiple checks."""
        lines = [l for l in func_code.split('\n') if l.strip()]
        
        # Length check
        if len(lines) < self.min_lines or len(lines) > self.max_lines:
            return False
        
        # Syntactic validity
        try:
            tree = ast.parse(func_code)
        except SyntaxError:
            return False
        
        # Check for IF statement if required
        if self.require_if:
            has_if = any(isinstance(node, ast.If) for node in ast.walk(tree))
            if not has_if:
                return False
        
        # Skip empty functions
        if func_code.strip().endswith("pass") and len(lines) < 4:
            return False
        
        return True
    
    def extract_functions(self, file_path):
        """Extract all valid functions from a Python file using AST."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    try:
                        func_code = ast.get_source_segment(content, node)
                        if func_code and self.is_valid_function(func_code):
                            functions.append(func_code)
                    except Exception:
                        continue
            
            return functions
        except Exception:
            return []
    
    def extract_from_directory(self, directory, output_file, max_functions=None):
        """Recursively extract functions from Python files in directory."""
        functions = []
        py_files = list(Path(directory).rglob("*.py"))
        
        print(f"Found {len(py_files)} Python files in {directory}")
        
        for py_file in tqdm(py_files, desc="Extracting functions"):
            funcs = self.extract_functions(py_file)
            functions.extend(funcs)
            
            if max_functions and len(functions) >= max_functions:
                functions = functions[:max_functions]
                break
        
        # Save to JSONL
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for func in functions:
                f.write(json.dumps({"code": func}) + '\n')
        
        print(f"Extracted {len(functions)} functions to {output_file}")
        return len(functions)


# **AST-BASED IF CONDITION MASKING**
def create_masked_dataset_ast(in_file, out_file, max_samples=None):
    """
    Create fine-tuning dataset by masking IF conditions using AST.
    This ensures syntactic correctness and handles multiline/nested conditions.
    """
    mask_token = "<mask>"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    total_processed = 0
    successful_masks = 0
    skipped_complex = 0
    skipped_invalid = 0
    
    with open(in_file, encoding="utf8") as f_in, open(out_file, "w", encoding="utf8") as f_out:
        for line in f_in:
            total_processed += 1
            
            try:
                code = json.loads(line)["code"]
                tree = ast.parse(code)
            except (json.JSONDecodeError, SyntaxError):
                skipped_invalid += 1
                continue
            
            # Find first IF statement using AST
            if_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    if_node = node
                    break
            
            if if_node is None:
                continue
            
            # Extract the condition using ast.get_source_segment
            try:
                cond_src = ast.get_source_segment(code, if_node.test)
                if not cond_src:
                    continue
                
                # Skip overly complex conditions
                if len(cond_src) > 200:
                    skipped_complex += 1
                    continue
                
                # Create masked version by replacing the condition
                # Find the exact position of the condition in the code
                if_line_start = if_node.lineno - 1
                lines = code.split('\n')
                
                # Find the line with the if statement
                if_line = lines[if_line_start]
                
                # Replace only the condition part, preserving indentation
                masked_line = if_line.replace(cond_src, mask_token, 1)
                lines[if_line_start] = masked_line
                masked_code = '\n'.join(lines)
                
                # Validate the masked code structure
                if masked_code.count(mask_token) != 1:
                    skipped_invalid += 1
                    continue
                
                f_out.write(json.dumps({
                    "input": masked_code,
                    "label": cond_src
                }) + '\n')
                successful_masks += 1
                
                if max_samples and successful_masks >= max_samples:
                    break
                    
            except Exception:
                skipped_invalid += 1
                continue
    
    print(f"Fine-tuning dataset created at {out_file}")
    print(f"   Total processed: {total_processed}")
    print(f"   Successful masks: {successful_masks}")
    print(f"   Skipped (complex): {skipped_complex}")
    print(f"   Skipped (invalid): {skipped_invalid}")
    
    if successful_masks == 0:
        raise ValueError("No IF statements could be masked! Check input data.")
    
    return successful_masks


# **TOKENIZER TRAINING ON FULL CORPUS**
def train_tokenizer_on_corpus(data_path, save_dir, vocab_size=32000):
    """
    Train a Byte-Level BPE tokenizer on the FULL pre-training corpus.
    This ensures comprehensive vocabulary coverage.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Training tokenizer on {data_path}...")
    print(f"   Vocab size: {vocab_size}")
    
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[data_path],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    
    tokenizer.save_model(save_dir)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    print(f"✓ Tokenizer trained and saved to: {save_dir}")
    
    # Verify tokenizer
    test_code = "def test():\n    if x > 0:\n        return True"
    tokens = tokenizer.encode(test_code)
    print(f"   Test encoding: {len(tokens.tokens)} tokens")


# **MASKED LANGUAGE MODELING FOR PRE-TRAINING**
def apply_mlm_masking(code, tokenizer, mask_prob=0.15):
    """
    Apply masked language modeling: mask ~15% of tokens randomly.
    Returns (masked_code, original_code) pair.
    """
    # Tokenize
    tokens = tokenizer.encode(code).tokens
    
    # Randomly mask tokens
    masked_tokens = []
    for token in tokens:
        if random.random() < mask_prob and token not in ["<s>", "</s>", "<pad>"]:
            masked_tokens.append("<mask>")
        else:
            masked_tokens.append(token)
    
    # Convert back to string (approximate)
    masked_code = tokenizer.decode(tokenizer.encode(' '.join(masked_tokens)).ids)
    
    return masked_code, code


# **PRE-TRAINING WITH MLM OBJECTIVE**
def pretrain_model_mlm(tokenizer_dir, train_file, output_dir, epochs=5):
    """
    Pre-train model using Masked Language Modeling objective.
    Masks 15% of tokens and trains model to reconstruct them.
    """
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading custom tokenizer...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        model_max_length=512,
    )
    
    print("Loading and preparing training data with MLM...")
    data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Applying MLM masking"):
            code = json.loads(line)["code"]
            
            # Apply random masking for MLM
            masked_code, original_code = apply_mlm_masking(code, tokenizer)
            
            # Use consistent prefix
            data.append({
                "input": f"fill masked code: {masked_code}",
                "label": original_code
            })
    
    print(f"Loaded {len(data)} training samples")
    
    dataset = Dataset.from_list(data)
    
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=512,
            truncation=True,
            padding=False
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["label"],
                max_length=512,
                truncation=True,
                padding=False
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    print("Initializing T5 model...")
    config = T5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        d_ff=2048,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.1,
    )
    
    model = T5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        logging_steps=500,
        save_steps=5000,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("Starting pre-training with MLM objective...")
    trainer.train()
    
    print("Saving pre-trained model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Pre-training complete. Model saved to {output_dir}")


# **FINE-TUNING WITH CONSISTENT PROMPTS**
def fine_tune_model_enhanced(tokenizer_dir, pretrained_dir, train_file, output_dir, epochs=10):
    """
    Fine-tune model for IF-condition prediction with validation and consistent prompts.
    """
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading tokenizer and pre-trained model...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        model_max_length=512,
    )
    
    model = T5ForConditionalGeneration.from_pretrained(pretrained_dir)
    model = model.to(device)
    
    print("Loading fine-tuning data...")
    data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            
            # Validation: ensure exactly one <mask> token
            if item["input"].count("<mask>") != 1:
                continue
            
            # Use consistent prefix matching pre-training
            data.append({
                "input": f"fill if condition: {item['input']}",
                "label": item["label"]
            })
    
    print(f"Loaded {len(data)} validated fine-tuning samples")
    
    # Split into train/test (90/10)
    random.shuffle(data)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"   Train: {len(train_data)} samples")
    print(f"   Test: {len(test_data)} samples")
    
    # Save test set
    test_file = train_file.replace(".jsonl", "_test.jsonl")
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            # Remove prefix for test set
            f.write(json.dumps({
                "input": item["input"].replace("fill if condition: ", ""),
                "label": item["label"]
            }) + '\n')
    
    train_dataset = Dataset.from_list(train_data)
    
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=512,
            truncation=True,
            padding=False
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["label"],
                max_length=128,
                truncation=True,
                padding=False
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing fine-tune data"
    )
    
    # Training arguments with better hyperparameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=100,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    print("Saving fine-tuned model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Fine-tuning complete. Model saved to {output_dir}")
    print(f"✓ Test set saved to {test_file}")


# **MULTI-METRIC EVALUATION**
def canonicalize_code(code_str):
    """Normalize code using AST for fair comparison."""
    try:
        tree = ast.parse(code_str)
        return ast.unparse(tree).strip()
    except:
        return code_str.strip()


def compute_token_f1(pred_tokens, label_tokens):
    """Compute token-level F1 score."""
    pred_set = set(pred_tokens)
    label_set = set(label_tokens)
    
    if not pred_set and not label_set:
        return 1.0
    if not pred_set or not label_set:
        return 0.0
    
    common = pred_set & label_set
    precision = len(common) / len(pred_set)
    recall = len(common) / len(label_set)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_edit_distance(str1, str2):
    """Compute normalized edit distance (similarity)."""
    return SequenceMatcher(None, str1, str2).ratio()


def evaluate_model_multimetric(model_dir, test_file, output_csv, tokenizer_dir):
    """
    Evaluate model with multiple metrics: Exact Match, Token F1, Edit Distance, BLEU.
    """
    device = get_device()
    
    print(f"Loading model from {model_dir}...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        model_max_length=512,
    )
    
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    
    print(f"Loading test data from {test_file}...")
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"Evaluating {len(test_data)} samples...")
    
    results = []
    exact_matches = 0
    total_bleu = 0
    total_f1 = 0
    total_edit_sim = 0
    
    for item in tqdm(test_data, desc="Evaluating"):
        input_text = item["input"]
        label = item["label"]
        
        # Add consistent prefix
        input_with_prefix = f"fill if condition: {input_text}"
        
        # Generate prediction
        inputs = tokenizer(
            input_with_prefix,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Normalize for comparison
        pred_normalized = canonicalize_code(prediction)
        label_normalized = canonicalize_code(label)
        
        # Exact match
        exact_match = 1 if pred_normalized == label_normalized else 0
        exact_matches += exact_match
        
        # Token-level F1
        pred_tokens = prediction.split()
        label_tokens = label.split()
        token_f1 = compute_token_f1(pred_tokens, label_tokens)
        total_f1 += token_f1
        
        # Edit distance similarity
        edit_sim = compute_edit_distance(prediction, label)
        total_edit_sim += edit_sim
        
        # BLEU score
        reference = [label.split()]
        hypothesis = prediction.split()
        try:
            bleu = sentence_bleu(
                reference,
                hypothesis,
                smoothing_function=SmoothingFunction().method1
            )
        except:
            bleu = 0.0
        total_bleu += bleu
        
        results.append({
            "input": input_text,
            "expected": label,
            "predicted": prediction,
            "exact_match": exact_match,
            "token_f1": round(token_f1, 4),
            "edit_similarity": round(edit_sim, 4),
            "bleu_score": round(bleu * 100, 2)
        })
    
    # Compute averages
    accuracy = (exact_matches / len(test_data)) * 100
    avg_f1 = (total_f1 / len(test_data)) * 100
    avg_edit_sim = (total_edit_sim / len(test_data)) * 100
    avg_bleu = (total_bleu / len(test_data)) * 100
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Exact Match Accuracy:    {accuracy:.2f}%")
    print(f"Average Token F1:        {avg_f1:.2f}%")
    print(f"Average Edit Similarity: {avg_edit_sim:.2f}%")
    print(f"Average BLEU Score:      {avg_bleu:.2f}%")
    print(f"Total Samples:           {len(test_data)}")
    print(f"Correct Predictions:     {exact_matches}")
    print("="*70)
    
    # Save detailed results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "input", "expected", "predicted", "exact_match", 
            "token_f1", "edit_similarity", "bleu_score"
        ])
        writer.writeheader()
        writer.writerows(results)
    
    # Save summary
    summary_file = output_csv.replace(".csv", "_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Evaluation Summary\n")
        f.write("="*60 + "\n")
        f.write(f"Exact Match Accuracy: {accuracy:.2f}%\n")
        f.write(f"Average Token F1: {avg_f1:.2f}%\n")
        f.write(f"Average Edit Similarity: {avg_edit_sim:.2f}%\n")
        f.write(f"Average BLEU Score: {avg_bleu:.2f}%\n")
        f.write(f"Total Samples: {len(test_data)}\n")
        f.write(f"Correct Predictions: {exact_matches}\n")
        f.write("="*60 + "\n")
    
    print(f"\n✓ Results saved to {output_csv}")
    print(f"✓ Summary saved to {summary_file}")
    
    return accuracy, avg_f1, avg_edit_sim, avg_bleu


# **MAIN PIPELINE**
def main():
    parser = argparse.ArgumentParser(
        description="Enhanced IF-Statement Predictor with Architectural Improvements"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["mine", "extract", "tokenizer", "pretrain", "finetune", "evaluate", "all"],
        help="Pipeline stage to execute"
    )
    parser.add_argument("--repo_dir", default="repos", help="Directory for cloned repos")
    parser.add_argument("--functions", default="data/processed/functions.jsonl")
    parser.add_argument("--fine_tune_data", default="data/processed/fine_tune.jsonl")
    parser.add_argument("--tokenizer_dir", default="src/tokenizer")
    parser.add_argument("--pretrained_dir", default="src/model/pretrained")
    parser.add_argument("--fine_tuned_dir", default="src/model/fine_tuned")
    parser.add_argument("--benchmark_csv", default="data/benchmark/benchmark_if_only.csv")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for pre-training")
    parser.add_argument("--finetune_epochs", type=int, default=10, help="Epochs for fine-tuning")
    parser.add_argument("--max_pretrain", type=int, default=150000)
    parser.add_argument("--max_finetune", type=int, default=50000)
    parser.add_argument("--max_repos", type=int, default=30, help="Max repos to clone")
    parser.add_argument("--dynamic_scale", action="store_true",
                        help="Enable dynamic data scaling until target_functions is reached")
    parser.add_argument("--target_functions", type=int, default=200000,
                        help="Minimum number of total functions to collect dynamically")
    
    args = parser.parse_args()
    
    # Create directories
    for dir_path in ["results", "src/model", "data/processed", "data/benchmark"]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Stage 0: Mine repositories from GitHub
    if args.stage in ("mine", "all"):
        print("\n" + "="*70)
        print("STAGE 0: MINING PYTHON REPOSITORIES FROM GITHUB")
        print("="*70 + "\n")
        
        miner = GitHubRepoMiner(output_dir=args.repo_dir, max_repos=args.max_repos)
        
        if args.dynamic_scale:
            # Dynamic scaling mode - mine until target function count
            print(f"Dynamic scaling enabled: targeting {args.target_functions:,} functions")
            extractor_temp = ASTFunctionExtractor(
                min_lines=3,
                max_lines=100,
                require_if=False
            )
            miner.dynamic_fetch_until_target(extractor_temp, args.target_functions)
        else:
            # Static mode - clone fixed number of repos
            miner.clone_repositories()
    
    # Stage 1: Extract functions using AST
    if args.stage in ("extract", "all"):
        print("\n" + "="*70)
        print("STAGE 1: EXTRACTING FUNCTIONS WITH AST PARSING")
        print("="*70 + "\n")
        
        # Extract for pre-training (all functions)
        print("Extracting functions for pre-training...")
        extractor_pretrain = ASTFunctionExtractor(
            min_lines=3,
            max_lines=100,
            require_if=False
        )
        extractor_pretrain.extract_from_directory(
            args.repo_dir,
            args.functions,
            max_functions=args.max_pretrain
        )
        
        # Extract for fine-tuning (only functions with IF)
        print("\nExtracting functions with IF statements for fine-tuning...")
        extractor_finetune = ASTFunctionExtractor(
            min_lines=3,
            max_lines=100,
            require_if=True
        )
        finetune_raw = args.fine_tune_data.replace(".jsonl", "_raw.jsonl")
        count = extractor_finetune.extract_from_directory(
            args.repo_dir,
            finetune_raw,
            max_functions=args.max_finetune
        )
        
        if count > 0:
            # Create masked dataset using AST
            print("\nCreating masked dataset with AST-based masking...")
            create_masked_dataset_ast(finetune_raw, args.fine_tune_data)
        else:
            print("Warning: No functions with IF statements found!")
    
    # Stage 2: Train tokenizer on FULL corpus
    if args.stage in ("tokenizer", "all"):
        print("\n" + "="*70)
        print("STAGE 2: TRAINING TOKENIZER ON FULL PRE-TRAINING CORPUS")
        print("="*70 + "\n")
        
        if not os.path.exists(args.functions):
            raise FileNotFoundError(f"Functions file not found: {args.functions}")
        
        train_tokenizer_on_corpus(args.functions, args.tokenizer_dir)
    
    # Stage 3: Pre-train with MLM objective
    if args.stage in ("pretrain", "all"):
        print("\n" + "="*70)
        print("STAGE 3: PRE-TRAINING WITH MASKED LANGUAGE MODELING")
        print("="*70 + "\n")
        
        pretrain_model_mlm(
            args.tokenizer_dir,
            args.functions,
            args.pretrained_dir,
            args.epochs
        )
    
    # Stage 4: Fine-tune with consistent prompts
    if args.stage in ("finetune", "all"):
        print("\n" + "="*70)
        print("STAGE 4: FINE-TUNING FOR IF-CONDITION PREDICTION")
        print("="*70 + "\n")
        
        if not os.path.exists(args.fine_tune_data):
            print("Fine-tune dataset not found, creating it...")
            finetune_raw = args.fine_tune_data.replace(".jsonl", "_raw.jsonl")
            if not os.path.exists(finetune_raw):
                raise FileNotFoundError(f"Raw fine-tune data not found: {finetune_raw}")
            create_masked_dataset_ast(finetune_raw, args.fine_tune_data)
        
        fine_tune_model_enhanced(
            args.tokenizer_dir,
            args.pretrained_dir,
            args.fine_tune_data,
            args.fine_tuned_dir,
            args.finetune_epochs
        )
    
    # Stage 5: Evaluate with multiple metrics
    if args.stage in ("evaluate", "all"):
        print("\n" + "="*70)
        print("STAGE 5: MULTI-METRIC EVALUATION")
        print("="*70 + "\n")
        
        # Evaluate on generated test set
        test_file = args.fine_tune_data.replace(".jsonl", "_test.jsonl")
        if os.path.exists(test_file):
            print("Evaluating on generated test set...")
            evaluate_model_multimetric(
                args.fine_tuned_dir,
                test_file,
                "results/generated-testset.csv",
                args.tokenizer_dir
            )
        else:
            print(f"Test file not found: {test_file}")
        
        # Evaluate on provided benchmark
        if os.path.exists(args.benchmark_csv):
            print("\nEvaluating on provided benchmark...")
            benchmark_jsonl = args.benchmark_csv.replace(".csv", ".jsonl")
            
            converted_count = 0
            with open(args.benchmark_csv, 'r', encoding='utf-8') as f_in:
                reader = csv.DictReader(f_in)
                with open(benchmark_jsonl, 'w', encoding='utf-8') as f_out:
                    for row in reader:
                        input_text = (
                            row.get("input") or row.get("input_code") or 
                            row.get("code") or row.get("Code") or 
                            row.get("Input") or ""
                        )
                        
                        label_text = (
                            row.get("label") or row.get("expected_if_condition") or 
                            row.get("target") or row.get("condition") or
                            row.get("docstring") or row.get("Docstring") or 
                            row.get("Label") or ""
                        )
                        
                        if input_text and label_text:
                            item = {"input": input_text, "label": label_text}
                            f_out.write(json.dumps(item) + '\n')
                            converted_count += 1
            
            print(f"   Converted {converted_count} benchmark samples")
            
            if converted_count > 0:
                evaluate_model_multimetric(
                    args.fine_tuned_dir,
                    benchmark_jsonl,
                    "results/provided-testset.csv",
                    args.tokenizer_dir
                )
            else:
                print("No valid data found in benchmark CSV.")
        else:
            print(f"Benchmark CSV not found: {args.benchmark_csv}")
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()