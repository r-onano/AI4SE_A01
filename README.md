# AI4SE Assignment 1: IF Statement Predictor

## Project Overview

This project trains a Transformer model from scratch to predict IF statement conditions in Python code. The model uses a two-stage training approach: pre-training on general Python code, then fine-tuning specifically for IF condition prediction.

## What This Does

Given a Python function with a masked IF condition like this:

```python
def calculate_discount(price, customer_type):
    base = 0.1
    if <extra_id_0>:
        base = 0.2
    return price * base
```

The model predicts: `customer_type == "premium"`

## System Requirements

- **Python**: 3.10 or higher
- **GPU**: Recommended (NVIDIA GPU with CUDA support)
  - Training will work on CPU but will be extremely slow
  - Recommended: 16GB+ VRAM for full training
- **RAM**: 32GB+ recommended
- **Disk Space**: 20GB+ for data and models
- **OS**: Linux (tested on Ubuntu 24.04)

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/r-onano/AI4SE_A01.git
cd AI4SE_A01
```

### Step 2: Create Python Virtual Environment

We use a virtual environment to keep dependencies isolated and avoid conflicts with your system Python packages.

```bash
# Create virtual environment named 'venv'
python3.10 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Your prompt should now show (venv) at the beginning
```

### Step 3: Install Dependencies

With the virtual environment activated, install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- Transformers (model architecture)
- tree-sitter (code parsing)
- tokenizers (custom tokenization)
- pandas, numpy (data processing)
- And other required dependencies

**Note**: Installation may take 5-10 minutes depending on the internet connection.

### Step 4: Verify Installation

Check that everything installed correctly:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import transformers; print('Transformers version:', transformers.__version__)"
python -c "import tree_sitter_languages; print('Tree-sitter: OK')"
```

If all three commands print version numbers without errors, you're ready to go!

## Project Structure

```
AI4SE_A01/
├── main_script.py          # Complete training pipeline
├── requirements.txt        # Python dependencies
├── quick_start.sh         # Automated setup script
├── README.md              # This file
├── data/                  # Created during execution
│   ├── raw_functions.jsonl
│   ├── pretrain_data.jsonl
│   ├── finetune_train.jsonl
│   ├── finetune_val.jsonl
│   ├── finetune_test.jsonl
│   └── benchmark_if_only_2_.csv  # Place your benchmark here
├── models/                # Created during execution
│   ├── tokenizer/
│   ├── pretrained/
│   └── finetuned/
└── output/               # Created during execution
    ├── generated-testset.csv
    ├── provided-testset.csv
    └── metrics.json
```

## Running the Pipeline

### Option 1: Automated Setup (Recommended)

The quick start script handles environment setup and execution:

```bash
# Make script executable
chmod +x quick_start.sh

# Run the setup wizard
./quick_start.sh
```

Follow the on-screen prompts to:
1. Verify environment setup
2. Check for existing progress
3. Choose to start fresh or resume from checkpoint

### Option 2: Manual Execution

If you prefer to run the pipeline manually:

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Run the complete pipeline
python main_script.py
```

**For long training runs**, use `nohup` to prevent interruption:

```bash
nohup python main_script.py > pipeline.log 2>&1 &
echo $! > pipeline.pid

# Monitor progress
tail -f pipeline.log

# Check if still running
ps -p $(cat pipeline.pid)
```

## Pipeline Stages and Timing

The pipeline runs in 6 stages with automatic checkpointing:

### Stage 1: Data Collection (~2-4 hours)
- Discovers Python repositories on GitHub
- Downloads and parses Python files
- Extracts functions containing IF statements
- Target: 500,000 functions
- **Checkpoint**: `data/raw_functions.jsonl`

### Stage 2: Tokenizer Training (~5-10 minutes)
- Trains a custom BPE tokenizer from scratch
- Learns code-specific vocabulary (32,000 tokens)
- **Checkpoint**: `models/tokenizer/`

### Stage 3: Data Preprocessing (~10-15 minutes)
- Creates pre-training examples (span corruption)
- Creates fine-tuning examples (IF masking)
- Splits data into train/validation/test sets
- **Checkpoints**: `data/pretrain_data.jsonl`, `data/finetune_*.jsonl`

### Stage 4: Pre-training (~3-5 hours)
- Trains T5 model from scratch on general Python
- 350,000 examples, 3 epochs
- **Checkpoint**: `models/pretrained/`

### Stage 5: Fine-tuning (~1-2 hours)
- Specializes model for IF condition prediction
- 240,000 examples, 5 epochs
- **Checkpoint**: `models/finetuned/`

### Stage 6: Evaluation (~15-30 minutes)
- Tests on generated test set
- Tests on provided benchmark
- Generates CSV reports and metrics
- **Outputs**: `output/*.csv`, `output/metrics.json`

**Total Time**: 6-12 hours (varies by hardware)

## Resuming After Interruption

The pipeline automatically detects completed stages and skips them. If training is interrupted:

```bash
# Just run again - it will resume where it left off!
python main_script.py
```

The pipeline checks for:
- ✅ Existing data files → Skip data collection
- ✅ Existing tokenizer → Skip tokenizer training
- ✅ Existing preprocessed data → Skip preprocessing
- ✅ Existing pre-trained model → Skip pre-training
- ✅ Existing fine-tuned model → Skip fine-tuning

### Understanding Output

During training, you'll see:

```
STEP 4: MODEL PRE-TRAINING
Model: 60,123,456 parameters
Starting pre-training...
  4%|▎   | 1096/30951 [08:15<3:44:45, 2.21it/s]
{'loss': 2.345, 'epoch': 0.1}
{'loss': 1.987, 'epoch': 0.3}  ← Loss should decrease!
```

**Good signs**:
- Loss decreasing over time
- Speed consistent (1-3 iterations/second)
- No CUDA out of memory errors

**Warning signs**:
- Loss not decreasing (stuck)
- Very slow speed (<0.5 it/s)
- CUDA errors

## Configuration and Customization

To modify training parameters, edit the `Config` class in `main_script.py`:

```python
class Config:
    # Data collection
    MAX_REPOS = 1000              # Number of repositories to process
    MIN_STARS = 10                # Minimum repo stars
    
    # Dataset sizes
    TOTAL_FUNCTIONS_TARGET = 500_000
    PRETRAIN_TARGET = 350_000
    FINETUNE_TARGET = 150_000
    
    # Model architecture
    VOCAB_SIZE = 32_000
    MODEL_DIM = 512               # Increase for larger model
    NUM_LAYERS = 6                # More layers = more capacity
    
    # Training
    PRETRAIN_BATCH_SIZE = 32      # Reduce if CUDA out of memory
    FINETUNE_BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
```

## Troubleshooting

### Problem: "CUDA out of memory"

**Solution**: Reduce batch size in config:
```python
PRETRAIN_BATCH_SIZE = 16  # was 32
FINETUNE_BATCH_SIZE = 8   # was 16
```

### Problem: "evaluation_strategy is not a valid argument"

**Solution**: Update transformers or change to `eval_strategy`:
```bash
pip install --upgrade transformers
```

Or in code:
```python
# Change evaluation_strategy= to eval_strategy=
```

### Problem: "Tree-sitter language not found"

**Solution**: Reinstall tree-sitter-languages:
```bash
pip uninstall tree-sitter-languages
pip install tree-sitter-languages
```

### Problem: Very slow training (CPU mode)

**Solution**: Training on CPU is extremely slow. Consider:
- Using Google Colab (free GPU)
- AWS/Azure GPU instances
- University computing resources

### Problem: Pipeline stops without error

**Check**:
```bash
# Check if process is still running
ps -p $(cat pipeline.pid)

# Check disk space
df -h

# Check last error in log
tail -100 pipeline.log
```

## Additional Information

### GPU Recommendations

- **Minimum**: 8GB VRAM (NVIDIA RTX 3060 or better)
- **Recommended**: 16GB+ VRAM (NVIDIA RTX 4080, A100, H100)
- **Can reduce batch size** if you have less VRAM

### Time Estimates by Hardware

| Hardware | Total Time |
|----------|------------|
| H100 GPU | 4-6 hours |
| A100 GPU | 6-8 hours |
| RTX 4090 | 8-10 hours |
| RTX 3080 | 10-14 hours |
| CPU only | 3-5 days |

### Dataset Quality Notes

The pipeline includes several quality filters:
- Only permissively licensed repositories
- Valid Python syntax (parsed successfully)
- Functions with IF statements
- Reasonable function length (5-500 lines)
- High-quality masking (85% success rate)

### Why Tree-Sitter?

We use tree-sitter for code parsing because:
- Builds proper Abstract Syntax Trees (ASTs)
- Much more accurate than regex for code
- Handles multi-line statements correctly
- Language-aware parsing

This was critical for achieving 85% masking success (vs 1.2% with regex).

## Getting Help

If you encounter issues:

1. Check this README's troubleshooting section
2. Review the pipeline.log for error messages
3. Verify all dependencies are installed correctly
4. Check available disk space and memory

### Model Architecture

- **T5 (Text-to-Text Transfer Transformer)**: Encoder-decoder architecture
- **60M parameters**: Large enough to learn, small enough to train
- **6 layers**: Good balance of capacity and speed
- **512 dimensions**: Standard size for medium models

## Acknowledgments

This project implements techniques from:
- T5: Exploring the Limits of Transfer Learning (Raffel et al., 2020)
- Tree-sitter: An incremental parsing system
- CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation

## License

This project is for educational purposes as part of the AI for Software Engineering course at William & Mary.
