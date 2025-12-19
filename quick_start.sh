#!/bin/bash
# Quick Start Script for AI4SE IF Prediction Pipeline

set -e  # Exit on error

echo "=================================="
echo "AI4SE IF Prediction - Quick Start"
echo "=================================="

# Step 1: Setup environment
echo ""
echo "Step 1: Setting up environment..."
cd ~/if_predictor

# Check if conda environment exists
if conda env list | grep -q "if_pred_env"; then
    echo " Environment 'if_pred_env' already exists"
    conda activate if_pred_env
else
    echo " Creating conda environment..."
    conda create -n if_pred_env python=3.10 -y
    conda activate if_pred_env
    
    echo " Installing dependencies..."
    pip install -q requests tqdm tokenizers transformers pandas \
        tree-sitter tree-sitter-languages datasets torch
    echo " Dependencies installed"
fi

# Step 2: Verify files
echo ""
echo "Step 2: Verifying files..."
if [ ! -f "main_script.py" ]; then
    echo " ERROR: main_script.py not found!"
    echo "   Please copy it to ~/if_predictor/"
    exit 1
fi
echo " main_script.py found"

# Step 3: Check for benchmark file
echo ""
echo "Step 3: Checking benchmark file..."
mkdir -p data
if [ -f "data/benchmark_if_only_2_.csv" ]; then
    echo " Benchmark file found"
else
    echo " Warning: benchmark_if_only_2_.csv not found in data/"
    echo "   Pipeline will still work, but won't evaluate on provided benchmark"
fi

# Step 4: Check existing progress
echo ""
echo "Step 4: Checking existing progress..."
if [ -f "data/raw_functions.jsonl" ]; then
    FUNC_COUNT=$(wc -l < data/raw_functions.jsonl)
    echo " Found ${FUNC_COUNT} collected functions"
fi

if [ -f "data/finetune_train.jsonl" ]; then
    TRAIN_COUNT=$(wc -l < data/finetune_train.jsonl)
    echo " Found ${TRAIN_COUNT} fine-tuning examples"
fi

if [ -d "models/pretrained" ]; then
    echo " Pre-trained model exists"
fi

if [ -d "models/finetuned" ]; then
    echo " Fine-tuned model exists"
fi

# Step 5: Ask user how to proceed
echo ""
echo "Step 5: Launch pipeline..."
echo ""
echo "Choose an option:"
echo "  1) Fresh start (will skip completed steps)"
echo "  2) Force restart from beginning (delete all checkpoints)"
echo "  3) Test run (dry run)"
echo "  4) Exit"
echo ""
read -p "Your choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo " Starting pipeline (resuming from checkpoints)..."
        nohup python main_script.py > pipeline.log 2>&1 &
        PID=$!
        echo $PID > pipeline.pid
        echo ""
        echo " Pipeline started!"
        echo "   PID: $PID"
        echo "   Log: pipeline.log"
        echo ""
        echo "Monitor with: tail -f pipeline.log"
        echo "Check status: ps -p $PID"
        echo "Stop: kill $PID"
        echo ""
        echo "First 20 lines of log:"
        sleep 2
        tail -20 pipeline.log
        ;;
    2)
        echo ""
        echo " WARNING: This will delete all progress!"
        read -p "Are you sure? [y/N]: " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo " Deleting checkpoints..."
            rm -rf data/*.jsonl
            rm -rf models/*
            rm -rf output/*
            echo " Checkpoints deleted"
            echo ""
            echo " Starting fresh pipeline..."
            nohup python main_script.py > pipeline.log 2>&1 &
            PID=$!
            echo $PID > pipeline.pid
            echo ""
            echo " Pipeline started!"
            echo "   PID: $PID"
            echo "   Log: pipeline.log"
        else
            echo "Cancelled"
        fi
        ;;
    3)
        echo ""
        echo " Running test (checking imports and config)..."
        python -c "
import sys
sys.path.insert(0, '.')
import main_script
print(' All imports successful')
print(f' Config loaded: {main_script.config.BASE_DIR}')
print(f' GPU available: {main_script.torch.cuda.is_available()}')
print(f' Tree-sitter parser: {main_script.PARSER is not None}')
"
        echo ""
        echo " Test passed! Pipeline is ready to run."
        ;;
    4)
        echo "Exited"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
