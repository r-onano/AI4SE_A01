#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   IF-Statement Predictor - Enhanced Setup Script          ║"
echo "║   With AST-based extraction and MLM pre-training           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p repos
mkdir -p data/benchmark
mkdir -p data/processed
mkdir -p src/tokenizer
mkdir -p src/model/pretrained
mkdir -p src/model/fine_tuned
mkdir -p results

echo "   ✓ Created repos/"
echo "   ✓ Created data/benchmark/"
echo "   ✓ Created data/processed/"
echo "   ✓ Created src/tokenizer/"
echo "   ✓ Created src/model/"
echo "   ✓ Created results/"
echo ""

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install torch transformers datasets tokenizers tqdm nltk --break-system-packages

if [ $? -eq 0 ]; then
    echo "   ✓ All packages installed successfully"
else
    echo "   ⚠ Some packages may have failed to install"
fi
echo ""

# Download NLTK data
echo "📥 Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True)"
echo "   ✓ NLTK punkt data downloaded"
echo ""

# Clone sample repositories (optional - can also use 'mine' stage)
echo "🔄 Cloning Python repositories for training data..."
cd repos

declare -a repos=(
    "https://github.com/psf/requests.git"
    "https://github.com/pallets/flask.git"
    "https://github.com/django/django.git"
    "https://github.com/numpy/numpy.git"
    "https://github.com/pandas-dev/pandas.git"
    "https://github.com/scikit-learn/scikit-learn.git"
    "https://github.com/ansible/ansible.git"
    "https://github.com/pytorch/pytorch.git"
    "https://github.com/matplotlib/matplotlib.git"
    "https://github.com/scipy/scipy.git"
)

for repo in "${repos[@]}"
do
    repo_name=$(basename "$repo" .git)
    if [ ! -d "$repo_name" ]; then
        echo "   → Cloning $repo_name..."
        git clone --depth 1 "$repo" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "      ✓ $repo_name cloned"
        else
            echo "      ⚠ Failed to clone $repo_name"
        fi
    else
        echo "   ○ $repo_name already exists, skipping"
    fi
done

cd ..
echo ""

# Count Python files
py_count=$(find repos/ -name "*.py" 2>/dev/null | wc -l)
echo "📊 Found $py_count Python files in repositories"
echo ""

# Verification
echo "🔍 Verifying setup..."
checks_passed=0
total_checks=5

# Check 1: Directory structure
if [ -d "repos" ] && [ -d "data/benchmark" ] && [ -d "src" ] && [ -d "results" ]; then
    echo "   ✓ Directory structure created"
    ((checks_passed++))
else
    echo "   ✗ Directory structure incomplete"
fi

# Check 2: Python dependencies
python3 -c "import torch, transformers, datasets, tokenizers, nltk" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ Python dependencies installed"
    ((checks_passed++))
else
    echo "   ✗ Python dependencies missing"
fi

# Check 3: NLTK data
python3 -c "import nltk; nltk.data.find('tokenizers/punkt')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ NLTK data downloaded"
    ((checks_passed++))
else
    echo "   ✗ NLTK data not found"
fi

# Check 4: Python repositories
if [ $py_count -gt 1000 ]; then
    echo "   ✓ Python repositories cloned ($py_count .py files)"
    ((checks_passed++))
else
    echo "   ⚠ Limited Python files found ($py_count .py files)"
fi

# Check 5: Main script
if [ -f "if_predictor_improved.py" ]; then
    echo "   ✓ Main script present"
    ((checks_passed++))
else
    echo "   ⚠ if_predictor_improved.py not found (add it manually)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Setup Status: $checks_passed/$total_checks checks passed"
echo "═══════════════════════════════════════════════════════════"
echo ""

if [ $checks_passed -ge 4 ]; then
    echo "✅ Setup completed successfully!"
    echo ""
    echo "Next Steps:"
    echo "1. Place benchmark_if_only.csv in data/benchmark/ (if available)"
    echo "2. Run full pipeline: python3 if_predictor_improved.py --stage all --epochs 5"
    echo ""
    echo "Or run stages individually:"
    echo "  • python3 if_predictor_improved.py --stage mine      # Mine more repos"
    echo "  • python3 if_predictor_improved.py --stage extract   # Extract functions"
    echo "  • python3 if_predictor_improved.py --stage tokenizer # Train tokenizer"
    echo "  • python3 if_predictor_improved.py --stage pretrain  # Pre-train model"
    echo "  • python3 if_predictor_improved.py --stage finetune  # Fine-tune model"
    echo "  • python3 if_predictor_improved.py --stage evaluate  # Evaluate model"
    echo ""
    echo "Key Improvements:"
    echo "  • AST-based function extraction (no regex)"
    echo "  • Proper MLM pre-training with 15% token masking"
    echo "  • Tokenizer trained on full corpus"
    echo "  • Multi-metric evaluation (EM, F1, Edit Distance, BLEU)"
    echo "  • Consistent prompt prefixes"
else
    echo "⚠️  Setup completed with warnings. Please check failed items above."
    echo ""
    echo "Common fixes:"
    echo "- If dependencies failed: Try running pip install commands manually"
    echo "- If repos not cloned: Check internet connection or use --stage mine"
    echo "- If script not found: Copy if_predictor_improved.py to this directory"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
