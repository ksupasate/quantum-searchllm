#!/bin/bash
# Install all required dependencies for TNAD experiments

echo "Installing TNAD Dependencies..."
echo "================================"

# Core dependencies
echo "Installing core dependencies..."
python3 -m pip install --user torch numpy scipy

# LLM dependencies
echo "Installing LLM dependencies..."
python3 -m pip install --user transformers accelerate sentencepiece protobuf

# Experiment dependencies
echo "Installing experiment dependencies..."
python3 -m pip install --user datasets pyyaml tqdm loguru pandas

# Visualization dependencies (optional)
echo "Installing visualization dependencies..."
python3 -m pip install --user matplotlib seaborn jupyter ipywidgets

# Testing dependencies (optional)
echo "Installing testing dependencies..."
python3 -m pip install --user pytest pytest-cov

echo ""
echo "================================"
echo "Installation complete!"
echo ""
echo "Test your setup:"
echo "  python3 test_setup.py"
echo ""
echo "Run experiments:"
echo "  python3 experiments/reproduce_paper_results.py --quick_test"
