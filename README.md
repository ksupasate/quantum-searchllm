# TNAD: Tensor Network-Augmented Decoding

**Quantum-inspired inference framework for improving logical coherence in Large Language Model reasoning**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

TNAD (Tensor Network-Augmented Decoding) is a novel inference framework that leverages quantum-inspired tensor network methods to enhance the logical coherence of Large Language Model (LLM) outputs. By maintaining a Matrix Product State (MPS) representation of token sequences during generation, TNAD can detect and penalize incoherent reasoning paths in real-time, leading to more reliable and logically consistent responses.

### Key Innovation

Traditional LLM decoding methods (greedy, beam search, sampling) optimize for local token probability without considering global logical structure. TNAD introduces **Fidelity-Guided Beam Search (FGBS)**, which balances:

- **Fluency** (standard LLM probability): How natural the text sounds
- **Coherence** (quantum-inspired fidelity score): How logically consistent the reasoning is

This is achieved through the **Coherence Fidelity Score (CFS)**, computed from the Schmidt spectrum of the MPS representation, which quantifies structural integrity of the generated sequence.

### Mathematical Foundation

**Standard Beam Search:**
```
Score(S) = log P(S)
```

**Fidelity-Guided Beam Search:**
```
Score(S) = α · log P(S) + (1-α) · log F(S)
```

where:
- `P(S)`: LLM probability (fluency)
- `F(S)`: Coherence Fidelity Score (structural integrity)
- `α ∈ [0,1]`: Balance parameter

The CFS is derived from quantum purity measures:
```
Given Schmidt values λ = [λ₁, λ₂, ..., λ_χ]:
Purity: P = Σᵢ λᵢ⁴
CFS: F = 1 / P

High F → uniform spectrum → high entanglement → coherent state
Low F → peaked spectrum → low entanglement → decoherent state
```

---

## Features

- **Quantum-Inspired Coherence Scoring**: Real-time structural monitoring via Matrix Product States
- **Fidelity-Guided Beam Search**: Novel decoding algorithm balancing fluency and coherence
- **Memory Efficient**: Optimized MPS implementation with bond dimension control
- **GPU Accelerated**: Full CUDA/MPS support with 8-bit and 4-bit quantization
- **Production Ready**: Comprehensive test suite, type hints, and detailed documentation
- **Research Grade**: Extensive experiment framework for benchmarking and analysis

---

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)
- 16GB+ RAM recommended (8GB minimum with quantization)

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/quantum-search-llm.git
cd quantum-search-llm

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Dependencies Installation Script

For a guided installation with dependency checks:

```bash
bash install_dependencies.sh
```

### Verify Installation

```bash
python test_setup.py
```

---

## Quick Start

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from tnad import FidelityGuidedBeamSearcher

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# Initialize FGBS
searcher = FidelityGuidedBeamSearcher(
    model=model,
    tokenizer=tokenizer,
    beam_width=5,      # Number of parallel beams
    alpha=0.5,         # Balance: 0=pure coherence, 1=pure LLM
    bond_dim=16,       # MPS bond dimension (controls coherence tracking)
)

# Generate coherent text
prompt = "If all cats are animals, and some animals can fly, can all cats fly? Let's think step by step."
result = searcher.generate(prompt, max_length=100)

print("Generated text:", result['text'])
print("Coherence score:", result['log_cfs'])
print("LLM probability:", result['log_prob'])
```

### Memory-Efficient Usage (8-bit Quantization)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 8-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=quant_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Use smaller bond dimension for memory efficiency
searcher = FidelityGuidedBeamSearcher(
    model=model,
    tokenizer=tokenizer,
    beam_width=3,      # Reduced for memory
    alpha=0.5,
    bond_dim=8,        # Smaller bond dimension
)
```

### Comparing with Baseline

```python
# Generate with both FGBS and standard beam search
comparison = searcher.compare_with_baseline(
    prompt="Solve: If x + 2 = 5, then x = ?",
    max_length=100,
)

print("FGBS output:", comparison['fgbs']['text'])
print("Baseline output:", comparison['baseline']['text'])
print("CFS improvement:", comparison['cfs_comparison']['cfs_improvement'])
```

---

## Configuration

TNAD uses YAML configuration files for reproducible experiments. See `configs/default.yaml` for the full configuration template.

### Key Parameters

**FGBS Algorithm:**
- `beam_width` (3-10): Number of parallel beams. Higher = better quality, slower
- `alpha` (0.0-1.0): Fluency vs coherence balance
  - 1.0 = pure LLM (standard beam search)
  - 0.5 = balanced (recommended)
  - 0.3 = prioritize coherence (good for reasoning)
- `bond_dim` (8-32): MPS bond dimension
  - Smaller: faster but limited logical tracking
  - Larger: slower but better coherence monitoring
- `top_k` (20-50): Number of top tokens to consider per beam

**Model Configuration:**
- `load_in_8bit`: Enable 8-bit quantization for memory efficiency
- `torch_dtype`: Precision (float16/bfloat16 for efficiency)
- `device`: Target device (auto/cuda/cpu/mps)

**Generation:**
- `max_length`: Maximum generation length (tokens)
- `min_length`: Minimum length before allowing EOS
- `temperature`: Sampling temperature (1.0 = no scaling)

### Example Configurations

**Memory-Optimized (for 8GB GPU):**
```yaml
fgbs:
  beam_width: 3
  alpha: 0.5
  bond_dim: 8
  top_k: 30

model:
  load_in_8bit: true
  torch_dtype: "float16"

generation:
  max_length: 256
```

**Quality-Optimized (for 24GB+ GPU):**
```yaml
fgbs:
  beam_width: 10
  alpha: 0.5
  bond_dim: 32
  top_k: 50

model:
  load_in_8bit: false
  torch_dtype: "bfloat16"

generation:
  max_length: 512
```

---

## Experiments

### Running GSM8K Benchmark

GSM8K is a dataset of grade school math word problems requiring multi-step reasoning.

```bash
# Run with default configuration
python experiments/run_gsm8k.py --config configs/default.yaml

# Override specific parameters
python experiments/run_gsm8k.py \
    --config configs/default.yaml \
    --alpha 0.5 \
    --bond_dim 16 \
    --beam_width 5 \
    --num_examples 100

# Use custom model
python experiments/run_gsm8k.py \
    --config configs/default.yaml \
    --model "meta-llama/Llama-3.1-8B-Instruct"
```

### Running StrategyQA Benchmark

StrategyQA tests multi-hop reasoning with implicit decomposition.

```bash
python experiments/run_strategyqa.py --config configs/default.yaml
```

### Ablation Studies

Run comprehensive ablation studies across hyperparameters:

```bash
python experiments/run_ablations.py --config configs/default.yaml
```

This will sweep over:
- Alpha values: [0.0, 0.3, 0.5, 0.7, 1.0]
- Bond dimensions: [4, 8, 16, 32]
- Beam widths: [1, 3, 5, 10]

### Baseline Comparisons

```bash
python experiments/baselines.py \
    --methods greedy beam_search self_consistency \
    --num_examples 100
```

### Reproducing Paper Results

```bash
python experiments/reproduce_paper_results.py
```

---

## Jupyter Notebooks

Interactive demonstrations and tutorials are available in the `notebooks/` directory:

- **`demo.ipynb`**: Quick introduction and basic usage
- **`tutorial_comprehensive.ipynb`**: In-depth tutorial covering all features
- **`performance_benchmark.ipynb`**: Performance analysis and profiling
- **`tnad_colab.ipynb`**: Google Colab compatible notebook

```bash
# Launch Jupyter
jupyter notebook notebooks/
```

---

## API Reference

### Core Components

#### `FidelityGuidedBeamSearcher`

Main FGBS implementation for LLM generation.

```python
searcher = FidelityGuidedBeamSearcher(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    beam_width: int = 5,
    alpha: float = 0.5,
    bond_dim: int = 16,
    top_k: int = 50,
    temperature: float = 1.0,
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
)
```

**Methods:**
- `generate(prompt, max_length, min_length, return_details, show_progress)`: Generate text
- `compare_with_baseline(prompt, max_length)`: Compare FGBS vs standard beam search

#### `MPSSequence`

Matrix Product State representation of token sequences.

```python
mps = MPSSequence(
    bond_dim: int,
    embedding_dim: int,
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
)
```

**Methods:**
- `add_token(token_embedding)`: Add token to MPS chain
- `get_schmidt_values(cut_position)`: Extract Schmidt spectrum
- `copy()`: Create deep copy for beam branching

#### `compute_cfs()`

Compute Coherence Fidelity Score from Schmidt values.

```python
from tnad import compute_cfs

cfs = compute_cfs(
    schmidt_values: Union[np.ndarray, torch.Tensor],
    normalize: bool = True,
    eps: float = 1e-10,
    return_purity: bool = False,
)
```

### Utility Functions

```python
from tnad.utils import (
    log_normalize,      # Log-space normalization
    safe_divide,        # Numerically stable division
    normalize_schmidt_values,  # Schmidt value normalization
    compute_purity,     # Quantum purity calculation
    get_device,         # Auto device selection
)
```

---

## Project Structure

```
quantum-searchllm/
├── tnad/                       # Core package
│   ├── __init__.py            # Package exports
│   ├── fgbs_searcher.py       # FGBS implementation
│   ├── mps_manager.py         # MPS representation
│   ├── coherence_score.py     # CFS computation
│   └── utils.py               # Utility functions
├── experiments/               # Experiment scripts
│   ├── run_gsm8k.py          # GSM8K benchmark
│   ├── run_strategyqa.py     # StrategyQA benchmark
│   ├── run_ablations.py      # Ablation studies
│   ├── baselines.py          # Baseline methods
│   ├── aggregate_results.py  # Results aggregation
│   └── visualize_results.py  # Plotting utilities
├── tests/                     # Test suite
│   ├── test_fgbs_integration.py
│   ├── test_coherence_score.py
│   └── test_mps_manager.py
├── notebooks/                 # Jupyter tutorials
│   ├── demo.ipynb
│   ├── tutorial_comprehensive.ipynb
│   └── performance_benchmark.ipynb
├── configs/                   # Configuration files
│   ├── default.yaml
│   ├── memory_optimized.yaml
│   └── full_publication.yaml
├── data/                      # Sample datasets
├── results/                   # Experiment outputs
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

---

## Performance Optimization

### Memory Efficiency

TNAD includes several optimizations for memory-constrained environments:

1. **Quantization**: 8-bit and 4-bit model loading
2. **Optimized MPS**: Reduced memory allocations, efficient copying
3. **Garbage Collection**: Aggressive cleanup during beam search
4. **Cache Management**: LRU cache for Schmidt values

```python
# Memory monitoring
from experiments.check_gpu_memory import monitor_memory

with monitor_memory():
    result = searcher.generate(prompt, max_length=100)
```

### Speed Optimization

1. **Batch Embeddings**: Pre-compute embeddings for top-k tokens
2. **Efficient Matrix Operations**: Optimized @ operator usage
3. **Caching**: Schmidt value caching with configurable size
4. **PyTorch Optimizations**: Gradient-free inference, mixed precision

```python
# For maximum speed (sacrifices some coherence tracking)
searcher = FidelityGuidedBeamSearcher(
    model=model,
    tokenizer=tokenizer,
    beam_width=3,      # Smaller beam
    bond_dim=8,        # Smaller bond dimension
    top_k=20,          # Fewer candidates
)
```

---

## Testing

Run the full test suite:

```bash
# All tests with coverage
pytest tests/ --cov=tnad --cov-report=html

# Specific test files
pytest tests/test_fgbs_integration.py -v
pytest tests/test_coherence_score.py -v
pytest tests/test_mps_manager.py -v
```

### Test Coverage

The test suite includes:
- Unit tests for all core components
- Integration tests for FGBS pipeline
- Numerical stability tests
- Edge case handling
- Memory leak detection

---

## Troubleshooting

### Common Issues

**Out of Memory (OOM):**
```python
# Solution 1: Enable 8-bit quantization
load_in_8bit: true

# Solution 2: Reduce beam width and bond dimension
beam_width: 3
bond_dim: 8

# Solution 3: Reduce generation length
max_length: 128
```

**Slow Generation:**
```python
# Reduce top_k to consider fewer tokens
top_k: 20

# Reduce beam width
beam_width: 3

# Use smaller model
model: "microsoft/phi-2"
```

**MPS (Apple Silicon) Errors:**
```python
# Some operations may not be supported on MPS
# The code automatically falls back to CPU for unsupported ops

# Force CPU for stability:
device: "cpu"
```

**Quantization Not Working:**
```bash
# Ensure bitsandbytes is installed
pip install bitsandbytes>=0.41.0

# Check GPU compatibility (CUDA 11.1+)
python -c "import torch; print(torch.cuda.is_available())"
```

### Debug Mode

Enable detailed logging:

```python
from tnad.utils import setup_logger

setup_logger(log_level="DEBUG", log_file="debug.log")
```

---

## Citation

If you use TNAD in your research, please cite:

```bibtex
@software{tnad2024,
  title={TNAD: Tensor Network-Augmented Decoding for Coherent LLM Reasoning},
  author={AI Research Team},
  year={2024},
  url={https://github.com/yourusername/quantum-search-llm}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Inspired by tensor network methods from quantum many-body physics
- Built on HuggingFace Transformers and PyTorch
- Uses Matrix Product State (MPS) formalism from quantum information theory

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests before committing
pytest tests/ --cov=tnad

# Format code
black tnad/ tests/ experiments/

# Type checking
mypy tnad/
```

---

## Contact

For questions and feedback:
- Open an issue on GitHub
- Email: research@example.com

---

## Roadmap

- [ ] Support for encoder-decoder models (T5, BART)
- [ ] Multi-GPU distributed beam search
- [ ] Integration with vLLM for production deployment
- [ ] Streaming generation support
- [ ] Fine-tuning pipeline with coherence rewards
- [ ] Web demo and API server

---

**Built with ❤️ for more reliable and coherent AI reasoning**
