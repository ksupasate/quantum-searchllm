"""
Setup script for TNAD (Tensor Network-Augmented Decoding)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="tnad",
    version="0.1.0",
    author="AI Research Team",
    author_email="research@example.com",
    description="Tensor Network-Augmented Decoding: Quantum-inspired inference for LLM coherence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-search-llm",
    packages=find_packages(exclude=["tests", "experiments", "notebooks"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "sentencepiece>=0.2.0",
        "protobuf>=3.20.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.4.0",
        ],
        "experiments": [
            "datasets>=2.14.0",
            "pandas>=2.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tnad-experiment=experiments.run_gsm8k:main",
        ],
    },
)
