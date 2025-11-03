#!/usr/bin/env python3
"""
Ablation Study Runner for TNAD

Systematically tests the impact of key hyperparameters:
- Alpha (α): Fluency vs coherence balance
- Bond dimension (χ): Logical bandwidth
- Beam width (B): Search quality

Usage:
    # Run alpha sweep
    python experiments/run_ablations.py --ablation alpha --num_examples 500

    # Run bond dimension sweep
    python experiments/run_ablations.py --ablation bond_dim --num_examples 500

    # Run beam width sweep
    python experiments/run_ablations.py --ablation beam_width --num_examples 500

    # Run all ablations
    python experiments/run_ablations.py --ablation all --num_examples 500
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

from run_gsm8k import run_gsm8k_experiment
from tnad.utils import setup_logger, get_device


def load_model_and_tokenizer(
    model_name: str,
    device: str = 'auto',
    torch_dtype_name: Optional[str] = None,
    load_in_8bit: bool = False,
) -> tuple[Any, Any]:
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_name}")

    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Resolve dtype
    if torch_dtype_name is None:
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = getattr(torch, torch_dtype_name)

    load_kwargs: Dict[str, Any] = {
        'torch_dtype': dtype,
        'low_cpu_mem_usage': True,
    }

    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs['quantization_config'] = quantization_config
        load_kwargs['device_map'] = 'auto'
    elif device != 'auto':
        load_kwargs['device_map'] = device

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except RuntimeError as exc:
        if 'out of memory' in str(exc).lower():
            logger.warning("OOM - retrying on CPU with float32")
            fallback_kwargs = {
                'torch_dtype': torch.float32,
                'device_map': 'cpu',
                'low_cpu_mem_usage': True,
            }
            model = AutoModelForCausalLM.from_pretrained(model_name, **fallback_kwargs)
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_single_ablation_config(
    param_name: str,
    param_value: Any,
    model: Any,
    tokenizer: Any,
    base_config: Dict[str, Any],
    benchmark: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Run experiment with a specific parameter value.

    Args:
        param_name: Parameter to vary ('alpha', 'bond_dim', 'beam_width')
        param_value: Value for the parameter
        model: LLM model
        tokenizer: Tokenizer
        base_config: Base configuration
        benchmark: Benchmark name ('gsm8k', 'strategyqa')
        seed: Random seed

    Returns:
        Results dictionary
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create config for this ablation
    config = base_config.copy()
    config['fgbs'] = config['fgbs'].copy()

    # Update the specific parameter
    config['fgbs'][param_name] = param_value

    logger.info(f"Running {benchmark} with {param_name}={param_value}, seed={seed}")

    # Run experiment
    device = get_device()

    if benchmark == 'gsm8k':
        result = run_gsm8k_experiment(config, model, tokenizer, device)
    else:
        # Import and use other benchmark runners
        from run_strategyqa import run_strategyqa_experiment
        result = run_strategyqa_experiment(config, model, tokenizer, device)

    # Add configuration info
    result['config'] = {
        'param_name': param_name,
        'param_value': param_value,
        'benchmark': benchmark,
        'seed': seed,
    }

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    return result


def run_ablation_sweep(
    param_name: str,
    param_values: List[Any],
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    benchmark: str,
    seeds: List[int],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Run ablation study sweeping a parameter.

    Args:
        param_name: Parameter to vary
        param_values: List of values to test
        model: LLM model
        tokenizer: Tokenizer
        config: Base configuration
        benchmark: Benchmark name
        seeds: List of random seeds
        output_dir: Output directory

    Returns:
        List of all results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"ABLATION STUDY: {param_name.upper()}")
    logger.info(f"Values: {param_values}")
    logger.info(f"Benchmark: {benchmark}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"{'='*80}\n")

    all_results = []

    for param_value in tqdm(param_values, desc=f"Sweeping {param_name}"):
        for seed in seeds:
            result = run_single_ablation_config(
                param_name,
                param_value,
                model,
                tokenizer,
                config,
                benchmark,
                seed
            )

            all_results.append(result)

            # Save individual result
            filename = f"ablation_{param_name}_{param_value}_{benchmark}_seed{seed}.json"
            filepath = output_dir / filename

            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            logger.info(f"Saved: {filepath}")

    return all_results


def aggregate_ablation_results(
    results: List[Dict[str, Any]],
    param_name: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate ablation results by parameter value.

    Args:
        results: List of result dictionaries
        param_name: Parameter name

    Returns:
        Dictionary mapping param_value -> aggregated stats
    """
    from statistical_analysis import compute_mean_std

    # Group by parameter value
    grouped = {}
    for result in results:
        param_value = result['config']['param_value']
        if param_value not in grouped:
            grouped[param_value] = []

        grouped[param_value].append(result['metrics']['accuracy'])

    # Compute statistics
    aggregated = {}
    for param_value, accuracies in grouped.items():
        mean, std = compute_mean_std(accuracies)
        aggregated[param_value] = {
            'mean': mean,
            'std': std,
            'values': accuracies,
            'n': len(accuracies),
        }

    return aggregated


def plot_ablation_results(
    aggregated: Dict[str, Dict[str, Any]],
    param_name: str,
    benchmark: str,
    output_dir: Path,
):
    """
    Plot ablation study results.

    Args:
        aggregated: Aggregated results from aggregate_ablation_results
        param_name: Parameter name
        benchmark: Benchmark name
        output_dir: Output directory for plots
    """
    # Extract data
    param_values = sorted(aggregated.keys())
    means = [aggregated[v]['mean'] * 100 for v in param_values]  # Convert to percentage
    stds = [aggregated[v]['std'] * 100 for v in param_values]

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Plot line with error bars
    plt.errorbar(param_values, means, yerr=stds, marker='o', markersize=8,
                 capsize=5, capthick=2, linewidth=2, label='FGBS')

    # Labels and title
    param_display_names = {
        'alpha': 'Alpha (α) - Fluency vs Coherence',
        'bond_dim': 'Bond Dimension (χ) - Logical Bandwidth',
        'beam_width': 'Beam Width (B) - Search Quality',
    }

    plt.xlabel(param_display_names.get(param_name, param_name), fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Ablation Study: {param_name.upper()} on {benchmark.upper()}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Save plot
    plot_filename = f"ablation_{param_name}_{benchmark}.png"
    plot_path = output_dir / plot_filename
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot: {plot_path}")

    # Also save as CSV
    csv_filename = f"ablation_{param_name}_{benchmark}.csv"
    csv_path = output_dir / csv_filename

    with open(csv_path, 'w') as f:
        f.write(f"{param_name},mean_accuracy,std_accuracy,n_seeds\n")
        for param_value in param_values:
            stats = aggregated[param_value]
            f.write(f"{param_value},{stats['mean']:.4f},{stats['std']:.4f},{stats['n']}\n")

    logger.info(f"Saved CSV: {csv_path}")


def main():
    """Main ablation study runner."""
    parser = argparse.ArgumentParser(
        description="Run ablation studies for TNAD hyperparameters"
    )

    parser.add_argument(
        '--ablation',
        type=str,
        default='alpha',
        choices=['alpha', 'bond_dim', 'beam_width', 'all'],
        help='Which ablation to run'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='mistralai/Mistral-7B-Instruct-v0.3',
        help='Model name'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        default='gsm8k',
        choices=['gsm8k', 'strategyqa'],
        help='Benchmark to use'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42],
        help='Random seeds (space-separated)'
    )
    parser.add_argument(
        '--num_examples',
        type=int,
        default=500,
        help='Number of examples'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/ablations',
        help='Output directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Base config file'
    )

    # Parameter-specific arguments
    parser.add_argument(
        '--alpha_values',
        type=float,
        nargs='+',
        default=[0.0, 0.3, 0.5, 0.7, 1.0],
        help='Alpha values to test'
    )
    parser.add_argument(
        '--bond_dim_values',
        type=int,
        nargs='+',
        default=[4, 8, 16, 32],
        help='Bond dimension values to test'
    )
    parser.add_argument(
        '--beam_width_values',
        type=int,
        nargs='+',
        default=[1, 3, 5, 10],
        help='Beam width values to test'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Update config
    config['model']['name'] = args.model
    config['experiment']['num_examples'] = args.num_examples

    # Setup logging
    setup_logger(log_level='INFO')

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_cfg = config.get('model', {})
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        device=model_cfg.get('device', 'auto'),
        torch_dtype_name=model_cfg.get('torch_dtype'),
        load_in_8bit=model_cfg.get('load_in_8bit', False),
    )

    # Determine which ablations to run
    ablations_to_run = []
    if args.ablation == 'all':
        ablations_to_run = [
            ('alpha', args.alpha_values),
            ('bond_dim', args.bond_dim_values),
            ('beam_width', args.beam_width_values),
        ]
    elif args.ablation == 'alpha':
        ablations_to_run = [('alpha', args.alpha_values)]
    elif args.ablation == 'bond_dim':
        ablations_to_run = [('bond_dim', args.bond_dim_values)]
    elif args.ablation == 'beam_width':
        ablations_to_run = [('beam_width', args.beam_width_values)]

    # Run ablations
    all_aggregated_results = {}

    for param_name, param_values in ablations_to_run:
        # Run sweep
        results = run_ablation_sweep(
            param_name,
            param_values,
            model,
            tokenizer,
            config,
            args.benchmark,
            args.seeds,
            output_dir
        )

        # Aggregate results
        aggregated = aggregate_ablation_results(results, param_name)
        all_aggregated_results[param_name] = aggregated

        # Plot results
        plot_ablation_results(aggregated, param_name, args.benchmark, output_dir)

        # Print summary
        print(f"\n{'='*80}")
        print(f"ABLATION SUMMARY: {param_name.upper()}")
        print(f"{'='*80}")
        for param_value in sorted(aggregated.keys()):
            stats = aggregated[param_value]
            print(f"{param_name}={param_value}: {stats['mean']*100:.1f}% ± {stats['std']*100:.1f}%")
        print(f"{'='*80}\n")

    # Save summary
    summary = {
        'ablations': args.ablation,
        'model': args.model,
        'benchmark': args.benchmark,
        'num_examples': args.num_examples,
        'seeds': args.seeds,
        'timestamp': datetime.now().isoformat(),
        'results': all_aggregated_results,
    }

    summary_path = output_dir / 'ablation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Saved summary: {summary_path}")

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"  - Individual results: {output_dir}/ablation_*.json")
    print(f"  - Plots: {output_dir}/ablation_*.png")
    print(f"  - CSV data: {output_dir}/ablation_*.csv")
    print(f"  - Summary: {output_dir}/ablation_summary.json")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
