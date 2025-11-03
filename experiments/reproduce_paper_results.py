#!/usr/bin/env python3
"""
Master Experiment Runner: Reproduce Paper Results

Runs the complete experimental suite from the research paper:
- Table 1: Accuracy on GSM8K, StrategyQA, EntailmentBank
- Table 2: Coherence metrics (Negation/Transitivity violations)

Compares:
- Greedy Decoding
- Standard Beam Search (B=5)
- Self-Consistency (N=10)
- FGBS (alpha=0.5, chi=16) [Our method]

NEW: Multi-seed and multi-model support for statistical significance testing

Usage:
    # Quick test (10 examples each)
    python experiments/reproduce_paper_results.py --quick_test

    # Full reproduction (as in paper) with multiple seeds
    python experiments/reproduce_paper_results.py --full --seeds 42 123 456

    # Test multiple models with multiple seeds
    python experiments/reproduce_paper_results.py \
        --models mistralai/Mistral-7B-Instruct-v0.3 microsoft/phi-2 \
        --seeds 42 123 456 \
        --num_examples 500

    # Custom configuration
    python experiments/reproduce_paper_results.py --model meta-llama/Llama-3.1-8B-Instruct \
        --num_examples 100 --alpha 0.5 --bond_dim 16
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

# Add parent directory to path to import tnad
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
try:  # pragma: no cover
    from loguru import logger
except ImportError:  # pragma: no cover
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

# Import all experiment modules
from baselines import GreedyDecoder, StandardBeamSearch, SelfConsistency
from coherence_metrics import run_coherence_evaluation
from run_gsm8k import (
    run_gsm8k_experiment,
    run_gsm8k_self_consistency_experiment,
    extract_answer_from_text as extract_gsm8k_answer,
)
from run_strategyqa import (
    run_strategyqa_experiment,
    run_strategyqa_self_consistency_experiment,
    extract_yes_no_answer,
)

from tnad import FidelityGuidedBeamSearcher
from tnad.utils import setup_logger, get_device

# Import statistical analysis module
from statistical_analysis import (
    generate_accuracy_table,
    generate_coherence_table,
    save_results_summary,
    generate_summary_stats,
)


def _resolve_dtype(dtype_name: Optional[str]) -> torch.dtype:
    """Resolve a dtype string into a torch.dtype, with sensible defaults."""
    if dtype_name is None:
        if torch.cuda.is_available():
            return torch.float16
        if torch.backends.mps.is_available():
            return torch.float16
        return torch.float32

    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:  # pragma: no cover - configuration error
        raise ValueError(f"Unsupported torch dtype '{dtype_name}'") from exc


def load_model_and_tokenizer(
    model_name: str,
    device: str = 'auto',
    torch_dtype_name: Optional[str] = None,
    load_in_8bit: bool = False,
) -> tuple[Any, Any]:
    """Load model and tokenizer with graceful fallbacks for constrained devices."""
    logger.info(f"Loading model: {model_name}")

    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dtype = _resolve_dtype(torch_dtype_name)
    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    load_kwargs: Dict[str, Any] = {
        'torch_dtype': dtype,
        'low_cpu_mem_usage': True,
    }

    if quantization_config is not None:
        load_kwargs['quantization_config'] = quantization_config
        load_kwargs['device_map'] = 'auto'

    if device is not None:
        load_kwargs['device_map'] = device

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except RuntimeError as exc:
        message = str(exc)
        if 'out of memory' in message.lower():
            logger.warning(
                "Initial model load failed due to OOM (%s). Retrying on CPU with float32.",
                exc.__class__.__name__,
            )
            fallback_kwargs = {
                'torch_dtype': torch.float32,
                'device_map': 'cpu',
                'low_cpu_mem_usage': True,
            }
            if load_in_8bit:
                fallback_cfg = BitsAndBytesConfig(load_in_8bit=True)
                fallback_kwargs['quantization_config'] = fallback_cfg
                fallback_kwargs['device_map'] = 'auto'
            model = AutoModelForCausalLM.from_pretrained(model_name, **fallback_kwargs)
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_single_method_single_seed(
    method_name: str,
    benchmark_name: str,
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    seed: int,
    model_name: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run a single method on a single benchmark with a specific seed.

    Args:
        method_name: 'greedy', 'beam_search', 'self_consistency', or 'fgbs'
        benchmark_name: 'gsm8k', 'strategyqa', or 'entailmentbank'
        model: LLM model
        tokenizer: Tokenizer
        config: Configuration dict
        seed: Random seed
        model_name: Model identifier for saving results
        output_dir: Directory to save results

    Returns:
        Dictionary with results
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    method_config = config.copy()

    # Configure method-specific parameters
    if method_name == 'greedy':
        method_config['fgbs'] = {
            'beam_width': 1,
            'alpha': 1.0,
            'bond_dim': 8,
            'top_k': config['fgbs']['top_k'],
            'temperature': config['fgbs']['temperature'],
            'normalize_embeddings': config['fgbs']['normalize_embeddings'],
        }
    elif method_name == 'beam_search':
        method_config['fgbs'] = {
            'beam_width': 5,
            'alpha': 1.0,  # Pure LLM probability
            'bond_dim': config['fgbs']['bond_dim'],
            'top_k': config['fgbs']['top_k'],
            'temperature': config['fgbs']['temperature'],
            'normalize_embeddings': config['fgbs']['normalize_embeddings'],
        }
    # FGBS and self_consistency use default config

    # Run experiment
    logger.info(f"Running {method_name} on {benchmark_name} with seed={seed}")

    if benchmark_name == 'gsm8k':
        if method_name == 'self_consistency':
            result = run_gsm8k_self_consistency_experiment(method_config, model, tokenizer, device)
        else:
            result = run_gsm8k_experiment(method_config, model, tokenizer, device)
    elif benchmark_name == 'strategyqa':
        if method_name == 'self_consistency':
            result = run_strategyqa_self_consistency_experiment(method_config, model, tokenizer, device)
        else:
            result = run_strategyqa_experiment(method_config, model, tokenizer, device)
    else:
        logger.warning(f"Benchmark {benchmark_name} not implemented")
        return {}

    # Save individual result
    model_short = model_name.split('/')[-1]
    result_filename = f"{method_name}_{benchmark_name}_{model_short}_seed{seed}.json"
    result_path = output_dir / result_filename

    result_data = {
        'method': method_name,
        'benchmark': benchmark_name,
        'model': model_name,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        **result['metrics']
    }

    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2, default=str)

    logger.info(f"Saved result to {result_path}")

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    return result


def run_single_benchmark_all_methods(
    benchmark_name: str,
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    seed: int,
    model_name: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run a single benchmark with all methods (baselines + FGBS) for one seed.

    Args:
        benchmark_name: 'gsm8k', 'strategyqa', or 'entailmentbank'
        model: LLM model
        tokenizer: Tokenizer
        config: Configuration dict
        seed: Random seed
        model_name: Model identifier
        output_dir: Directory to save results

    Returns:
        Dictionary with results from all methods
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {benchmark_name.upper()} with all methods (seed={seed})")
    logger.info(f"{'='*60}\n")

    results = {}
    methods = ['greedy', 'beam_search', 'self_consistency', 'fgbs']

    for method in methods:
        try:
            result = run_single_method_single_seed(
                method,
                benchmark_name,
                model,
                tokenizer,
                config,
                seed,
                model_name,
                output_dir
            )
            results[method] = result
        except Exception as e:
            logger.error(f"Failed to run {method}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return results


def run_coherence_evaluation_single_method(
    method_name: str,
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run coherence evaluation for a single method with a specific seed.

    Args:
        method_name: 'greedy', 'beam_search', 'self_consistency', or 'fgbs'
        model: LLM model
        tokenizer: Tokenizer
        config: Configuration dict
        seed: Random seed
        output_dir: Directory to save results

    Returns:
        Dictionary with coherence metrics
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()

    logger.info(f"Evaluating {method_name} coherence (seed={seed})...")

    # Create decoder based on method
    if method_name == 'greedy':
        decoder = GreedyDecoder(model, tokenizer, device)
    elif method_name == 'beam_search':
        decoder = StandardBeamSearch(model, tokenizer, beam_width=5, device=device)
    elif method_name == 'self_consistency':
        decoder = SelfConsistency(model, tokenizer, num_samples=10, device=device)
    elif method_name == 'fgbs':
        decoder = FidelityGuidedBeamSearcher(
            model=model,
            tokenizer=tokenizer,
            beam_width=config['fgbs']['beam_width'],
            alpha=config['fgbs']['alpha'],
            bond_dim=config['fgbs']['bond_dim'],
            top_k=config['fgbs']['top_k'],
            temperature=config['fgbs']['temperature'],
            device=device,
            normalize_embeddings=config['fgbs']['normalize_embeddings'],
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    result = run_coherence_evaluation(decoder, model, tokenizer)

    # Save result
    result_filename = f"coherence_{method_name}_seed{seed}.json"
    result_path = output_dir / result_filename

    result_data = {
        'method': method_name,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        **result
    }

    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2, default=str)

    logger.info(f"Saved coherence result to {result_path}")

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    return result


def run_coherence_evaluation_all_methods(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run coherence evaluation (Table 2) with all methods for one seed.

    Args:
        model: LLM model
        tokenizer: Tokenizer
        config: Configuration dict
        seed: Random seed
        output_dir: Directory to save results

    Returns:
        Dictionary with coherence metrics for all methods
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Coherence Evaluation (Table 2) (seed={seed})")
    logger.info(f"{'='*60}\n")

    results = {}
    methods = ['greedy', 'beam_search', 'self_consistency', 'fgbs']

    for method in methods:
        try:
            result = run_coherence_evaluation_single_method(
                method,
                model,
                tokenizer,
                config,
                seed,
                output_dir
            )
            results[method] = result
        except Exception as e:
            logger.error(f"Failed to evaluate {method} coherence: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return results


def generate_table_1_single_seed(all_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate Table 1 from paper: Accuracy Results (for single seed, no stats)

    | Benchmark | Greedy | Beam Search (B=5) | Self-Consistency | FGBS (Ours) |
    """
    table = "\n" + "="*80 + "\n"
    table += "TABLE 1: ACCURACY RESULTS (%)\n"
    table += "="*80 + "\n"
    table += f"{'Benchmark':<20} {'Greedy':>12} {'Beam (B=5)':>12} {'Self-Cons':>12} {'FGBS':>12}\n"
    table += "-"*80 + "\n"

    benchmarks = ['gsm8k', 'strategyqa', 'entailmentbank']

    for benchmark in benchmarks:
        if benchmark not in all_results:
            continue

        bench_results = all_results[benchmark]
        row = f"{benchmark.upper():<20}"

        for method in ['greedy', 'beam_search', 'self_consistency', 'fgbs']:
            if method in bench_results and 'metrics' in bench_results[method]:
                acc = bench_results[method]['metrics']['accuracy'] * 100
                row += f"{acc:>12.1f}"
            else:
                row += f"{'N/A':>12}"

        table += row + "\n"

    table += "="*80 + "\n"
    return table


def generate_table_2_single_seed(coherence_results: Dict[str, Any]) -> str:
    """
    Generate Table 2 from paper: Coherence Metrics (Lower is Better) (for single seed)

    | Method | Negation Violation (%) | Transitivity Violation (%) |
    """
    table = "\n" + "="*80 + "\n"
    table += "TABLE 2: COHERENCE METRICS (Lower is Better)\n"
    table += "="*80 + "\n"
    table += f"{'Method':<25} {'Negation Viol (%)':>20} {'Transitivity Viol (%)':>20}\n"
    table += "-"*80 + "\n"

    methods = [
        ('greedy', 'Greedy'),
        ('beam_search', 'Beam Search'),
        ('self_consistency', 'Self-Consistency'),
        ('fgbs', 'FGBS (Ours)')
    ]

    for method_key, method_name in methods:
        if method_key not in coherence_results:
            continue

        result = coherence_results[method_key]
        row = f"{method_name:<25}"

        # Negation violation
        if 'negation_invariance' in result:
            neg_viol = result['negation_invariance']['violation_rate'] * 100
            row += f"{neg_viol:>20.1f}"
        else:
            row += f"{'N/A':>20}"

        # Transitivity violation
        if 'transitivity' in result:
            trans_viol = result['transitivity']['violation_rate'] * 100
            row += f"{trans_viol:>20.1f}"
        else:
            row += f"{'N/A':>20}"

        table += row + "\n"

    table += "="*80 + "\n"
    return table


def save_all_results(results: Dict[str, Any], output_dir: Path):
    """Save all experimental results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"full_results_{timestamp}.json"

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Full results saved to: {json_path}")

    # Save tables as text
    tables_path = output_dir / f"tables_{timestamp}.txt"

    with open(tables_path, 'w') as f:
        f.write("PAPER RESULTS REPRODUCTION\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        if 'table_1' in results:
            f.write(results['table_1'])
            f.write("\n\n")

        if 'table_2' in results:
            f.write(results['table_2'])

    logger.info(f"Tables saved to: {tables_path}")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(
        description="Reproduce all paper results (Tables 1 and 2) with multi-seed support"
    )

    parser.add_argument(
        '--model',
        type=str,
        default='mistralai/Mistral-7B-Instruct-v0.3',
        help='Model name (single model)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Multiple models to test (space-separated)'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42],
        help='Random seeds for multiple runs (space-separated, e.g., 42 123 456)'
    )
    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Run quick test with 10 examples each'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full experiment (as in paper)'
    )
    parser.add_argument(
        '--num_examples',
        type=int,
        default=100,
        help='Number of examples per benchmark'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='FGBS alpha parameter'
    )
    parser.add_argument(
        '--bond_dim',
        type=int,
        default=16,
        help='FGBS bond dimension'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Config file'
    )
    parser.add_argument(
        '--skip_coherence',
        action='store_true',
        help='Skip Table 2 (coherence metrics) evaluation to save time'
    )
    parser.add_argument(
        '--benchmarks',
        type=str,
        nargs='+',
        default=['gsm8k', 'strategyqa'],
        help='Benchmarks to run (space-separated, e.g., gsm8k strategyqa)'
    )

    args = parser.parse_args()

    # Determine models to test
    if args.models:
        models_to_test = args.models
    else:
        models_to_test = [args.model]

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Update config with args
    config['fgbs']['alpha'] = args.alpha
    config['fgbs']['bond_dim'] = args.bond_dim

    if args.quick_test:
        config['experiment']['num_examples'] = 10
    elif args.full:
        config['experiment']['num_examples'] = -1  # All examples
    else:
        config['experiment']['num_examples'] = args.num_examples

    # Setup logging
    setup_logger(log_level='INFO')

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track all experiments
    experiment_manifest = {
        'models': models_to_test,
        'seeds': args.seeds,
        'benchmarks': args.benchmarks,
        'config': config,
        'timestamp': datetime.now().isoformat(),
    }

    # Save experiment manifest
    with open(output_dir / 'experiment_manifest.json', 'w') as f:
        json.dump(experiment_manifest, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info(f"{'='*80}")
    logger.info(f"Models: {models_to_test}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Benchmarks: {args.benchmarks}")
    logger.info(f"Number of examples: {config['experiment']['num_examples']}")
    logger.info(f"Total experiments: {len(models_to_test) * len(args.seeds) * len(args.benchmarks) * 4}")
    logger.info(f"{'='*80}\n")

    # Run experiments for each model
    for model_name in models_to_test:
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING MODEL: {model_name}")
        logger.info(f"{'='*80}\n")

        # Update config with current model
        config['model']['name'] = model_name

        # Load model
        model_cfg = config.get('model', {})
        device_target = model_cfg.get('device', 'auto')
        torch_dtype_name = model_cfg.get('torch_dtype')
        load_in_8bit = model_cfg.get('load_in_8bit', False)

        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            device=device_target,
            torch_dtype_name=torch_dtype_name,
            load_in_8bit=load_in_8bit,
        )

        # Run with multiple seeds
        for seed in args.seeds:
            logger.info(f"\n{'='*80}")
            logger.info(f"RUNNING WITH SEED: {seed}")
            logger.info(f"{'='*80}\n")

            # Table 1: Accuracy benchmarks
            for benchmark in args.benchmarks:
                try:
                    bench_results = run_single_benchmark_all_methods(
                        benchmark,
                        model,
                        tokenizer,
                        config,
                        seed,
                        model_name,
                        output_dir
                    )

                    # Clear GPU memory between benchmarks
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()
                    logger.info(f"Memory cleared after {benchmark.upper()} benchmark")
                except Exception as e:
                    logger.error(f"Failed to run {benchmark}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

            # Table 2: Coherence metrics
            if not args.skip_coherence:
                try:
                    coherence_results = run_coherence_evaluation_all_methods(
                        model,
                        tokenizer,
                        config,
                        seed,
                        output_dir
                    )

                    # Clear GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()
                except Exception as e:
                    logger.error(f"Failed to run coherence evaluation: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

        # Clean up model
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    # Generate aggregated results with statistics
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING AGGREGATED RESULTS")
    logger.info(f"{'='*80}\n")

    methods = ['greedy', 'beam_search', 'self_consistency', 'fgbs']

    # Generate Table 1 with statistics
    logger.info("Generating Table 1 (Accuracy) with statistics...")
    table_1 = generate_accuracy_table(
        output_dir,
        methods,
        args.benchmarks,
        models_to_test,
        baseline_method='greedy',
        output_format='markdown'
    )
    print("\n" + table_1)

    # Save Table 1
    with open(output_dir / 'table_1_accuracy.txt', 'w') as f:
        f.write(table_1)

    # Generate LaTeX version
    table_1_latex = generate_accuracy_table(
        output_dir,
        methods,
        args.benchmarks,
        models_to_test,
        baseline_method='greedy',
        output_format='latex'
    )
    with open(output_dir / 'table_1_accuracy.tex', 'w') as f:
        f.write(table_1_latex)

    # Generate Table 2 with statistics
    if not args.skip_coherence:
        logger.info("Generating Table 2 (Coherence) with statistics...")
        table_2 = generate_coherence_table(
            output_dir,
            methods,
            output_format='markdown'
        )
        print("\n" + table_2)

        # Save Table 2
        with open(output_dir / 'table_2_coherence.txt', 'w') as f:
            f.write(table_2)

        # Generate LaTeX version
        table_2_latex = generate_coherence_table(
            output_dir,
            methods,
            output_format='latex'
        )
        with open(output_dir / 'table_2_coherence.tex', 'w') as f:
            f.write(table_2_latex)

    # Generate summary statistics
    logger.info("Generating summary statistics...")
    summary_stats = generate_summary_stats(output_dir)
    print(f"\nSummary Statistics:\n{json.dumps(summary_stats, indent=2)}")

    with open(output_dir / 'summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)

    # Save comprehensive results summary
    save_results_summary(
        output_dir,
        output_dir / 'results_summary.json',
        methods,
        args.benchmarks,
        models_to_test
    )

    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - Aggregated tables: {output_dir}/table_1_accuracy.txt, table_2_coherence.txt")
    print(f"  - LaTeX tables: {output_dir}/table_1_accuracy.tex, table_2_coherence.tex")
    print(f"  - Summary statistics: {output_dir}/summary_stats.json")
    print(f"  - Full summary: {output_dir}/results_summary.json")
    print(f"  - Individual results: {output_dir}/*.json")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
