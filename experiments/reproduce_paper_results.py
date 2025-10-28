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
- FGBS (α=0.5, χ=16) [Our method]

Usage:
    # Quick test (10 examples each)
    python experiments/reproduce_paper_results.py --quick_test

    # Full reproduction (as in paper)
    python experiments/reproduce_paper_results.py --full

    # Custom configuration
    python experiments/reproduce_paper_results.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --num_examples 100 --alpha 0.5 --bond_dim 16
"""

import argparse
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
        'dtype': dtype,
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
                'dtype': torch.float32,
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


def run_single_benchmark_all_methods(
    benchmark_name: str,
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a single benchmark with all methods (baselines + FGBS).

    Args:
        benchmark_name: 'gsm8k', 'strategyqa', or 'entailmentbank'
        model: LLM model
        tokenizer: Tokenizer
        config: Configuration dict

    Returns:
        Dictionary with results from all methods
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {benchmark_name.upper()} with all methods")
    logger.info(f"{'='*60}\n")

    device = get_device()
    results = {}

    # 1. Greedy Decoding
    logger.info("Running Greedy Decoding...")
    greedy = GreedyDecoder(model, tokenizer, device)
    greedy_config = config.copy()
    # Keep all FGBS parameters but override specific ones for greedy
    greedy_config['fgbs'] = {
        'beam_width': 1,
        'alpha': 1.0,
        'bond_dim': 8,
        'top_k': config['fgbs']['top_k'],
        'temperature': config['fgbs']['temperature'],
        'normalize_embeddings': config['fgbs']['normalize_embeddings'],
    }

    if benchmark_name == 'gsm8k':
        # Modify the run function to use greedy decoder
        # For simplicity, we'll run with FGBS but α=1, B=1 (equivalent to greedy)
        greedy_config['fgbs']['beam_width'] = 1
        greedy_config['fgbs']['alpha'] = 1.0
        results['greedy'] = run_gsm8k_experiment(greedy_config)
    elif benchmark_name == 'strategyqa':
        greedy_config['fgbs']['beam_width'] = 1
        greedy_config['fgbs']['alpha'] = 1.0
        results['greedy'] = run_strategyqa_experiment(greedy_config)

    # 2. Standard Beam Search (B=5)
    logger.info("Running Standard Beam Search (B=5)...")
    beam_config = config.copy()
    # Keep all FGBS parameters but override specific ones for beam search
    beam_config['fgbs'] = {
        'beam_width': 5,
        'alpha': 1.0,  # Pure LLM probability
        'bond_dim': config['fgbs']['bond_dim'],
        'top_k': config['fgbs']['top_k'],
        'temperature': config['fgbs']['temperature'],
        'normalize_embeddings': config['fgbs']['normalize_embeddings'],
    }

    if benchmark_name == 'gsm8k':
        results['beam_search'] = run_gsm8k_experiment(beam_config)
    elif benchmark_name == 'strategyqa':
        results['beam_search'] = run_strategyqa_experiment(beam_config)

    # 3. Self-Consistency (N=10)
    logger.info("Running Self-Consistency (N=10)...")
    sc_config = config.copy()

    if benchmark_name == 'gsm8k':
        results['self_consistency'] = run_gsm8k_self_consistency_experiment(sc_config)
    elif benchmark_name == 'strategyqa':
        results['self_consistency'] = run_strategyqa_self_consistency_experiment(sc_config)
    else:
        logger.info("Self-Consistency runner not implemented for this benchmark.")
        results['self_consistency'] = {'note': 'Self-Consistency not implemented for this benchmark'}

    # 4. FGBS (Our Method)
    logger.info(f"Running FGBS (α={config['fgbs']['alpha']}, χ={config['fgbs']['bond_dim']})...")
    fgbs_config = config.copy()

    if benchmark_name == 'gsm8k':
        results['fgbs'] = run_gsm8k_experiment(fgbs_config)
    elif benchmark_name == 'strategyqa':
        results['fgbs'] = run_strategyqa_experiment(fgbs_config)

    return results


def run_coherence_evaluation_all_methods(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run coherence evaluation (Table 2) with all methods.

    Args:
        model: LLM model
        tokenizer: Tokenizer
        config: Configuration dict

    Returns:
        Dictionary with coherence metrics for all methods
    """
    logger.info(f"\n{'='*60}")
    logger.info("Running Coherence Evaluation (Table 2)")
    logger.info(f"{'='*60}\n")

    device = get_device()
    results = {}

    # 1. Greedy
    logger.info("Evaluating Greedy Decoding coherence...")
    greedy = GreedyDecoder(model, tokenizer, device)
    results['greedy'] = run_coherence_evaluation(greedy, model, tokenizer)

    # 2. Beam Search
    logger.info("Evaluating Beam Search coherence...")
    beam_search = StandardBeamSearch(model, tokenizer, beam_width=5, device=device)
    results['beam_search'] = run_coherence_evaluation(beam_search, model, tokenizer)

    # 3. Self-Consistency
    logger.info("Evaluating Self-Consistency coherence...")
    self_consistency = SelfConsistency(model, tokenizer, num_samples=10, device=device)
    results['self_consistency'] = run_coherence_evaluation(self_consistency, model, tokenizer)

    # 4. FGBS
    logger.info("Evaluating FGBS coherence...")
    fgbs = FidelityGuidedBeamSearcher(
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
    results['fgbs'] = run_coherence_evaluation(fgbs, model, tokenizer)

    return results


def generate_table_1(all_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate Table 1 from paper: Accuracy Results

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


def generate_table_2(coherence_results: Dict[str, Any]) -> str:
    """
    Generate Table 2 from paper: Coherence Metrics (Lower is Better)

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
        description="Reproduce all paper results (Tables 1 and 2)"
    )

    parser.add_argument(
        '--model',
        type=str,
        default='mistralai/Mistral-7B-Instruct-v0.3',
        help='Model name'
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
        default='./paper_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Config file'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Update config with args
    config['model']['name'] = args.model
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

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    model_cfg = config.get('model', {})
    model_name = model_cfg.get('name', args.model)
    device_target = model_cfg.get('device', 'auto')
    torch_dtype_name = model_cfg.get('torch_dtype')
    load_in_8bit = model_cfg.get('load_in_8bit', False)

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device_target,
        torch_dtype_name=torch_dtype_name,
        load_in_8bit=load_in_8bit,
    )

    # Run all experiments
    all_results = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
    }

    # Table 1: Accuracy benchmarks
    logger.info("\n" + "="*80)
    logger.info("REPRODUCING TABLE 1: ACCURACY RESULTS")
    logger.info("="*80 + "\n")

    for benchmark in ['gsm8k', 'strategyqa']:  # Add 'entailmentbank' if available
        try:
            bench_results = run_single_benchmark_all_methods(
                benchmark,
                model,
                tokenizer,
                config
            )
            all_results[benchmark] = bench_results
        except Exception as e:
            logger.error(f"Failed to run {benchmark}: {e}")

    # Generate Table 1
    all_results['table_1'] = generate_table_1(all_results)
    print(all_results['table_1'])

    # Table 2: Coherence metrics
    logger.info("\n" + "="*80)
    logger.info("REPRODUCING TABLE 2: COHERENCE METRICS")
    logger.info("="*80 + "\n")

    try:
        coherence_results = run_coherence_evaluation_all_methods(
            model,
            tokenizer,
            config
        )
        all_results['coherence'] = coherence_results

        # Generate Table 2
        all_results['table_2'] = generate_table_2(coherence_results)
        print(all_results['table_2'])

    except Exception as e:
        logger.error(f"Failed to run coherence evaluation: {e}")

    # Save results
    output_dir = Path(args.output_dir)
    save_all_results(all_results, output_dir)

    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print("\nTo view results:")
    print(f"  - Tables: {output_dir}/tables_*.txt")
    print(f"  - Full JSON: {output_dir}/full_results_*.json")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
