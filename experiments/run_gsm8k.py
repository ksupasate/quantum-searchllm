#!/usr/bin/env python3
"""
GSM8K Benchmark Experiment

Evaluates TNAD/FGBS on the GSM8K mathematical reasoning benchmark.

GSM8K Dataset:
    - 8,500 grade school math word problems
    - Requires multi-step arithmetic reasoning
    - Ground truth answers provided

Usage:
    python experiments/run_gsm8k.py --config configs/default.yaml
    python experiments/run_gsm8k.py --alpha 0.5 --bond_dim 16 --beam_width 5
"""

import argparse
import json
import re
import yaml
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

# Add parent directory to path to import tnad
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
from loguru import logger

from tnad import FidelityGuidedBeamSearcher
from tnad.utils import setup_logger, get_device
from experiments.baselines import SelfConsistency


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_answer_from_text(text: str, regex_pattern: str = r"####\s*(\d+)") -> Optional[int]:
    """
    Extract numerical answer from generated text.

    GSM8K format: "#### 42" at the end of solution.

    Args:
        text: Generated solution text
        regex_pattern: Regex pattern to extract answer

    Returns:
        Extracted answer as integer, or None if not found
    """
    match = re.search(regex_pattern, text)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, IndexError):
            return None

    # Fallback: try to find last number in text
    numbers = re.findall(r'\d+', text)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            return None

    return None


def format_prompt(question: str, prompt_template: str) -> str:
    """Format GSM8K question with prompt template."""
    return prompt_template.format(question=question)


def evaluate_single_example(
    searcher: FidelityGuidedBeamSearcher,
    question: str,
    ground_truth_answer: int,
    config: Dict[str, Any],
    example_id: int,
) -> Dict[str, Any]:
    """
    Evaluate FGBS on a single GSM8K example.

    Args:
        searcher: FGBS instance
        question: GSM8K question
        ground_truth_answer: Ground truth numerical answer
        config: Configuration dict
        example_id: Example index for logging

    Returns:
        Result dictionary with metrics
    """
    # Format prompt
    prompt_template = config['dataset']['gsm8k']['prompt_template']
    prompt = format_prompt(question, prompt_template)

    # Generate solution
    gen_config = config['generation']
    try:
        result = searcher.generate(
            prompt,
            max_length=gen_config['max_length'],
            min_length=gen_config['min_length'],
            return_details=gen_config['return_details'],
            show_progress=False,  # Disable for batch processing
        )
    except Exception as e:
        logger.error(f"Generation failed for example {example_id}: {e}")
        return {
            'example_id': example_id,
            'question': question,
            'generated_text': "",
            'predicted_answer': None,
            'ground_truth_answer': ground_truth_answer,
            'correct': False,
            'error': str(e),
        }

    # Extract predicted answer
    generated_text = result['text']
    extract_regex = config['dataset']['gsm8k']['extract_answer_regex']
    predicted_answer = extract_answer_from_text(generated_text, extract_regex)

    # Check correctness
    correct = (predicted_answer == ground_truth_answer)

    # Prepare result
    eval_result = {
        'example_id': example_id,
        'question': question,
        'prompt': prompt,
        'generated_text': generated_text,
        'predicted_answer': predicted_answer,
        'ground_truth_answer': ground_truth_answer,
        'correct': correct,
        'log_prob': result['log_prob'],
        'log_cfs': result['log_cfs'],
        'composite_score': result['composite_score'],
    }

    # Add detailed metrics if available
    if gen_config['return_details']:
        eval_result['cfs_trajectory'] = result.get('cfs_trajectory', [])
        eval_result['score_trajectory'] = result.get('score_trajectory', [])

    return eval_result


def _load_gsm8k_dataset(config: Dict[str, Any]):
    logger.info("Loading GSM8K dataset")
    dataset = load_dataset("gsm8k", "main", split=config['dataset']['split'])
    num_examples = config['experiment']['num_examples']
    if num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
    logger.info(f"Evaluating on {len(dataset)} examples")
    return dataset


def _load_gsm8k_model_and_tokenizer(config: Dict[str, Any]):
    model_config = config['model']
    logger.info(f"Loading model: {model_config['name']}")

    tokenizer = AutoTokenizer.from_pretrained(model_config['name'])

    quant_config = None
    if model_config.get('load_in_8bit', False):
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs = {
        'torch_dtype': getattr(torch, model_config['torch_dtype']),
        'quantization_config': quant_config,
    }
    if quant_config is not None:
        model_kwargs['device_map'] = 'auto'

    model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        **model_kwargs,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_config['device'] == 'auto':
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(model_config['device'])

    logger.info(f"Using device: {device}")

    return model, tokenizer, device


def run_gsm8k_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run complete GSM8K benchmark experiment with FGBS."""
    logger.info("Starting GSM8K experiment")

    dataset = _load_gsm8k_dataset(config)
    model, tokenizer, device = _load_gsm8k_model_and_tokenizer(config)

    # Initialize FGBS
    fgbs_config = config['fgbs']
    searcher = FidelityGuidedBeamSearcher(
        model=model,
        tokenizer=tokenizer,
        beam_width=fgbs_config['beam_width'],
        alpha=fgbs_config['alpha'],
        bond_dim=fgbs_config['bond_dim'],
        top_k=fgbs_config['top_k'],
        temperature=fgbs_config['temperature'],
        device=device,
        normalize_embeddings=fgbs_config['normalize_embeddings'],
    )

    logger.info(f"FGBS Configuration: B={fgbs_config['beam_width']}, "
                f"α={fgbs_config['alpha']}, χ={fgbs_config['bond_dim']}")

    # Evaluate on dataset
    results = []
    correct_count = 0

    for idx, example in enumerate(tqdm(dataset, desc="Evaluating GSM8K")):
        question = example['question']
        # GSM8K answer format: "#### 42"
        answer_text = example['answer']
        ground_truth = extract_answer_from_text(answer_text)

        if ground_truth is None:
            logger.warning(f"Could not extract ground truth for example {idx}")
            continue

        # Evaluate
        eval_result = evaluate_single_example(
            searcher=searcher,
            question=question,
            ground_truth_answer=ground_truth,
            config=config,
            example_id=idx,
        )

        results.append(eval_result)

        if eval_result['correct']:
            correct_count += 1

        # Log progress
        if (idx + 1) % 10 == 0:
            current_accuracy = correct_count / len(results)
            logger.info(f"Progress: {idx + 1}/{len(dataset)}, "
                       f"Accuracy: {current_accuracy:.2%}")

    # Compute final metrics
    accuracy = correct_count / len(results) if results else 0.0

    # Aggregate CFS metrics
    final_cfs_values = [r['log_cfs'] for r in results if 'log_cfs' in r]
    avg_final_cfs = np.mean(final_cfs_values) if final_cfs_values else 0.0

    # Prepare experiment results
    experiment_results = {
        'config': config,
        'dataset': {
            'name': 'gsm8k',
            'split': config['dataset']['split'],
            'num_examples': len(results),
        },
        'metrics': {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': len(results),
            'avg_log_cfs': avg_final_cfs,
            'avg_cfs': np.exp(avg_final_cfs),
        },
        'predictions': results,
        'timestamp': datetime.now().isoformat(),
    }

    logger.info(f"Experiment complete!")
    logger.info(f"Final Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
    logger.info(f"Average CFS: {np.exp(avg_final_cfs):.2f}")

    return experiment_results


def run_gsm8k_self_consistency_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run GSM8K with the self-consistency baseline."""
    logger.info("Starting GSM8K self-consistency experiment")

    dataset = _load_gsm8k_dataset(config)
    model, tokenizer, device = _load_gsm8k_model_and_tokenizer(config)

    sc_cfg = config.get('baselines', {}).get('self_consistency', {})
    num_samples = sc_cfg.get('num_samples', 10)
    temperature = sc_cfg.get('temperature', 0.7)

    decoder = SelfConsistency(
        model=model,
        tokenizer=tokenizer,
        num_samples=num_samples,
        temperature=temperature,
        device=str(device),
    )

    prompt_template = config['dataset']['gsm8k']['prompt_template']
    extract_regex = config['dataset']['gsm8k']['extract_answer_regex']
    gen_config = config['generation']

    results: List[Dict[str, Any]] = []
    correct = 0

    for idx, example in enumerate(tqdm(dataset, desc="Evaluating GSM8K (Self-Consistency)")):
        question = example['question']
        answer_text = example['answer']
        ground_truth = extract_answer_from_text(answer_text)

        if ground_truth is None:
            logger.warning(f"Could not extract ground truth for example {idx}")
            continue

        prompt = format_prompt(question, prompt_template)

        try:
            sc_result = decoder.generate(
                prompt,
                max_length=gen_config['max_length'],
                min_length=gen_config['min_length'],
                show_progress=False,
                answer_extractor=lambda text: extract_answer_from_text(text, extract_regex),
            )
        except Exception as exc:
            logger.error(f"Self-consistency generation failed for example {idx}: {exc}")
            results.append({
                'example_id': idx,
                'question': question,
                'prompt': prompt,
                'generated_text': "",
                'predicted_answer': None,
                'ground_truth_answer': ground_truth,
                'correct': False,
                'error': str(exc),
            })
            continue

        predicted = sc_result.get('majority_answer')
        correct_flag = predicted == ground_truth
        if correct_flag:
            correct += 1

        results.append({
            'example_id': idx,
            'question': question,
            'prompt': prompt,
            'generated_text': sc_result.get('text', ""),
            'predicted_answer': predicted,
            'ground_truth_answer': ground_truth,
            'correct': correct_flag,
            'all_samples': sc_result.get('all_samples', []),
            'all_answers': sc_result.get('all_answers'),
            'confidence': sc_result.get('confidence'),
            'answer_counts': sc_result.get('answer_counts'),
            'method': sc_result.get('method'),
        })

    accuracy = correct / len(results) if results else 0.0

    experiment_results = {
        'config': config,
        'dataset': {
            'name': 'gsm8k',
            'split': config['dataset']['split'],
            'num_examples': len(results),
        },
        'metrics': {
            'accuracy': accuracy,
            'correct_count': correct,
            'total_count': len(results),
        },
        'predictions': results,
        'timestamp': datetime.now().isoformat(),
    }

    logger.info(
        f"Self-consistency experiment complete! Accuracy: {accuracy:.2%} "
        f"({correct}/{len(results)})"
    )

    return experiment_results


def save_results(results: Dict[str, Any], output_path: Path):
    """Save experiment results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def main():
    """Main entry point for GSM8K experiment."""
    parser = argparse.ArgumentParser(description="Run TNAD on GSM8K benchmark")

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration YAML file'
    )

    # Override specific parameters
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--alpha', type=float, help='Alpha parameter')
    parser.add_argument('--bond_dim', type=int, help='Bond dimension')
    parser.add_argument('--beam_width', type=int, help='Beam width')
    parser.add_argument('--num_examples', type=int, help='Number of examples to evaluate')
    parser.add_argument('--device', type=str, help='Device (cpu/cuda/mps)')

    # Output
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for results (default: auto-generated)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command-line arguments
    if args.model:
        config['model']['name'] = args.model
    if args.alpha is not None:
        config['fgbs']['alpha'] = args.alpha
    if args.bond_dim:
        config['fgbs']['bond_dim'] = args.bond_dim
    if args.beam_width:
        config['fgbs']['beam_width'] = args.beam_width
    if args.num_examples:
        config['experiment']['num_examples'] = args.num_examples
    if args.device:
        config['model']['device'] = args.device

    # Setup logging
    setup_logger(
        log_level=config['experiment']['log_level'],
        log_file=config['experiment']['log_file'],
    )

    # Set random seed for reproducibility
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Run experiment
    results = run_gsm8k_experiment(config)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate output filename
        results_dir = Path(config['experiment']['results_dir'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alpha = config['fgbs']['alpha']
        chi = config['fgbs']['bond_dim']
        output_path = results_dir / f"gsm8k_alpha{alpha}_chi{chi}_{timestamp}.json"

    if config['experiment']['save_results']:
        save_results(results, output_path)

    # Print summary
    print("\n" + "="*60)
    print("GSM8K EXPERIMENT RESULTS")
    print("="*60)
    print(f"Model: {config['model']['name']}")
    print(f"FGBS Config: α={config['fgbs']['alpha']}, "
          f"χ={config['fgbs']['bond_dim']}, "
          f"B={config['fgbs']['beam_width']}")
    print(f"Examples: {results['metrics']['total_count']}")
    print(f"Accuracy: {results['metrics']['accuracy']:.2%}")
    print(f"Average CFS: {results['metrics']['avg_cfs']:.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
