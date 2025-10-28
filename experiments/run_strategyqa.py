#!/usr/bin/env python3
"""
StrategyQA Benchmark Experiment

Evaluates TNAD/FGBS on the StrategyQA commonsense reasoning benchmark.

StrategyQA Dataset:
    - Yes/No questions requiring implicit multi-hop reasoning
    - Tests commonsense knowledge and logical inference
    - 2,780 examples

Usage:
    python experiments/run_strategyqa.py --config configs/default.yaml
    python experiments/run_strategyqa.py --alpha 0.5 --bond_dim 16
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
from datasets import Dataset, load_dataset
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

from tnad import FidelityGuidedBeamSearcher
from tnad.utils import get_device, setup_logger
from experiments.baselines import SelfConsistency


def _load_strategyqa_local(path: Path) -> Dataset:
    """Load a local JSON/JSONL StrategyQA sample."""
    if not path.exists():
        raise FileNotFoundError(f"Local StrategyQA file not found: {path}")

    records: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            answer = sample.get('answer')
            if isinstance(answer, str):
                sample['answer'] = answer.strip().lower() in {'yes', 'true', '1'}
            records.append(sample)

    if not records:
        raise ValueError(f"No records found in local StrategyQA file: {path}")

    return Dataset.from_list(records)


def load_strategyqa_dataset(config: Dict[str, Any]) -> Dataset:
    """Load StrategyQA with graceful fallbacks (Hub → local sample)."""
    dataset_cfg = config['dataset'].get('strategyqa', {})
    split = config['dataset'].get('split', 'validation')
    hub_ids = dataset_cfg.get('hub_ids', [])

    for hub_id in hub_ids:
        try:
            logger.info(f"Loading StrategyQA from Hub dataset '{hub_id}' (split={split})")
            return load_dataset(hub_id, split=split)
        except Exception as err:
            logger.warning(f"Unable to load StrategyQA dataset '{hub_id}': {err}")

    local_path = dataset_cfg.get('local_path')
    if local_path:
        path = Path(local_path)
        try:
            logger.info(f"Loading StrategyQA from local file: {path}")
            return _load_strategyqa_local(path)
        except Exception as err:
            logger.error(f"Failed to load local StrategyQA file '{path}': {err}")

    raise RuntimeError(
        "Could not load StrategyQA dataset from Hub or local fallback. "
        "Check dataset configuration or provide a local sample."
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_yes_no_answer(text: str) -> Optional[str]:
    """
    Extract yes/no answer from generated text.

    Looks for explicit "yes" or "no" in the generated response,
    typically at the end of the reasoning chain.
    """
    text_lower = text.lower()

    # Look for explicit answer patterns
    patterns = [
        r'(?:answer|conclusion|therefore)(?:\s+is)?:\s*(yes|no)',
        r'(?:^|\n)\s*(yes|no)\s*[.,!]?\s*$',
        r'the answer is\s*(yes|no)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower, re.MULTILINE)
        if match:
            return match.group(1).lower()

    # Count occurrences and take majority
    yes_count = len(re.findall(r'\byes\b', text_lower))
    no_count = len(re.findall(r'\bno\b', text_lower))

    if yes_count > no_count:
        return 'yes'
    elif no_count > yes_count:
        return 'no'

    return None


def format_prompt(question: str, prompt_template: str) -> str:
    """Format StrategyQA question with prompt template."""
    return prompt_template.format(question=question)


def evaluate_single_example(
    searcher: FidelityGuidedBeamSearcher,
    question: str,
    ground_truth_answer: bool,
    config: Dict[str, Any],
    example_id: int,
) -> Dict[str, Any]:
    """
    Evaluate FGBS on a single StrategyQA example.

    Args:
        searcher: FGBS instance
        question: StrategyQA question
        ground_truth_answer: Ground truth boolean
        config: Configuration dict
        example_id: Example index

    Returns:
        Result dictionary with metrics
    """
    if isinstance(ground_truth_answer, str):
        ground_truth_answer = ground_truth_answer.strip().lower() in {'yes', 'true', '1'}

    # Format prompt
    prompt_template = config['dataset']['strategyqa']['prompt_template']
    prompt = format_prompt(question, prompt_template)

    # Generate solution
    gen_config = config['generation']
    try:
        result = searcher.generate(
            prompt,
            max_length=gen_config['max_length'],
            min_length=gen_config['min_length'],
            return_details=gen_config['return_details'],
            show_progress=False,
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
    predicted_answer_str = extract_yes_no_answer(generated_text)

    # Convert to boolean
    predicted_answer = None
    if predicted_answer_str == 'yes':
        predicted_answer = True
    elif predicted_answer_str == 'no':
        predicted_answer = False

    # Check correctness
    correct = (predicted_answer == ground_truth_answer) if predicted_answer is not None else False

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


def _load_strategyqa_resources(config: Dict[str, Any]):
    logger.info("Loading StrategyQA dataset")
    dataset = load_strategyqa_dataset(config)

    num_examples = config['experiment']['num_examples']
    if num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    logger.info(f"Evaluating on {len(dataset)} examples")

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

    return dataset, model, tokenizer, device


def run_strategyqa_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run complete StrategyQA benchmark experiment."""
    logger.info("Starting StrategyQA experiment")

    dataset, model, tokenizer, device = _load_strategyqa_resources(config)

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

    for idx, example in enumerate(tqdm(dataset, desc="Evaluating StrategyQA")):
        question = example['question']
        ground_truth = example['answer']
        if isinstance(ground_truth, str):
            ground_truth = ground_truth.strip().lower() in {'yes', 'true', '1'}

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
            'name': 'strategyqa',
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


def run_strategyqa_self_consistency_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run StrategyQA evaluation with the self-consistency baseline."""
    logger.info("Starting StrategyQA self-consistency experiment")

    dataset, model, tokenizer, device = _load_strategyqa_resources(config)

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

    prompt_template = config['dataset']['strategyqa']['prompt_template']
    gen_config = config['generation']

    results: List[Dict[str, Any]] = []
    correct = 0

    for idx, example in enumerate(tqdm(dataset, desc="Evaluating StrategyQA (Self-Consistency)")):
        question = example['question']
        ground_truth = example['answer']
        if isinstance(ground_truth, str):
            ground_truth_bool = ground_truth.strip().lower() in {'yes', 'true', '1'}
        else:
            ground_truth_bool = bool(ground_truth)

        prompt = format_prompt(question, prompt_template)

        try:
            sc_result = decoder.generate(
                prompt,
                max_length=gen_config['max_length'],
                min_length=gen_config['min_length'],
                show_progress=False,
                answer_extractor=extract_yes_no_answer,
            )
        except Exception as exc:
            logger.error(f"Self-consistency generation failed for example {idx}: {exc}")
            results.append({
                'example_id': idx,
                'question': question,
                'prompt': prompt,
                'generated_text': "",
                'predicted_answer': None,
                'ground_truth_answer': ground_truth_bool,
                'correct': False,
                'error': str(exc),
            })
            continue

        majority_answer = sc_result.get('majority_answer')
        if isinstance(majority_answer, str):
            predicted_bool = majority_answer.strip().lower() == 'yes'
        else:
            predicted_bool = bool(majority_answer) if majority_answer is not None else None

        correct_flag = predicted_bool == ground_truth_bool if predicted_bool is not None else False
        if correct_flag:
            correct += 1

        results.append({
            'example_id': idx,
            'question': question,
            'prompt': prompt,
            'generated_text': sc_result.get('text', ""),
            'predicted_answer': predicted_bool,
            'ground_truth_answer': ground_truth_bool,
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
            'name': 'strategyqa',
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
    """Main entry point for StrategyQA experiment."""
    parser = argparse.ArgumentParser(description="Run TNAD on StrategyQA benchmark")

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
    parser.add_argument('--num_examples', type=int, help='Number of examples')
    parser.add_argument('--device', type=str, help='Device')
    parser.add_argument('--output', type=str, help='Output path for results')

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

    # Set random seed
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Run experiment
    results = run_strategyqa_experiment(config)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path(config['experiment']['results_dir'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alpha = config['fgbs']['alpha']
        chi = config['fgbs']['bond_dim']
        output_path = results_dir / f"strategyqa_alpha{alpha}_chi{chi}_{timestamp}.json"

    if config['experiment']['save_results']:
        save_results(results, output_path)

    # Print summary
    print("\n" + "="*60)
    print("STRATEGYQA EXPERIMENT RESULTS")
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
