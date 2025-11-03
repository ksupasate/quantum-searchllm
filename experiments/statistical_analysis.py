"""
Statistical Analysis Module for TNAD Experiments

Provides functions for aggregating multi-seed results, computing statistical
significance, and generating publication-ready tables.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import pandas as pd


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """
    Compute mean and standard deviation of a list of values.

    Args:
        values: List of numerical values

    Returns:
        Tuple of (mean, std)
    """
    if not values:
        return 0.0, 0.0

    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval.

    Args:
        values: List of numerical values
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not values or len(values) < 2:
        mean_val = values[0] if values else 0.0
        return mean_val, mean_val, mean_val

    arr = np.array(values)
    mean = np.mean(arr)
    sem = stats.sem(arr)
    interval = sem * stats.t.ppf((1 + confidence) / 2., len(arr) - 1)

    return float(mean), float(mean - interval), float(mean + interval)


def run_ttest(
    baseline_values: List[float],
    method_values: List[float],
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Run independent samples t-test between baseline and method.

    Args:
        baseline_values: Results from baseline method
        method_values: Results from comparison method
        alternative: Type of test ('two-sided', 'greater', 'less')

    Returns:
        Tuple of (t_statistic, p_value)
    """
    if len(baseline_values) < 2 or len(method_values) < 2:
        return 0.0, 1.0

    t_stat, p_val = stats.ttest_ind(
        method_values,
        baseline_values,
        alternative=alternative
    )

    return float(t_stat), float(p_val)


def format_mean_std(
    mean: float,
    std: float,
    decimals: int = 1,
    percentage: bool = True
) -> str:
    """
    Format mean ± std for display.

    Args:
        mean: Mean value
        std: Standard deviation
        decimals: Number of decimal places
        percentage: If True, display as percentage

    Returns:
        Formatted string like "65.3 ± 1.2"
    """
    if percentage:
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
    else:
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def add_significance_marker(p_value: float) -> str:
    """
    Add significance markers based on p-value.

    Args:
        p_value: P-value from statistical test

    Returns:
        Significance marker: '***' (p<0.001), '**' (p<0.01), '*' (p<0.05), '' (n.s.)
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


def aggregate_results_by_seed(
    results_dir: Path,
    method_name: str,
    dataset_name: str,
    model_name: str,
    metric: str = 'accuracy'
) -> Dict:
    """
    Aggregate results across multiple seeds for a given method/dataset/model.

    Args:
        results_dir: Directory containing result JSON files
        method_name: Name of the method (e.g., 'fgbs', 'greedy')
        dataset_name: Name of dataset (e.g., 'gsm8k')
        model_name: Model identifier
        metric: Metric to aggregate (default: 'accuracy')

    Returns:
        Dictionary with aggregated statistics
    """
    # Find all matching result files
    pattern = f"{method_name}_{dataset_name}_*_seed*.json"
    result_files = list(results_dir.glob(pattern))

    if not result_files:
        return {
            'mean': 0.0,
            'std': 0.0,
            'values': [],
            'n_seeds': 0
        }

    values = []
    for result_file in result_files:
        # Check if it's for the right model
        if model_name.split('/')[-1] not in result_file.stem:
            continue

        with open(result_file, 'r') as f:
            data = json.load(f)
            if metric in data:
                values.append(data[metric])

    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'values': [],
            'n_seeds': 0
        }

    mean, std = compute_mean_std(values)

    return {
        'mean': mean,
        'std': std,
        'values': values,
        'n_seeds': len(values)
    }


def generate_accuracy_table(
    results_dir: Path,
    methods: List[str],
    datasets: List[str],
    models: List[str],
    baseline_method: str = 'greedy',
    output_format: str = 'markdown'
) -> str:
    """
    Generate publication-ready accuracy comparison table (Table 1).

    Args:
        results_dir: Directory containing results
        methods: List of method names
        datasets: List of dataset names
        models: List of model names
        baseline_method: Method to use as baseline for significance testing
        output_format: 'markdown', 'latex', or 'csv'

    Returns:
        Formatted table as string
    """
    # Aggregate results for all combinations
    table_data = []

    for model in models:
        model_short = model.split('/')[-1]
        for dataset in datasets:
            row = {'Model': model_short, 'Dataset': dataset.upper()}

            # Get baseline results
            baseline_stats = aggregate_results_by_seed(
                results_dir, baseline_method, dataset, model, 'accuracy'
            )
            baseline_values = baseline_stats['values']

            for method in methods:
                method_stats = aggregate_results_by_seed(
                    results_dir, method, dataset, model, 'accuracy'
                )

                # Format with mean ± std
                mean_str = format_mean_std(
                    method_stats['mean'] * 100,  # Convert to percentage
                    method_stats['std'] * 100,
                    decimals=1
                )

                # Add significance marker if not baseline
                if method != baseline_method and len(method_stats['values']) > 0:
                    _, p_val = run_ttest(baseline_values, method_stats['values'], alternative='greater')
                    marker = add_significance_marker(p_val)
                    mean_str += marker

                row[method.upper()] = mean_str

            table_data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(table_data)

    # Format based on output type
    if output_format == 'markdown':
        return df.to_markdown(index=False)
    elif output_format == 'latex':
        return df.to_latex(index=False, escape=False)
    elif output_format == 'csv':
        return df.to_csv(index=False)
    else:
        return str(df)


def generate_coherence_table(
    results_dir: Path,
    methods: List[str],
    output_format: str = 'markdown'
) -> str:
    """
    Generate coherence metrics comparison table (Table 2).

    Args:
        results_dir: Directory containing results
        methods: List of method names
        output_format: 'markdown', 'latex', or 'csv'

    Returns:
        Formatted table as string
    """
    table_data = []

    for method in methods:
        row = {'Method': method.upper()}

        # Look for coherence metric files
        coherence_files = list(results_dir.glob(f"coherence_{method}_*.json"))

        if coherence_files:
            neg_violations = []
            trans_violations = []

            for file in coherence_files:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if 'negation_violation_rate' in data:
                        neg_violations.append(data['negation_violation_rate'] * 100)
                    if 'transitivity_violation_rate' in data:
                        trans_violations.append(data['transitivity_violation_rate'] * 100)

            if neg_violations:
                mean, std = compute_mean_std(neg_violations)
                row['Negation Viol. (%)'] = format_mean_std(mean, std, decimals=1)

            if trans_violations:
                mean, std = compute_mean_std(trans_violations)
                row['Transitivity Viol. (%)'] = format_mean_std(mean, std, decimals=1)

        table_data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(table_data)

    # Format based on output type
    if output_format == 'markdown':
        return df.to_markdown(index=False)
    elif output_format == 'latex':
        return df.to_latex(index=False, escape=False)
    elif output_format == 'csv':
        return df.to_csv(index=False)
    else:
        return str(df)


def generate_summary_stats(results_dir: Path) -> Dict:
    """
    Generate summary statistics for all experiments.

    Args:
        results_dir: Directory containing results

    Returns:
        Dictionary with summary statistics
    """
    all_files = list(results_dir.glob("*.json"))

    # Count experiments by type
    methods = set()
    datasets = set()
    models = set()
    seeds = set()

    for file in all_files:
        parts = file.stem.split('_')
        if len(parts) >= 3:
            methods.add(parts[0])
            datasets.add(parts[1])

            # Extract seed if present
            for part in parts:
                if part.startswith('seed'):
                    seeds.add(part)

    return {
        'total_experiments': len(all_files),
        'num_methods': len(methods),
        'num_datasets': len(datasets),
        'num_seeds': len(seeds),
        'methods': sorted(list(methods)),
        'datasets': sorted(list(datasets)),
        'seeds': sorted(list(seeds))
    }


def save_results_summary(
    results_dir: Path,
    output_file: Path,
    methods: List[str],
    datasets: List[str],
    models: List[str]
):
    """
    Generate and save comprehensive results summary.

    Args:
        results_dir: Directory containing results
        output_file: Path to save summary
        methods: List of methods
        datasets: List of datasets
        models: List of models
    """
    summary = {
        'experiment_info': generate_summary_stats(results_dir),
        'accuracy_table_markdown': generate_accuracy_table(
            results_dir, methods, datasets, models, output_format='markdown'
        ),
        'accuracy_table_latex': generate_accuracy_table(
            results_dir, methods, datasets, models, output_format='latex'
        ),
        'coherence_table_markdown': generate_coherence_table(
            results_dir, methods, output_format='markdown'
        ),
        'coherence_table_latex': generate_coherence_table(
            results_dir, methods, output_format='latex'
        )
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results summary saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    results_dir = Path("results")

    if results_dir.exists():
        methods = ['greedy', 'beam', 'self_consistency', 'fgbs']
        datasets = ['gsm8k', 'strategyqa']
        models = ['mistralai/Mistral-7B-Instruct-v0.3', 'microsoft/phi-2']

        print("=== ACCURACY TABLE (Table 1) ===")
        print(generate_accuracy_table(results_dir, methods, datasets, models))

        print("\n=== COHERENCE TABLE (Table 2) ===")
        print(generate_coherence_table(results_dir, methods))

        print("\n=== SUMMARY STATISTICS ===")
        stats = generate_summary_stats(results_dir)
        print(json.dumps(stats, indent=2))
    else:
        print(f"Results directory not found: {results_dir}")
