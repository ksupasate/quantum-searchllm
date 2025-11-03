#!/usr/bin/env python3
"""
Visualization Tools for TNAD Experimental Results

Generates publication-quality plots for:
- Accuracy comparisons with error bars
- Ablation study results
- CFS trajectory analysis
- Method comparisons across benchmarks

Usage:
    # Generate all plots
    python experiments/visualize_results.py --results_dir ./results

    # Generate specific plots
    python experiments/visualize_results.py --results_dir ./results --plots accuracy ablation

    # Customize output
    python experiments/visualize_results.py \
        --results_dir ./results \
        --output_dir ./plots \
        --dpi 300 \
        --style seaborn-v0_8
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from statistical_analysis import aggregate_results_by_seed


def plot_accuracy_comparison(
    results_dir: Path,
    methods: List[str],
    benchmarks: List[str],
    models: List[str],
    output_dir: Path,
    dpi: int = 300,
):
    """
    Plot accuracy comparison across methods with error bars.

    Args:
        results_dir: Directory containing results
        methods: List of methods
        benchmarks: List of benchmarks
        models: List of models
        output_dir: Output directory for plots
        dpi: DPI for saved figures
    """
    print("Generating accuracy comparison plot...")

    for model in models:
        model_short = model.split('/')[-1] if '/' in model else model

        fig, axes = plt.subplots(1, len(benchmarks), figsize=(6*len(benchmarks), 5))
        if len(benchmarks) == 1:
            axes = [axes]

        for idx, benchmark in enumerate(benchmarks):
            ax = axes[idx]

            # Collect data
            method_names = []
            means = []
            stds = []

            for method in methods:
                stats = aggregate_results_by_seed(
                    results_dir,
                    method,
                    benchmark,
                    model,
                    metric='accuracy'
                )

                if stats['n_seeds'] > 0:
                    method_names.append(method.replace('_', ' ').title())
                    means.append(stats['mean'] * 100)
                    stds.append(stats['std'] * 100)

            if not method_names:
                continue

            # Create bar plot
            x_pos = np.arange(len(method_names))
            colors = sns.color_palette("husl", len(method_names))

            bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                         color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

            # Customize plot
            ax.set_xlabel('Method', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'{benchmark.upper()}', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 100)

            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                       f'{mean:.1f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.suptitle(f'Accuracy Comparison - {model_short}',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save plot
        plot_filename = f"accuracy_comparison_{model_short}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"Saved: {plot_path}")


def plot_method_comparison_heatmap(
    results_dir: Path,
    methods: List[str],
    benchmarks: List[str],
    models: List[str],
    output_dir: Path,
    dpi: int = 300,
):
    """
    Plot heatmap of method performance across benchmarks and models.

    Args:
        results_dir: Directory containing results
        methods: List of methods
        benchmarks: List of benchmarks
        models: List of models
        output_dir: Output directory
        dpi: DPI for saved figures
    """
    print("Generating method comparison heatmap...")

    # Collect data
    data = []
    row_labels = []

    for model in models:
        model_short = model.split('/')[-1] if '/' in model else model

        for benchmark in benchmarks:
            row_label = f"{model_short}\n{benchmark.upper()}"
            row_labels.append(row_label)

            row_data = []
            for method in methods:
                stats = aggregate_results_by_seed(
                    results_dir,
                    method,
                    benchmark,
                    model,
                    metric='accuracy'
                )
                row_data.append(stats['mean'] * 100 if stats['n_seeds'] > 0 else 0)

            data.append(row_data)

    # Create heatmap
    data_array = np.array(data)

    fig, ax = plt.subplots(figsize=(10, max(6, len(row_labels) * 0.5)))

    im = ax.imshow(data_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Set ticks
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
    ax.set_yticklabels(row_labels)

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(methods)):
            value = data_array[i, j]
            text = ax.text(j, i, f'{value:.1f}',
                          ha="center", va="center",
                          color="black" if value > 50 else "white",
                          fontsize=10, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

    ax.set_title('Method Performance Heatmap', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "method_comparison_heatmap.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Saved: {plot_path}")


def plot_cfs_trajectories(
    results_dir: Path,
    methods: List[str],
    output_dir: Path,
    dpi: int = 300,
    max_examples: int = 5,
):
    """
    Plot CFS (Coherence Fidelity Score) trajectories.

    Args:
        results_dir: Directory containing results
        methods: List of methods
        output_dir: Output directory
        dpi: DPI for saved figures
        max_examples: Maximum number of examples to plot
    """
    print("Generating CFS trajectory plots...")

    # Look for result files with CFS data
    result_files = list(results_dir.glob("fgbs_*_seed*.json"))

    if not result_files:
        print("No FGBS results with CFS data found.")
        return

    # Sample some files
    import random
    sample_files = random.sample(result_files, min(max_examples, len(result_files)))

    fig, axes = plt.subplots(len(sample_files), 1, figsize=(12, 4*len(sample_files)))
    if len(sample_files) == 1:
        axes = [axes]

    for idx, result_file in enumerate(sample_files):
        ax = axes[idx]

        with open(result_file, 'r') as f:
            data = json.load(f)

        # Check if CFS trajectory exists
        if 'cfs_trajectory' in data:
            cfs_values = data['cfs_trajectory']
            steps = range(len(cfs_values))

            ax.plot(steps, cfs_values, marker='o', linewidth=2, markersize=4)
            ax.set_xlabel('Generation Step', fontsize=11, fontweight='bold')
            ax.set_ylabel('Coherence Fidelity Score', fontsize=11, fontweight='bold')
            ax.set_title(f'CFS Trajectory - {result_file.stem}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No CFS data available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "cfs_trajectories.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Saved: {plot_path}")


def plot_improvement_over_baseline(
    results_dir: Path,
    baseline_method: str,
    comparison_methods: List[str],
    benchmarks: List[str],
    models: List[str],
    output_dir: Path,
    dpi: int = 300,
):
    """
    Plot improvement over baseline method.

    Args:
        results_dir: Directory containing results
        baseline_method: Baseline method name
        comparison_methods: Methods to compare
        benchmarks: List of benchmarks
        models: List of models
        output_dir: Output directory
        dpi: DPI for saved figures
    """
    print(f"Generating improvement over {baseline_method} plot...")

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(6*len(benchmarks), 5))
    if len(benchmarks) == 1:
        axes = [axes]

    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx]

        # Collect improvements across models
        method_names = []
        improvements = []

        for method in comparison_methods:
            if method == baseline_method:
                continue

            method_improvements = []

            for model in models:
                # Get baseline
                baseline_stats = aggregate_results_by_seed(
                    results_dir, baseline_method, benchmark, model, 'accuracy'
                )

                # Get method
                method_stats = aggregate_results_by_seed(
                    results_dir, method, benchmark, model, 'accuracy'
                )

                if baseline_stats['n_seeds'] > 0 and method_stats['n_seeds'] > 0:
                    improvement = (method_stats['mean'] - baseline_stats['mean']) * 100
                    method_improvements.append(improvement)

            if method_improvements:
                method_names.append(method.replace('_', ' ').title())
                improvements.append(np.mean(method_improvements))

        if not method_names:
            continue

        # Create bar plot
        x_pos = np.arange(len(method_names))
        colors = ['green' if imp > 0 else 'red' for imp in improvements]

        bars = ax.bar(x_pos, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Customize
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Improvement over {baseline_method.title()} (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{benchmark.upper()}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.,
                   height + (0.5 if height > 0 else -0.5),
                   f'{imp:+.1f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=9, fontweight='bold')

    plt.suptitle(f'Improvement over {baseline_method.title()} Baseline',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"improvement_over_{baseline_method}.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Saved: {plot_path}")


def main():
    """Main visualization script."""
    parser = argparse.ArgumentParser(
        description="Generate visualization plots for TNAD results"
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Directory containing result JSON files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for plots (defaults to results_dir/plots)'
    )
    parser.add_argument(
        '--plots',
        type=str,
        nargs='+',
        default=['all'],
        choices=['accuracy', 'heatmap', 'cfs', 'improvement', 'all'],
        help='Which plots to generate'
    )
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['greedy', 'beam_search', 'self_consistency', 'fgbs'],
        help='Methods to include'
    )
    parser.add_argument(
        '--benchmarks',
        type=str,
        nargs='+',
        default=['gsm8k', 'strategyqa'],
        help='Benchmarks to include'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['mistralai/Mistral-7B-Instruct-v0.3'],
        help='Models to include'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='greedy',
        help='Baseline method for improvement plots'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved figures'
    )
    parser.add_argument(
        '--style',
        type=str,
        default='seaborn-v0_8',
        help='Matplotlib style'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    try:
        plt.style.use(args.style)
    except:
        print(f"Warning: Style '{args.style}' not found, using default")

    sns.set_palette("husl")

    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATION PLOTS")
    print(f"{'='*80}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Plots to generate: {args.plots}")
    print(f"{'='*80}\n")

    # Determine which plots to generate
    plots_to_generate = args.plots if 'all' not in args.plots else ['accuracy', 'heatmap', 'cfs', 'improvement']

    # Generate plots
    if 'accuracy' in plots_to_generate:
        plot_accuracy_comparison(
            results_dir,
            args.methods,
            args.benchmarks,
            args.models,
            output_dir,
            args.dpi
        )

    if 'heatmap' in plots_to_generate:
        plot_method_comparison_heatmap(
            results_dir,
            args.methods,
            args.benchmarks,
            args.models,
            output_dir,
            args.dpi
        )

    if 'cfs' in plots_to_generate:
        plot_cfs_trajectories(
            results_dir,
            args.methods,
            output_dir,
            args.dpi
        )

    if 'improvement' in plots_to_generate:
        comparison_methods = [m for m in args.methods if m != args.baseline]
        plot_improvement_over_baseline(
            results_dir,
            args.baseline,
            comparison_methods,
            args.benchmarks,
            args.models,
            output_dir,
            args.dpi
        )

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"Plots saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
