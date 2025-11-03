#!/usr/bin/env python3
"""
Results Aggregation Script for TNAD Experiments

Aggregates multi-seed experimental results and generates publication-ready tables.

Usage:
    # Aggregate all results in results directory
    python experiments/aggregate_results.py --results_dir ./results

    # Generate specific output format
    python experiments/aggregate_results.py --results_dir ./results --format latex

    # Specify methods and benchmarks
    python experiments/aggregate_results.py \
        --results_dir ./results \
        --methods greedy beam_search self_consistency fgbs \
        --benchmarks gsm8k strategyqa
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from statistical_analysis import (
    generate_accuracy_table,
    generate_coherence_table,
    save_results_summary,
    generate_summary_stats,
    aggregate_results_by_seed,
)


def main():
    """Main aggregation script."""
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed experiment results"
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Directory containing result JSON files'
    )
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['greedy', 'beam_search', 'self_consistency', 'fgbs'],
        help='Methods to include in tables'
    )
    parser.add_argument(
        '--benchmarks',
        type=str,
        nargs='+',
        default=['gsm8k', 'strategyqa'],
        help='Benchmarks to include in tables'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Models to include (auto-detect if not specified)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='all',
        choices=['markdown', 'latex', 'csv', 'all'],
        help='Output format'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (defaults to results_dir)'
    )
    parser.add_argument(
        '--baseline_method',
        type=str,
        default='greedy',
        help='Baseline method for significance testing'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Auto-detect models if not specified
    if args.models is None:
        # Scan result files to find models
        result_files = list(results_dir.glob("*_seed*.json"))
        models_found = set()

        for file in result_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if 'model' in data:
                        models_found.add(data['model'])
            except:
                continue

        if not models_found:
            # Fallback: try to infer from filenames
            for file in result_files:
                parts = file.stem.split('_')
                # Look for model name patterns in filename
                for i, part in enumerate(parts):
                    if any(model_hint in part.lower() for model_hint in ['mistral', 'phi', 'llama', 'gemma']):
                        # Try to reconstruct model name
                        if i + 1 < len(parts) and 'seed' not in parts[i+1]:
                            model_name = f"{part}/{parts[i+1]}"
                            models_found.add(model_name)

        args.models = list(models_found) if models_found else ['unknown']
        print(f"Auto-detected models: {args.models}")

    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS")
    print(f"{'='*80}")
    print(f"Results directory: {results_dir}")
    print(f"Methods: {args.methods}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Models: {args.models}")
    print(f"Output format: {args.format}")
    print(f"{'='*80}\n")

    # Generate summary statistics
    print("Generating summary statistics...")
    summary_stats = generate_summary_stats(results_dir)
    print(f"\nExperiment Summary:")
    print(f"  Total experiments: {summary_stats['total_experiments']}")
    print(f"  Methods: {summary_stats['num_methods']}")
    print(f"  Datasets: {summary_stats['num_datasets']}")
    print(f"  Seeds: {summary_stats['num_seeds']}")

    # Generate tables based on format
    formats_to_generate = ['markdown', 'latex', 'csv'] if args.format == 'all' else [args.format]

    # Table 1: Accuracy
    print("\n" + "="*80)
    print("GENERATING TABLE 1: ACCURACY RESULTS")
    print("="*80)

    for fmt in formats_to_generate:
        print(f"\nGenerating {fmt.upper()} format...")
        table_1 = generate_accuracy_table(
            results_dir,
            args.methods,
            args.benchmarks,
            args.models,
            baseline_method=args.baseline_method,
            output_format=fmt
        )

        # Save to file
        ext = 'tex' if fmt == 'latex' else fmt
        output_file = output_dir / f'table_1_accuracy.{ext}'
        with open(output_file, 'w') as f:
            f.write(table_1)

        print(f"Saved: {output_file}")

        # Print markdown version to console
        if fmt == 'markdown':
            print("\n" + table_1)

    # Table 2: Coherence
    print("\n" + "="*80)
    print("GENERATING TABLE 2: COHERENCE METRICS")
    print("="*80)

    # Check if coherence results exist
    coherence_files = list(results_dir.glob("coherence_*.json"))

    if coherence_files:
        for fmt in formats_to_generate:
            print(f"\nGenerating {fmt.upper()} format...")
            table_2 = generate_coherence_table(
                results_dir,
                args.methods,
                output_format=fmt
            )

            # Save to file
            ext = 'tex' if fmt == 'latex' else fmt
            output_file = output_dir / f'table_2_coherence.{ext}'
            with open(output_file, 'w') as f:
                f.write(table_2)

            print(f"Saved: {output_file}")

            # Print markdown version to console
            if fmt == 'markdown':
                print("\n" + table_2)
    else:
        print("\nNo coherence results found. Skipping Table 2.")

    # Save comprehensive summary
    print("\n" + "="*80)
    print("SAVING COMPREHENSIVE SUMMARY")
    print("="*80)

    save_results_summary(
        results_dir,
        output_dir / 'results_summary.json',
        args.methods,
        args.benchmarks,
        args.models
    )

    # Print detailed statistics for each method
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)

    for model in args.models:
        model_short = model.split('/')[-1] if '/' in model else model
        print(f"\nModel: {model_short}")
        print("-" * 60)

        for benchmark in args.benchmarks:
            print(f"\n  {benchmark.upper()}:")

            for method in args.methods:
                stats = aggregate_results_by_seed(
                    results_dir,
                    method,
                    benchmark,
                    model,
                    metric='accuracy'
                )

                if stats['n_seeds'] > 0:
                    mean_pct = stats['mean'] * 100
                    std_pct = stats['std'] * 100
                    print(f"    {method:20s}: {mean_pct:5.1f}% ± {std_pct:4.1f}%  (n={stats['n_seeds']})")
                else:
                    print(f"    {method:20s}: No results found")

    # Final summary
    print("\n" + "="*80)
    print("AGGREGATION COMPLETE!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  - Accuracy tables: table_1_accuracy.[txt|tex|csv]")
    if coherence_files:
        print(f"  - Coherence tables: table_2_coherence.[txt|tex|csv]")
    print(f"  - Summary: results_summary.json")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
