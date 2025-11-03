#!/usr/bin/env python3
"""
Clear all HuggingFace caches and large model files from storage.

This script removes:
- HuggingFace model cache (~/.cache/huggingface/hub)
- HuggingFace datasets cache (~/.cache/huggingface/datasets)
- HuggingFace XET cache (~/.cache/huggingface/xet)
- Other HuggingFace cache files

CAUTION: This will delete all downloaded models and datasets.
They will need to be re-downloaded when used again.
"""

import os
import shutil
import sys
from pathlib import Path


def get_dir_size(path: Path) -> int:
    """Calculate total size of a directory in bytes."""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except (PermissionError, OSError):
        pass
    return total


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def clear_cache(cache_dir: Path, interactive: bool = True) -> None:
    """Clear HuggingFace cache directory."""

    if not cache_dir.exists():
        print(f"✓ Cache directory not found: {cache_dir}")
        return

    # Calculate size before deletion
    print(f"\nAnalyzing: {cache_dir}")
    total_size = get_dir_size(cache_dir)
    print(f"  Current size: {format_size(total_size)}")

    # List subdirectories
    subdirs = [d for d in cache_dir.iterdir() if d.is_dir()]
    if subdirs:
        print(f"  Subdirectories:")
        for subdir in subdirs:
            subdir_size = get_dir_size(subdir)
            print(f"    - {subdir.name}: {format_size(subdir_size)}")

    # Confirm deletion
    if interactive:
        response = input(f"\n  Delete this cache? [y/N]: ").strip().lower()
        if response != 'y':
            print("  Skipped.")
            return

    # Delete cache
    try:
        shutil.rmtree(cache_dir)
        print(f"  ✓ Deleted {format_size(total_size)}")
    except Exception as e:
        print(f"  ✗ Error deleting cache: {e}")


def main():
    """Main function to clear all HuggingFace caches."""

    print("=" * 70)
    print("HuggingFace Cache Cleaner")
    print("=" * 70)

    # Determine if running in interactive mode
    interactive = sys.stdin.isatty() and '--yes' not in sys.argv

    # HuggingFace cache directory
    hf_cache_home = Path.home() / '.cache' / 'huggingface'

    if not hf_cache_home.exists():
        print("\n✓ No HuggingFace cache found. Nothing to clear.")
        return

    # Show total cache size
    total_size = get_dir_size(hf_cache_home)
    print(f"\nTotal HuggingFace cache size: {format_size(total_size)}")
    print(f"Location: {hf_cache_home}")

    # Caches to clear
    caches_to_clear = [
        hf_cache_home / 'hub',        # Model cache
        hf_cache_home / 'datasets',   # Dataset cache
        hf_cache_home / 'xet',        # XET cache
        hf_cache_home / 'modules',    # Downloaded modules
    ]

    # Clear each cache
    for cache_path in caches_to_clear:
        if cache_path.exists():
            clear_cache(cache_path, interactive=interactive)

    # Clear token files if requested
    token_files = [
        hf_cache_home / 'token',
        hf_cache_home / 'stored_tokens',
    ]

    if interactive:
        print("\n" + "=" * 70)
        response = input("Also clear authentication tokens? [y/N]: ").strip().lower()
        if response == 'y':
            for token_file in token_files:
                if token_file.exists():
                    try:
                        token_file.unlink()
                        print(f"  ✓ Deleted {token_file.name}")
                    except Exception as e:
                        print(f"  ✗ Error deleting {token_file.name}: {e}")

    # Final summary
    print("\n" + "=" * 70)
    print("Cache clearing complete!")
    print("=" * 70)

    remaining_size = get_dir_size(hf_cache_home) if hf_cache_home.exists() else 0
    freed_space = total_size - remaining_size

    print(f"\nFreed space: {format_size(freed_space)}")
    print(f"Remaining cache: {format_size(remaining_size)}")

    print("\nNote: Models and datasets will be re-downloaded when needed.")


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        print("\nUsage:")
        print("  python clear_cache.py          # Interactive mode")
        print("  python clear_cache.py --yes    # Non-interactive mode (delete all)")
        sys.exit(0)

    main()
