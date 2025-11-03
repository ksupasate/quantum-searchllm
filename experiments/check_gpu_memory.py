#!/usr/bin/env python3
"""
GPU Memory Diagnostic Tool

Checks current GPU memory usage and model loading.
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set memory optimization BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc


def print_gpu_memory(stage=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n{'='*60}")
        print(f"GPU Memory {stage}")
        print(f"{'='*60}")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")
        print(f"Free:      {total - allocated:.2f} GB")
        print(f"Total:     {total:.2f} GB")
        print(f"{'='*60}\n")
    else:
        print("No CUDA GPU available")


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_test(model_name: str, use_8bit: bool = True):
    """Test loading a model and check memory."""
    print(f"\nTesting model: {model_name}")
    print(f"8-bit quantization: {use_8bit}")

    print_gpu_memory("BEFORE model load")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure 8-bit loading
    if use_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

    print_gpu_memory("AFTER model load")

    # Check if model is quantized
    print("\nModel Details:")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")

    # Check first layer dtype
    first_param = next(model.parameters())
    print(f"First parameter dtype: {first_param.dtype}")
    print(f"First parameter device: {first_param.device}")

    # Try a simple generation
    print("\nTrying simple generation...")
    inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)

    print_gpu_memory("BEFORE generation")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)

    print_gpu_memory("AFTER generation")

    generated_text = tokenizer.decode(outputs[0])
    print(f"Generated: {generated_text}")

    # Clean up
    del model, tokenizer, inputs, outputs
    clear_memory()

    print_gpu_memory("AFTER cleanup")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit quantization")
    args = parser.parse_args()

    print("="*60)
    print("GPU MEMORY DIAGNOSTIC")
    print("="*60)

    print(f"\nEnvironment:")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    print_gpu_memory("INITIAL")

    try:
        load_model_test(args.model, use_8bit=not args.no_8bit)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print_gpu_memory("AFTER ERROR")
