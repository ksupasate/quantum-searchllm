#!/usr/bin/env python3
"""
Quick diagnostic script to check if 8-bit quantization is working.
This will help identify why the model is using 78 GB instead of ~14 GB.
"""

import os
# Set memory allocator BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("="*80)
print("DIAGNOSTIC: Model Loading Test")
print("="*80)

# Check GPU
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print()

# Check bitsandbytes
try:
    import bitsandbytes as bnb
    print(f"bitsandbytes version: {bnb.__version__}")
    print(f"bitsandbytes CUDA available: {hasattr(bnb, 'functional')}")
except ImportError as e:
    print(f"ERROR: bitsandbytes not installed or not working: {e}")
    print("\nPlease install: pip install bitsandbytes")
    exit(1)

print()
print("-"*80)
print("Test 1: Loading WITHOUT 8-bit quantization")
print("-"*80)

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Clear any existing memory
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

mem_before = torch.cuda.memory_allocated() / 1e9
print(f"Memory before load: {mem_before:.2f} GB")

try:
    # Load in fp16 without quantization
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    mem_after = torch.cuda.memory_allocated() / 1e9
    mem_peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"Memory after load: {mem_after:.2f} GB")
    print(f"Peak memory: {mem_peak:.2f} GB")
    print(f"Model footprint: {mem_after - mem_before:.2f} GB")

    # Check dtypes
    dtypes = {}
    for name, param in model_fp16.named_parameters():
        dtype = str(param.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + 1

    print(f"Parameter dtypes: {dtypes}")

    # Clean up
    del model_fp16
    torch.cuda.empty_cache()

except Exception as e:
    print(f"ERROR loading FP16 model: {e}")

print()
print("-"*80)
print("Test 2: Loading WITH 8-bit quantization")
print("-"*80)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

mem_before = torch.cuda.memory_allocated() / 1e9
print(f"Memory before load: {mem_before:.2f} GB")

try:
    # Configure 8-bit quantization
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    print("Loading with 8-bit quantization config...")
    model_8bit = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )

    mem_after = torch.cuda.memory_allocated() / 1e9
    mem_peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"Memory after load: {mem_after:.2f} GB")
    print(f"Peak memory: {mem_peak:.2f} GB")
    print(f"Model footprint: {mem_after - mem_before:.2f} GB")

    # Check dtypes
    dtypes = {}
    for name, param in model_8bit.named_parameters():
        dtype = str(param.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + 1

    print(f"Parameter dtypes: {dtypes}")

    # Expected: should be ~14 GB for 7B model with 8-bit
    expected_gb = 14.0
    actual_gb = mem_after - mem_before

    print()
    if actual_gb > 25:
        print(f"❌ FAILED: Model using {actual_gb:.2f} GB (expected ~{expected_gb:.2f} GB)")
        print("   8-bit quantization is NOT working!")
        print()
        print("Possible causes:")
        print("1. bitsandbytes not compiled for your CUDA version")
        print("2. Incompatible transformers/bitsandbytes versions")
        print("3. Missing CUDA libraries")
        print()
        print("Try:")
        print("  pip uninstall bitsandbytes")
        print("  pip install bitsandbytes --no-cache-dir")
    else:
        print(f"✅ SUCCESS: Model using {actual_gb:.2f} GB (expected ~{expected_gb:.2f} GB)")
        print("   8-bit quantization is working correctly!")

    del model_8bit
    torch.cuda.empty_cache()

except Exception as e:
    print(f"ERROR loading 8-bit model: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
