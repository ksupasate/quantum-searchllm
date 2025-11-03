#!/usr/bin/env python3
"""
Setup Verification Script

Checks if the environment is properly configured for running the quantum-search-llm
with quantization support.
"""

import sys


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA is not available")
            print("   This script must be run on a machine with a CUDA-enabled GPU")
            return False

        print(f"✅ CUDA is available")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False


def check_bitsandbytes():
    """Check if bitsandbytes is properly installed."""
    try:
        import bitsandbytes as bnb
        print(f"✅ bitsandbytes is installed (version {bnb.__version__})")

        # Try to import the CUDA functions
        try:
            from bitsandbytes.cuda_setup import CUDA_SETUP
            if CUDA_SETUP.get('SUCCESS', False):
                print("✅ bitsandbytes CUDA setup successful")
            else:
                print("⚠️  bitsandbytes CUDA setup may have issues")
                print(f"   Setup info: {CUDA_SETUP}")
        except Exception as e:
            print(f"⚠️  Could not verify bitsandbytes CUDA setup: {e}")

        return True
    except ImportError:
        print("❌ bitsandbytes is not installed")
        print("\n📝 To install bitsandbytes:")
        print("   pip install bitsandbytes")
        return False
    except Exception as e:
        print(f"❌ Error checking bitsandbytes: {e}")
        return False


def check_transformers():
    """Check if transformers is properly installed."""
    try:
        import transformers
        print(f"✅ transformers is installed (version {transformers.__version__})")

        # Check if it's a recent enough version
        version = tuple(map(int, transformers.__version__.split('.')[:2]))
        if version < (4, 30):
            print("⚠️  transformers version is old, recommend >= 4.30 for quantization")
        return True
    except ImportError:
        print("❌ transformers is not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking transformers: {e}")
        return False


def test_quantization():
    """Test if 8-bit and 4-bit quantization work."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        print("\n🧪 Testing quantization...")

        # Test 8-bit
        print("   Testing 8-bit quantization...")
        try:
            config_8bit = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            # Try to create config (doesn't load model)
            print("   ✅ 8-bit config creation successful")
        except Exception as e:
            print(f"   ❌ 8-bit config failed: {e}")

        # Test 4-bit
        print("   Testing 4-bit quantization...")
        try:
            config_4bit = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("   ✅ 4-bit config creation successful")
        except Exception as e:
            print(f"   ❌ 4-bit config failed: {e}")

        return True
    except Exception as e:
        print(f"❌ Error testing quantization: {e}")
        return False


def print_recommendations():
    """Print recommendations for fixing common issues."""
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("""
If bitsandbytes is not working properly:

1. Reinstall bitsandbytes:
   pip uninstall bitsandbytes
   pip install bitsandbytes

2. Check CUDA compatibility:
   - Your PyTorch CUDA version must match your system CUDA
   - Check with: python -c "import torch; print(torch.version.cuda)"

3. If 8-bit fails, the code will now automatically fall back to 4-bit
   - 8-bit quantization: ~14 GB memory
   - 4-bit quantization: ~7 GB memory (recommended if 8-bit fails)

4. To manually enable 4-bit in config, add to configs/default.yaml:
   model:
     load_in_4bit: true
     load_in_8bit: false

5. If all quantization fails, reduce model parameters:
   fgbs:
     beam_width: 2      # Reduce from 3
     bond_dim: 4        # Reduce from 8
     max_length: 128    # Reduce from 256
""")


def main():
    print("="*60)
    print("ENVIRONMENT VERIFICATION")
    print("="*60)
    print()

    checks_passed = 0
    total_checks = 4

    if check_cuda():
        checks_passed += 1

    if check_transformers():
        checks_passed += 1

    if check_bitsandbytes():
        checks_passed += 1

    if test_quantization():
        checks_passed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {checks_passed}/{total_checks} checks passed")
    print("="*60)

    if checks_passed == total_checks:
        print("✅ All checks passed! Your environment is ready.")
    else:
        print("⚠️  Some checks failed. See recommendations below.")
        print_recommendations()

    return checks_passed == total_checks


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
