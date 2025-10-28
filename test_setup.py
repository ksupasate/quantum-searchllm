#!/usr/bin/env python3
"""
Quick test script to verify setup is working correctly.

Usage:
    python3 test_setup.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing TNAD Setup...")
print("=" * 60)

# Test 1: Import core modules
print("\n1. Testing core module imports...")
try:
    from tnad import FidelityGuidedBeamSearcher, MPSSequence, compute_cfs
    print("   ✓ Core modules imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import core modules: {e}")
    sys.exit(1)

# Test 2: Import utilities
print("\n2. Testing utility imports...")
try:
    from tnad.utils import get_device, setup_logger, log_normalize
    print("   ✓ Utility modules imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import utilities: {e}")
    sys.exit(1)

# Test 3: Test MPS creation
print("\n3. Testing MPS creation...")
try:
    import torch
    mps = MPSSequence(bond_dim=8, embedding_dim=64)
    token_embedding = torch.randn(64)
    mps.add_token(token_embedding)
    schmidt_values = mps.get_schmidt_values()
    print(f"   ✓ MPS created successfully (length={mps.get_current_length()})")
except Exception as e:
    print(f"   ✗ Failed to create MPS: {e}")
    sys.exit(1)

# Test 4: Test CFS computation
print("\n4. Testing CFS computation...")
try:
    cfs = compute_cfs(schmidt_values)
    print(f"   ✓ CFS computed successfully (CFS={cfs:.2f})")
except Exception as e:
    print(f"   ✗ Failed to compute CFS: {e}")
    sys.exit(1)

# Test 5: Check PyTorch and device
print("\n5. Checking PyTorch and device...")
try:
    import torch
    device = get_device()
    print(f"   ✓ PyTorch available (version: {torch.__version__})")
    print(f"   ✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available (GPU: {torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        print(f"   ✓ MPS (Apple Silicon) available")
    else:
        print(f"   ⚠ Running on CPU (experiments will be slower)")
except Exception as e:
    print(f"   ✗ PyTorch issue: {e}")

# Test 6: Check transformers
print("\n6. Checking transformers library...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"   ✓ Transformers library available")
except ImportError:
    print(f"   ✗ Transformers not installed")
    print(f"   Install with: pip install transformers")

# Test 7: Check other dependencies
print("\n7. Checking other dependencies...")
deps_status = []

try:
    import numpy
    deps_status.append(("numpy", numpy.__version__, True))
except ImportError:
    deps_status.append(("numpy", None, False))

try:
    import scipy
    deps_status.append(("scipy", scipy.__version__, True))
except ImportError:
    deps_status.append(("scipy", None, False))

try:
    import yaml
    deps_status.append(("pyyaml", "OK", True))
except ImportError:
    deps_status.append(("pyyaml", None, False))

try:
    import tqdm
    deps_status.append(("tqdm", tqdm.__version__, True))
except ImportError:
    deps_status.append(("tqdm", None, False))

try:
    from datasets import load_dataset
    deps_status.append(("datasets", "OK", True))
except ImportError:
    deps_status.append(("datasets", None, False))

for name, version, status in deps_status:
    if status:
        print(f"   ✓ {name:15} {version}")
    else:
        print(f"   ✗ {name:15} NOT INSTALLED")

# Summary
print("\n" + "=" * 60)
print("SETUP TEST SUMMARY")
print("=" * 60)

all_core_ok = True
missing_deps = [name for name, _, status in deps_status if not status]

if all_core_ok and not missing_deps:
    print("✓ All tests passed! Setup is complete.")
    print("\nYou can now run experiments:")
    print("  python3 experiments/reproduce_paper_results.py --quick_test")
elif missing_deps:
    print("⚠ Setup mostly complete, but some optional dependencies missing:")
    for dep in missing_deps:
        print(f"  - {dep}")
    print("\nInstall missing dependencies:")
    print(f"  pip3 install {' '.join(missing_deps)}")
    print("\nYou can still run experiments if transformers is installed.")
else:
    print("✗ Setup has issues. Please check error messages above.")
    sys.exit(1)

print("=" * 60)
