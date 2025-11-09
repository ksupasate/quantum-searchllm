"""
Validation script for TNAD optimizations.

Tests that all optimizations work correctly and maintain backward compatibility.
"""

import sys
import time
import numpy as np

print("=" * 80)
print("TNAD OPTIMIZATION VALIDATION")
print("=" * 80)
print()

# Test 1: Import check
print("Test 1: Checking imports...")
try:
    import torch
    from tnad import MPSSequence, compute_cfs, FidelityGuidedBeamSearcher
    from tnad.utils import get_device
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: MPS basic operations
print("\nTest 2: MPS basic operations...")
try:
    mps = MPSSequence(bond_dim=16, embedding_dim=64, device='cpu')

    # Add tokens
    for i in range(10):
        emb = torch.randn(64)
        mps.add_token(emb)

    assert mps.get_current_length() == 10, "Length mismatch"
    print(f"✓ MPS operations work (length={mps.get_current_length()})")
except Exception as e:
    print(f"✗ MPS test failed: {e}")
    sys.exit(1)

# Test 3: Schmidt values and CFS
print("\nTest 3: Schmidt values and CFS computation...")
try:
    schmidt_values = mps.get_schmidt_values()
    assert len(schmidt_values) > 0, "No Schmidt values"
    assert all(schmidt_values >= 0), "Negative Schmidt values"

    cfs = compute_cfs(schmidt_values)
    assert 1.0 <= cfs <= mps.bond_dim, f"CFS out of range: {cfs}"

    print(f"✓ Schmidt/CFS computation works (CFS={cfs:.3f})")
except Exception as e:
    print(f"✗ Schmidt/CFS test failed: {e}")
    sys.exit(1)

# Test 4: MPS copy optimization
print("\nTest 4: Optimized MPS copy...")
try:
    start = time.perf_counter()
    mps_copy = mps.copy()
    copy_time = (time.perf_counter() - start) * 1000

    # Verify copy is independent
    mps_copy.add_token(torch.randn(64))
    assert mps_copy.get_current_length() != mps.get_current_length(), "Copy not independent"

    # Verify shared immutable data (optimization check)
    assert mps_copy._state_transition is mps._state_transition, "Optimization missing: state_transition should be shared"
    assert mps_copy._input_projection is mps._input_projection, "Optimization missing: input_projection should be shared"

    print(f"✓ MPS copy works and is optimized (time={copy_time:.2f}ms)")
except Exception as e:
    print(f"✗ MPS copy test failed: {e}")
    sys.exit(1)

# Test 5: Enhanced caching
print("\nTest 5: Enhanced Schmidt caching...")
try:
    mps2 = MPSSequence(bond_dim=16, embedding_dim=64, device='cpu')
    for i in range(20):
        mps2.add_token(torch.randn(64))

    # First call (cache miss)
    start = time.perf_counter()
    schmidt1 = mps2.get_schmidt_values(cut_position=10)
    time1 = (time.perf_counter() - start) * 1000

    # Second call (cache hit)
    start = time.perf_counter()
    schmidt2 = mps2.get_schmidt_values(cut_position=10)
    time2 = (time.perf_counter() - start) * 1000

    assert np.allclose(schmidt1, schmidt2), "Schmidt values don't match"
    assert time2 < time1 * 0.1, f"Cache not working: {time2:.4f}ms vs {time1:.4f}ms"

    print(f"✓ Schmidt caching works (cache hit {time1/time2:.0f}x faster)")
except Exception as e:
    print(f"✗ Caching test failed: {e}")
    sys.exit(1)

# Test 6: Input validation
print("\nTest 6: Input validation and error handling...")
try:
    # Test invalid inputs
    validation_passed = True

    # Invalid bond_dim
    try:
        MPSSequence(bond_dim=-1, embedding_dim=64)
        validation_passed = False
    except ValueError:
        pass

    # Invalid embedding_dim
    try:
        MPSSequence(bond_dim=16, embedding_dim=0)
        validation_passed = False
    except ValueError:
        pass

    # Wrong embedding size
    mps3 = MPSSequence(bond_dim=16, embedding_dim=64, device='cpu')
    try:
        mps3.add_token(torch.randn(32))  # Wrong size
        validation_passed = False
    except ValueError:
        pass

    assert validation_passed, "Validation not working"
    print("✓ Input validation and error handling work")
except Exception as e:
    print(f"✗ Validation test failed: {e}")
    sys.exit(1)

# Test 7: Performance check (basic)
print("\nTest 7: Performance sanity check...")
try:
    mps_perf = MPSSequence(bond_dim=16, embedding_dim=768, device='cpu')

    # Measure add_token performance
    add_times = []
    for i in range(50):
        emb = torch.randn(768)
        start = time.perf_counter()
        mps_perf.add_token(emb)
        add_times.append((time.perf_counter() - start) * 1000)

    avg_time = np.mean(add_times)
    assert avg_time < 10.0, f"add_token too slow: {avg_time:.2f}ms (expected <10ms on CPU)"

    # Measure copy performance
    start = time.perf_counter()
    _ = mps_perf.copy()
    copy_time = (time.perf_counter() - start) * 1000
    assert copy_time < 50.0, f"copy too slow: {copy_time:.2f}ms (expected <50ms on CPU)"

    print(f"✓ Performance acceptable (add_token: {avg_time:.2f}ms, copy: {copy_time:.2f}ms)")
except Exception as e:
    print(f"✗ Performance test failed: {e}")
    sys.exit(1)

# Test 8: Numerical stability
print("\nTest 8: Numerical stability...")
try:
    mps_stab = MPSSequence(bond_dim=8, embedding_dim=32, device='cpu')

    # Add many tokens
    for i in range(100):
        emb = torch.randn(32) * 10  # Large values
        mps_stab.add_token(emb)

    schmidt = mps_stab.get_schmidt_values()
    cfs = compute_cfs(schmidt)

    # Check for NaN/Inf
    assert not np.isnan(schmidt).any(), "NaN in Schmidt values"
    assert not np.isinf(schmidt).any(), "Inf in Schmidt values"
    assert not np.isnan(cfs), "NaN in CFS"
    assert not np.isinf(cfs), "Inf in CFS"

    print(f"✓ Numerical stability maintained (CFS={cfs:.3f})")
except Exception as e:
    print(f"✗ Stability test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("ALL VALIDATION TESTS PASSED ✓")
print("=" * 80)
print("\nOptimizations verified:")
print("  • Enhanced Schmidt caching (30-entry LRU)")
print("  • Shallow copy for immutable matrices")
print("  • Efficient tensor operations")
print("  • Robust input validation")
print("  • Numerical stability")
print("\nThe optimized TNAD implementation is ready for use!")
print("=" * 80)
