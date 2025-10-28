"""
Unit Tests for Coherence Fidelity Score (CFS)

Tests the CFS computation including:
- Basic CFS calculation from Schmidt values
- Edge cases (uniform, peaked, degenerate distributions)
- Numerical stability
- Batch processing
- Integration with MPSSequence
"""

import pytest
import torch
import numpy as np

from tnad.coherence_score import (
    compute_cfs,
    compute_cfs_from_mps,
    compute_cfs_batch,
    analyze_coherence_spectrum,
)
from tnad.mps_manager import MPSSequence


class TestComputeCFS:
    """Test suite for compute_cfs function."""

    def test_uniform_distribution(self):
        """Test CFS for uniform Schmidt distribution (maximum coherence)."""
        # Uniform distribution: all λᵢ = 1/√χ
        chi = 16
        schmidt_values = np.ones(chi) / np.sqrt(chi)

        cfs = compute_cfs(schmidt_values, normalize=False)

        # For uniform: P = χ * (1/χ²) = 1/χ, so F = χ
        expected_cfs = chi
        assert np.isclose(cfs, expected_cfs, rtol=0.1)

    def test_peaked_distribution(self):
        """Test CFS for peaked Schmidt distribution (minimum coherence)."""
        # Peaked distribution: λ₁ = 1, others = 0
        schmidt_values = np.array([1.0, 0.0, 0.0, 0.0])

        cfs = compute_cfs(schmidt_values, normalize=False)

        # For peaked: P = 1⁴ = 1, so F = 1
        expected_cfs = 1.0
        assert np.isclose(cfs, expected_cfs, atol=0.1)

    def test_two_level_system(self):
        """Test CFS for simple 2-level system."""
        # Bell state-like: λ = [1/√2, 1/√2]
        schmidt_values = np.array([1.0, 1.0]) / np.sqrt(2)

        cfs = compute_cfs(schmidt_values, normalize=False)

        # P = 2 * (1/2)² = 1/2, so F = 2
        expected_cfs = 2.0
        assert np.isclose(cfs, expected_cfs, atol=0.1)

    def test_normalization(self):
        """Test that normalization works correctly."""
        # Unnormalized values
        unnormalized = np.array([3.0, 4.0])  # ||v|| = 5

        # Compute CFS with normalization
        cfs_normalized = compute_cfs(unnormalized, normalize=True)

        # Manually normalize and compute
        normalized = unnormalized / np.linalg.norm(unnormalized)
        cfs_manual = compute_cfs(normalized, normalize=False)

        assert np.isclose(cfs_normalized, cfs_manual)

    def test_torch_tensor_input(self):
        """Test that PyTorch tensors are accepted."""
        schmidt_torch = torch.tensor([0.8, 0.6])  # Will be normalized

        cfs = compute_cfs(schmidt_torch, normalize=True)

        # Should work without errors
        assert isinstance(cfs, float)
        assert cfs >= 1.0

    def test_list_input(self):
        """Test that Python lists are accepted."""
        schmidt_list = [0.7, 0.5, 0.3, 0.1]

        cfs = compute_cfs(schmidt_list, normalize=True)

        assert isinstance(cfs, float)
        assert cfs >= 1.0

    def test_empty_input(self):
        """Test error handling for empty input."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_cfs([])

    def test_return_purity(self):
        """Test returning both CFS and purity."""
        schmidt_values = np.array([0.8, 0.6])
        schmidt_values = schmidt_values / np.linalg.norm(schmidt_values)

        cfs, purity = compute_cfs(schmidt_values, return_purity=True)

        # Check purity is in valid range [1/χ, 1]
        chi = len(schmidt_values)
        assert 1.0 / chi <= purity <= 1.0

        # Check CFS = 1/purity
        assert np.isclose(cfs, 1.0 / purity, atol=1e-6)

    def test_cfs_bounds(self):
        """Test that CFS is always within [1, χ]."""
        for chi in [4, 8, 16, 32]:
            # Random Schmidt values
            schmidt = np.random.rand(chi)
            schmidt = schmidt / np.linalg.norm(schmidt)

            cfs = compute_cfs(schmidt, normalize=False)

            # CFS should be in [1, χ]
            assert 1.0 <= cfs <= chi + 0.1  # Small tolerance for numerical error

    def test_monotonicity(self):
        """Test that more uniform distribution → higher CFS."""
        # Peaked distribution
        peaked = np.array([0.95, 0.05])
        peaked = peaked / np.linalg.norm(peaked)

        # More uniform distribution
        uniform = np.array([0.6, 0.4])
        uniform = uniform / np.linalg.norm(uniform)

        cfs_peaked = compute_cfs(peaked, normalize=False)
        cfs_uniform = compute_cfs(uniform, normalize=False)

        # Uniform should have higher CFS
        assert cfs_uniform > cfs_peaked


class TestCFSNumericalStability:
    """Test numerical stability of CFS computation."""

    def test_very_small_values(self):
        """Test CFS with very small Schmidt values."""
        schmidt = np.array([1e-8, 1e-9, 1e-10])
        schmidt = schmidt / np.linalg.norm(schmidt)

        cfs = compute_cfs(schmidt, normalize=False)

        # Should not produce NaN or Inf
        assert not np.isnan(cfs)
        assert not np.isinf(cfs)
        assert cfs >= 1.0

    def test_near_zero_purity(self):
        """Test handling when purity approaches zero."""
        # Very uniform distribution → low purity
        chi = 100
        schmidt = np.ones(chi) / np.sqrt(chi)

        cfs = compute_cfs(schmidt, normalize=False)

        # Should be close to χ
        assert np.isclose(cfs, chi, rtol=0.2)

    def test_all_zeros(self):
        """Test handling of all-zero Schmidt values."""
        schmidt = np.zeros(10)

        # Should return fallback CFS = 1.0
        cfs = compute_cfs(schmidt, normalize=True)
        assert np.isclose(cfs, 1.0)

    def test_high_precision(self):
        """Test CFS computation with high precision values."""
        schmidt = np.array([0.707106781186547, 0.707106781186547])

        cfs = compute_cfs(schmidt, normalize=False)

        # Should be very close to 2.0
        assert np.isclose(cfs, 2.0, atol=1e-6)


class TestCFSBatch:
    """Test batch CFS computation."""

    def test_batch_computation(self):
        """Test computing CFS for multiple Schmidt arrays."""
        schmidt_list = [
            np.array([0.8, 0.6]),
            np.array([0.7, 0.5, 0.3, 0.1]),
            np.array([1.0, 0.0, 0.0]),
        ]

        # Normalize each
        schmidt_list = [s / np.linalg.norm(s) for s in schmidt_list]

        # Batch computation
        cfs_batch = compute_cfs_batch(schmidt_list, normalize=False)

        # Individual computation
        cfs_individual = [compute_cfs(s, normalize=False) for s in schmidt_list]

        # Should match
        assert np.allclose(cfs_batch, cfs_individual)

    def test_batch_empty_list(self):
        """Test batch computation with empty list."""
        cfs_batch = compute_cfs_batch([])
        assert len(cfs_batch) == 0


class TestCFSFromMPS:
    """Test CFS computation from MPSSequence."""

    def test_cfs_from_mps(self):
        """Test computing CFS directly from MPS."""
        mps = MPSSequence(bond_dim=16, embedding_dim=64)

        # Add some tokens
        for _ in range(5):
            mps.add_token(torch.randn(64))

        # Compute CFS from MPS
        cfs = compute_cfs_from_mps(mps)

        # Should be valid CFS
        assert isinstance(cfs, float)
        assert cfs >= 1.0
        assert cfs <= mps.bond_dim + 0.1

    def test_cfs_evolution(self):
        """Test that CFS changes as tokens are added."""
        mps = MPSSequence(bond_dim=16, embedding_dim=64)

        cfs_values = []

        # Add tokens and track CFS
        for _ in range(10):
            mps.add_token(torch.randn(64))
            cfs = compute_cfs_from_mps(mps)
            cfs_values.append(cfs)

        # All should be valid
        assert all(1.0 <= c <= 16.1 for c in cfs_values)

        # Should have some variation (not all identical)
        assert np.std(cfs_values) > 0


class TestAnalyzeCoherenceSpectrum:
    """Test detailed coherence spectrum analysis."""

    def test_analysis_output(self):
        """Test that analysis returns all expected metrics."""
        schmidt = np.array([0.7, 0.5, 0.3, 0.2])
        schmidt = schmidt / np.linalg.norm(schmidt)

        analysis = analyze_coherence_spectrum(schmidt, normalize=False)

        # Check all keys present
        expected_keys = [
            'cfs', 'purity', 'effective_rank', 'entropy',
            'max_schmidt', 'uniformity', 'bond_dim'
        ]
        for key in expected_keys:
            assert key in analysis

        # Check value ranges
        assert analysis['cfs'] >= 1.0
        assert 0.0 <= analysis['purity'] <= 1.0
        assert analysis['effective_rank'] >= 1.0
        assert analysis['entropy'] >= 0.0
        assert 0.0 <= analysis['max_schmidt'] <= 1.0
        assert analysis['bond_dim'] == 4

    def test_analysis_uniform(self):
        """Test analysis for uniform distribution."""
        chi = 16
        schmidt = np.ones(chi) / np.sqrt(chi)

        analysis = analyze_coherence_spectrum(schmidt, normalize=False)

        # Uniform distribution should have high effective rank
        assert analysis['effective_rank'] > chi * 0.8

        # Should have high entropy
        assert analysis['entropy'] > np.log(chi) * 0.8

    def test_analysis_peaked(self):
        """Test analysis for peaked distribution."""
        schmidt = np.array([1.0, 0.0, 0.0, 0.0])

        analysis = analyze_coherence_spectrum(schmidt, normalize=False)

        # Peaked distribution should have low effective rank
        assert analysis['effective_rank'] < 1.5

        # Should have low entropy (close to 0)
        assert analysis['entropy'] < 0.1

    def test_analysis_max_schmidt(self):
        """Test that max_schmidt is correctly identified."""
        schmidt = np.array([0.9, 0.3, 0.2, 0.1])
        schmidt = schmidt / np.linalg.norm(schmidt)

        analysis = analyze_coherence_spectrum(schmidt, normalize=False)

        # Max should be close to first element (after normalization)
        assert analysis['max_schmidt'] > 0.5


class TestCFSProperties:
    """Test mathematical properties of CFS."""

    def test_cfs_purity_relationship(self):
        """Test that CFS = 1 / purity."""
        schmidt = np.random.rand(10)
        schmidt = schmidt / np.linalg.norm(schmidt)

        cfs, purity = compute_cfs(schmidt, return_purity=True, normalize=False)

        # Should satisfy F = 1/P
        assert np.isclose(cfs * purity, 1.0, atol=1e-6)

    def test_cfs_convexity(self):
        """Test CFS behavior under mixing of distributions."""
        # Two extreme distributions
        peaked = np.array([1.0, 0.0])
        uniform = np.array([1.0, 1.0]) / np.sqrt(2)

        # Mixture
        alpha = 0.5
        mixed = alpha * peaked + (1 - alpha) * uniform
        mixed = mixed / np.linalg.norm(mixed)

        cfs_peaked = compute_cfs(peaked, normalize=False)
        cfs_uniform = compute_cfs(uniform, normalize=False)
        cfs_mixed = compute_cfs(mixed, normalize=False)

        # Mixed should be between extremes
        assert min(cfs_peaked, cfs_uniform) <= cfs_mixed <= max(cfs_peaked, cfs_uniform)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
