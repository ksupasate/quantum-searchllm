"""
Unit Tests for MPSSequence

Tests the core tensor network operations including:
- MPS construction and token addition
- Schmidt decomposition correctness
- Bond dimension truncation
- Numerical stability
"""

import pytest
import torch
import numpy as np

from tnad.mps_manager import MPSSequence


class TestMPSSequence:
    """Test suite for MPSSequence class."""

    def test_initialization(self):
        """Test MPS initialization with various parameters."""
        # Basic initialization
        mps = MPSSequence(bond_dim=16, embedding_dim=768)
        assert mps.bond_dim == 16
        assert mps.embedding_dim == 768
        assert len(mps.tensors) == 0
        assert mps.get_current_length() == 0

        # With device specification
        mps_cpu = MPSSequence(bond_dim=8, embedding_dim=512, device="cpu")
        assert mps_cpu.device == torch.device("cpu")

        # With normalization
        mps_norm = MPSSequence(
            bond_dim=16,
            embedding_dim=768,
            normalize_embeddings=True
        )
        assert mps_norm.normalize_embeddings is True

    def test_add_single_token(self):
        """Test adding a single token to empty MPS."""
        mps = MPSSequence(bond_dim=16, embedding_dim=64)

        # Create random token embedding
        token_embedding = torch.randn(64)

        # Add token
        mps.add_token(token_embedding)

        # Check MPS state
        assert len(mps.tensors) == 1
        assert mps.get_current_length() == 1

        # First tensor should have left boundary = 1
        tensor = mps.tensors[0]
        assert tensor.shape[0] == 1  # Left boundary
        assert tensor.shape[1] == 64  # Physical dimension
        assert tensor.shape[2] <= 16  # Right boundary ≤ χ

    def test_add_multiple_tokens(self):
        """Test adding multiple tokens sequentially."""
        mps = MPSSequence(bond_dim=16, embedding_dim=64)

        # Add 5 tokens
        for i in range(5):
            token_embedding = torch.randn(64)
            mps.add_token(token_embedding)

        # Check length
        assert mps.get_current_length() == 5
        assert len(mps.tensors) == 5

        # Check each tensor has correct structure
        for i, tensor in enumerate(mps.tensors):
            assert tensor.ndim == 3  # Rank-3 tensor
            assert tensor.shape[1] == 64  # Physical dimension preserved

    def test_schmidt_values_single_token(self):
        """Test Schmidt value extraction from single-token MPS."""
        mps = MPSSequence(bond_dim=16, embedding_dim=64)
        token_embedding = torch.randn(64)
        mps.add_token(token_embedding)

        # Get Schmidt values
        schmidt_values = mps.get_schmidt_values()

        # Check properties
        assert len(schmidt_values) > 0
        assert np.allclose(np.sum(schmidt_values ** 2), 1.0, atol=1e-5)  # Normalized
        assert np.all(schmidt_values >= 0)  # Non-negative
        assert np.all(schmidt_values[:-1] >= schmidt_values[1:])  # Descending order

    def test_schmidt_values_multiple_tokens(self):
        """Test Schmidt value extraction from multi-token MPS."""
        mps = MPSSequence(bond_dim=16, embedding_dim=64)

        # Add multiple tokens
        for _ in range(10):
            token_embedding = torch.randn(64)
            mps.add_token(token_embedding)

        # Get Schmidt values
        schmidt_values = mps.get_schmidt_values()

        # Check properties
        assert len(schmidt_values) <= mps.bond_dim  # Bounded by χ
        assert np.allclose(np.sum(schmidt_values ** 2), 1.0, atol=1e-5)
        assert np.all(schmidt_values >= 0)

    def test_bond_dimension_truncation(self):
        """Test that bond dimension is properly truncated to χ."""
        small_chi = 4
        mps = MPSSequence(bond_dim=small_chi, embedding_dim=32)

        # Add enough tokens to exceed bond dimension
        for _ in range(10):
            token_embedding = torch.randn(32)
            mps.add_token(token_embedding)

        # Check bond dimensions stay within limit
        for tensor in mps.tensors:
            assert tensor.shape[2] <= small_chi  # Right bond ≤ χ

        # Schmidt values should also respect limit
        schmidt_values = mps.get_schmidt_values()
        assert len(schmidt_values) <= small_chi

    def test_embedding_dimension_mismatch(self):
        """Test error handling for wrong embedding dimension."""
        mps = MPSSequence(bond_dim=16, embedding_dim=64)

        # Try to add token with wrong dimension
        wrong_embedding = torch.randn(128)  # Wrong dimension

        with pytest.raises(ValueError, match="Embedding dim mismatch"):
            mps.add_token(wrong_embedding)

    def test_numpy_input(self):
        """Test that NumPy arrays are accepted as input."""
        mps = MPSSequence(bond_dim=16, embedding_dim=64)

        # NumPy array input
        token_embedding = np.random.randn(64)
        mps.add_token(token_embedding)

        assert len(mps.tensors) == 1

    def test_normalization(self):
        """Test embedding normalization option."""
        # With normalization
        mps_norm = MPSSequence(
            bond_dim=16,
            embedding_dim=64,
            normalize_embeddings=True
        )

        token = torch.randn(64) * 100  # Large values
        mps_norm.add_token(token)

        # Without normalization
        mps_no_norm = MPSSequence(
            bond_dim=16,
            embedding_dim=64,
            normalize_embeddings=False
        )

        mps_no_norm.add_token(token)

        # Both should work without errors
        assert len(mps_norm.tensors) == 1
        assert len(mps_no_norm.tensors) == 1

    def test_copy(self):
        """Test MPS copying for beam search branching."""
        # Create original MPS
        mps1 = MPSSequence(bond_dim=16, embedding_dim=64)
        for _ in range(3):
            mps1.add_token(torch.randn(64))

        # Copy
        mps2 = mps1.copy()

        # Check independence
        assert mps2.get_current_length() == mps1.get_current_length()
        assert mps2.bond_dim == mps1.bond_dim

        # Modify copy
        mps2.add_token(torch.randn(64))

        # Original should be unchanged
        assert mps2.get_current_length() == mps1.get_current_length() + 1

    def test_schmidt_cut_position(self):
        """Test Schmidt decomposition at different cut positions."""
        mps = MPSSequence(bond_dim=16, embedding_dim=64)

        # Add tokens
        for _ in range(6):
            mps.add_token(torch.randn(64))

        # Try different cut positions
        for cut_pos in range(1, 6):
            schmidt_values = mps.get_schmidt_values(cut_position=cut_pos)
            assert len(schmidt_values) > 0
            assert np.allclose(np.sum(schmidt_values ** 2), 1.0, atol=1e-5)

    def test_schmidt_invalid_cut(self):
        """Test error handling for invalid cut positions."""
        mps = MPSSequence(bond_dim=16, embedding_dim=64)

        for _ in range(5):
            mps.add_token(torch.randn(64))

        # Invalid cut positions
        with pytest.raises(ValueError):
            mps.get_schmidt_values(cut_position=0)  # Too small

        with pytest.raises(ValueError):
            mps.get_schmidt_values(cut_position=5)  # Too large

    def test_empty_mps_schmidt(self):
        """Test Schmidt values from empty MPS."""
        mps = MPSSequence(bond_dim=16, embedding_dim=64)

        # Empty MPS should return trivial Schmidt values
        schmidt_values = mps.get_schmidt_values()
        assert len(schmidt_values) == 1
        assert np.isclose(schmidt_values[0], 1.0)

    def test_device_consistency(self):
        """Test that tensors stay on correct device."""
        if torch.cuda.is_available():
            mps = MPSSequence(bond_dim=16, embedding_dim=64, device="cuda")

            token = torch.randn(64)
            mps.add_token(token)

            # Check tensor device
            assert mps.tensors[0].device.type == "cuda"

    def test_long_sequence(self):
        """Test MPS with longer sequences (stress test)."""
        mps = MPSSequence(bond_dim=16, embedding_dim=128)

        # Add 50 tokens
        for i in range(50):
            mps.add_token(torch.randn(128))

        # Should complete without errors
        assert mps.get_current_length() == 50

        # Schmidt values should still be valid
        schmidt_values = mps.get_schmidt_values()
        assert np.allclose(np.sum(schmidt_values ** 2), 1.0, atol=1e-4)

    def test_different_bond_dimensions(self):
        """Test various bond dimensions."""
        for chi in [4, 8, 16, 32]:
            mps = MPSSequence(bond_dim=chi, embedding_dim=64)

            for _ in range(10):
                mps.add_token(torch.randn(64))

            schmidt_values = mps.get_schmidt_values()
            assert len(schmidt_values) <= chi

    def test_repr(self):
        """Test string representation."""
        mps = MPSSequence(bond_dim=16, embedding_dim=768)
        mps.add_token(torch.randn(768))

        repr_str = repr(mps)
        assert "MPSSequence" in repr_str
        assert "length=1" in repr_str
        assert "bond_dim=16" in repr_str


class TestMPSNumericalStability:
    """Test numerical stability of MPS operations."""

    def test_very_small_values(self):
        """Test MPS with very small embedding values."""
        mps = MPSSequence(bond_dim=16, embedding_dim=64)

        # Very small values
        token = torch.randn(64) * 1e-8
        mps.add_token(token)

        schmidt_values = mps.get_schmidt_values()
        assert not np.any(np.isnan(schmidt_values))
        assert not np.any(np.isinf(schmidt_values))

    def test_very_large_values(self):
        """Test MPS with very large embedding values."""
        mps = MPSSequence(
            bond_dim=16,
            embedding_dim=64,
            normalize_embeddings=True  # Important for stability
        )

        # Very large values
        token = torch.randn(64) * 1e8
        mps.add_token(token)

        schmidt_values = mps.get_schmidt_values()
        assert not np.any(np.isnan(schmidt_values))
        assert not np.any(np.isinf(schmidt_values))

    def test_zero_embedding(self):
        """Test handling of zero embeddings."""
        mps = MPSSequence(bond_dim=16, embedding_dim=64)

        # Zero embedding
        token = torch.zeros(64)
        mps.add_token(token)

        # Should handle gracefully
        assert len(mps.tensors) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
