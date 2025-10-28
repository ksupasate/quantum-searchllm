"""
MPS Manager: Matrix Product State Representation

Provides a lightweight, differentiable approximation of the Matrix Product
State (MPS) machinery required for Fidelity-Guided Beam Search (FGBS).

Design goals:
    1. Track the evolving "logical state" of a token sequence using a
       compact bond-dimension latent (χ-dimensional hidden state).
    2. Produce Schmidt spectra across arbitrary bipartitions to support the
       Coherence Fidelity Score (CFS) in real time.
    3. Remain numerically stable and inexpensive enough for use inside a
       decoding loop where the structure must be cloned per beam.

Implementation sketch:
    * Each incoming token embedding e_t ∈ ℝ^d is projected into the χ-dimensional
      latent space via a deterministic random projection W_in.
    * A recurrent transition matrix W_state (orthonormal) mixes the latent state
      to capture correlations across time.
    * The latent vectors {h_t} act as the virtual bonds of an MPS. Correlation
      matrices constructed from the prefix/suffix latent stacks yield the
      Schmidt singular values required for CFS.
    * Rank-3 tensors are stored for compatibility/testing. They are synthesised
      from neighbouring latent vectors and the original embedding, providing a
      physically interpretable footprint without inflating compute.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

try:  # pragma: no cover - fallback for environments without loguru
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

from tnad.utils import normalize_schmidt_values

TensorLike = Union[torch.Tensor, np.ndarray]
EPS = 1e-8


def _orthonormal_matrix(
    dim: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create an orthonormal matrix using QR decomposition."""
    raw = torch.randn((dim, dim), generator=generator, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(raw)
    return q


def _svdvals_with_fallback(matrix: torch.Tensor, max_rank: int) -> torch.Tensor:
    """
    Compute singular values with a CPU fallback for devices lacking SVD/EIGH support.
    """
    device = matrix.device
    work_matrix = matrix
    moved_to_cpu = False

    if device.type == "mps":
        work_matrix = matrix.detach().to("cpu")
        moved_to_cpu = True

    try:
        singular_values = torch.linalg.svdvals(work_matrix)
    except RuntimeError:
        gram = work_matrix @ work_matrix.transpose(0, 1)
        gram = torch.clamp(gram, min=0.0)
        eigvals = torch.linalg.eigvalsh(gram)
        eigvals = torch.clamp(eigvals, min=0.0)
        singular_values = torch.sqrt(eigvals)

    singular_values = singular_values[:max_rank]

    if moved_to_cpu:
        singular_values = singular_values.to(device)

    return singular_values


class MPSSequence:
    """
    Matrix Product State representation of an autoregressive token sequence.

    The class maintains a list of latent "bond" states together with synthetic
    rank-3 tensors that mimic a classical MPS chain. The latent states are used
    to compute Schmidt spectra while the tensors provide a convenient diagnostic
    view and support existing tests.

    Attributes
    ----------
    bond_dim:
        Maximum virtual bond dimension χ.
    embedding_dim:
        Token embedding dimension d (taken from the model's embedding table).
    tensors:
        List of rank-3 tensors storing a compact structural footprint of each
        step. They are regenerated from latent states and embeddings.
    device:
        Compute device (cpu / cuda / mps).
    """

    def __init__(
        self,
        bond_dim: int,
        embedding_dim: int,
        device: Optional[Union[str, torch.device]] = None,
        normalize_embeddings: bool = True,
        seed: int = 0,
        state_transition: Optional[torch.Tensor] = None,
        input_projection: Optional[torch.Tensor] = None,
    ):
        if bond_dim <= 0:
            raise ValueError("bond_dim must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        self.bond_dim = int(bond_dim)
        self.embedding_dim = int(embedding_dim)
        self.normalize_embeddings = normalize_embeddings
        self.seed = seed

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.dtype = torch.float32

        # Random generators for reproducible projections.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        if input_projection is None:
            projection = torch.randn(
                self.embedding_dim,
                self.bond_dim,
                generator=generator,
                dtype=self.dtype,
            ) / math.sqrt(self.embedding_dim)
        else:
            projection = input_projection.detach().to(dtype=self.dtype)

        if state_transition is None:
            transition = _orthonormal_matrix(
                self.bond_dim,
                generator=generator,
                device="cpu",
                dtype=self.dtype,
            )
        else:
            transition = state_transition.detach().to(dtype=self.dtype)

        self._input_projection = projection.to(self.device)
        self._state_transition = transition.to(self.device)

        self.tensors: List[torch.Tensor] = []
        self._latent_states: List[torch.Tensor] = []
        self._embeddings: List[torch.Tensor] = []
        self._schmidt_cache: Dict[int, np.ndarray] = {}
        self._current_right_bond_dim: int = 1

        logger.debug(
            "Initialized MPSSequence: χ=%d, d=%d, device=%s",
            self.bond_dim,
            self.embedding_dim,
            self.device,
        )

    # --------------------------------------------------------------------- #
    # Core operations
    # --------------------------------------------------------------------- #
    def add_token(self, token_embedding: TensorLike) -> None:
        """
        Add a new token embedding to the MPS chain.

        Parameters
        ----------
        token_embedding:
            Token embedding vector of shape [embedding_dim].
        """
        embedding = self._prepare_embedding(token_embedding)

        projected = embedding @ self._input_projection  # [χ]
        prev_state = (
            self._latent_states[-1]
            if self._latent_states
            else torch.zeros(self.bond_dim, device=self.device, dtype=self.dtype)
        )

        # Update latent state with tanh non-linearity for bounded dynamics.
        state_raw = projected + prev_state @ self._state_transition
        state = torch.tanh(state_raw)
        state = F.normalize(state, dim=0, eps=EPS)
        if torch.linalg.vector_norm(state) < EPS:
            state = torch.zeros_like(state)
            state[0] = 1.0

        self._latent_states.append(state)
        self._embeddings.append(embedding)
        self._schmidt_cache.clear()

        self._append_tensor(state, embedding)
        self._current_right_bond_dim = min(self.bond_dim, len(self._latent_states))

        logger.debug(
            "Token %d added: ||state||=%.4f, right_bond=%d",
            len(self._latent_states),
            float(torch.linalg.vector_norm(state)),
            self._current_right_bond_dim,
        )

    def _prepare_embedding(self, embedding: TensorLike) -> torch.Tensor:
        """Validate, cast and optionally normalise an embedding vector."""
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)
        if not isinstance(embedding, torch.Tensor):
            raise TypeError("token_embedding must be torch.Tensor or np.ndarray")

        embedding = embedding.to(device=self.device, dtype=self.dtype).flatten()
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embedding_dim}, "
                f"got {embedding.shape[0]}"
            )

        if self.normalize_embeddings:
            norm = torch.linalg.vector_norm(embedding)
            if norm > EPS:
                embedding = embedding / norm

        return embedding

    def _append_tensor(self, state: torch.Tensor, embedding: torch.Tensor) -> None:
        """
        Construct a rank-3 tensor footprint for the new site.

        The left virtual dimension uses the previous latent state (if available),
        while the right virtual dimension uses the current state. This mirrors the
        intuition that the entanglement between prefixes/suffixes is mediated by
        successive latent states.
        """
        seq_len = len(self._latent_states)
        if seq_len == 0:
            return

        if seq_len == 1:
            left_dim = 1
            left_vector = torch.ones(1, device=self.device, dtype=self.dtype)
        else:
            left_dim = min(self.bond_dim, max(1, seq_len - 1))
            left_vector = self._latent_states[-2][:left_dim]

        right_dim = min(self.bond_dim, seq_len)
        right_vector = state[:right_dim]

        tensor = torch.einsum(
            "i,j,k->ijk",
            left_vector,
            embedding,
            right_vector,
        ).to(self.device)

        self.tensors.append(tensor)

    # --------------------------------------------------------------------- #
    # Schmidt spectrum extraction
    # --------------------------------------------------------------------- #
    def get_schmidt_values(self, cut_position: Optional[int] = None) -> np.ndarray:
        """
        Compute Schmidt coefficients across a bipartition of the latent chain.

        Parameters
        ----------
        cut_position:
            Integer index 1 ≤ cut_position ≤ len(sequence) - 1.
            If None, the sequence is split in half.
        """
        num_tokens = len(self._latent_states)
        if num_tokens == 0:
            return np.array([1.0], dtype=np.float32)
        if num_tokens == 1:
            return np.array([1.0], dtype=np.float32)

        if cut_position is None:
            cut_position = num_tokens // 2

        if not (0 < cut_position < num_tokens):
            raise ValueError(
                f"Invalid cut position {cut_position} for MPS of length {num_tokens}"
            )

        cache_key = cut_position
        cached = self._schmidt_cache.get(cache_key)
        if cached is not None:
            return cached

        left_states = torch.stack(
            self._latent_states[:cut_position], dim=0
        ).to(self.dtype)
        right_states = torch.stack(
            self._latent_states[cut_position:], dim=0
        ).to(self.dtype)

        # Centre states to avoid bias towards absolute offsets.
        left_states = left_states - left_states.mean(dim=0, keepdim=True)
        right_states = right_states - right_states.mean(dim=0, keepdim=True)

        cross_correlation = left_states @ right_states.transpose(0, 1)
        cross_correlation = cross_correlation / max(cross_correlation.shape[0], 1)

        singular_values = _svdvals_with_fallback(
            cross_correlation, max_rank=self.bond_dim
        )

        if singular_values.numel() == 0:
            singular_values = torch.ones(1, device=self.device, dtype=self.dtype)

        schmidt = normalize_schmidt_values(singular_values.detach().cpu().numpy())
        schmidt = np.abs(schmidt)
        self._schmidt_cache[cache_key] = schmidt
        return schmidt

    # --------------------------------------------------------------------- #
    # Introspection helpers
    # --------------------------------------------------------------------- #
    def get_current_length(self) -> int:
        return len(self._latent_states)

    def get_current_bond_dim(self) -> int:
        return self._current_right_bond_dim

    def copy(self) -> "MPSSequence":
        """
        Deep copy the sequence (used when branching beams in search).
        """
        new_mps = MPSSequence(
            bond_dim=self.bond_dim,
            embedding_dim=self.embedding_dim,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
            seed=self.seed,
            state_transition=self._state_transition.clone().detach(),
            input_projection=self._input_projection.clone().detach(),
        )

        new_mps.tensors = [tensor.clone() for tensor in self.tensors]
        new_mps._latent_states = [state.clone() for state in self._latent_states]
        new_mps._embeddings = [emb.clone() for emb in self._embeddings]
        new_mps._current_right_bond_dim = self._current_right_bond_dim
        new_mps._schmidt_cache = {
            key: value.copy() for key, value in self._schmidt_cache.items()
        }
        return new_mps

    # --------------------------------------------------------------------- #
    # Debug utilities
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        return (
            "MPSSequence(length={length}, bond_dim={bond}, embedding_dim={embed}, "
            "current_bond={current})"
        ).format(
            length=len(self._latent_states),
            bond=self.bond_dim,
            embed=self.embedding_dim,
            current=self._current_right_bond_dim,
        )


__all__ = ["MPSSequence"]
