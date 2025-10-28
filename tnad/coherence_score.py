"""
Coherence Fidelity Score (CFS) Computation

Implements the core metric for quantifying structural coherence
in token sequences via quantum-inspired purity measures.

Mathematical Foundation:
    Given Schmidt values λ = [λ₁, λ₂, ..., λ_χ] from MPS bipartition:

    1. Purity: P = Σᵢ λᵢ⁴  (measures "mixedness" of quantum state)
    2. CFS: F = 1 / P  (inverse purity, higher = more coherent)

    Interpretation:
    - High F (→ χ): Uniform Schmidt spectrum = high entanglement = coherent state
    - Low F (→ 1): Peaked Schmidt spectrum = low entanglement = decoherent state

    Physical Intuition:
    - Pure quantum state (single λᵢ = 1): P = 1, F = 1 (minimum coherence)
    - Maximally mixed state (uniform λᵢ = 1/√χ): P = 1/χ, F = χ (maximum coherence)
"""

import numpy as np
import torch
from typing import Union

try:  # pragma: no cover
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

from tnad.utils import safe_divide, compute_purity, normalize_schmidt_values


def compute_cfs(
    schmidt_values: Union[np.ndarray, torch.Tensor, list],
    normalize: bool = True,
    eps: float = 1e-10,
    return_purity: bool = False,
) -> Union[float, tuple[float, float]]:
    """
    Compute Coherence Fidelity Score from Schmidt values.

    Args:
        schmidt_values: Schmidt coefficients from MPS decomposition
        normalize: If True, ensure Σλᵢ² = 1 before computing (recommended)
        eps: Small epsilon for numerical stability in division
        return_purity: If True, return (CFS, purity) tuple

    Returns:
        CFS value (float), or (CFS, purity) if return_purity=True

    Raises:
        ValueError: If schmidt_values is empty or invalid

    Example:
        >>> # Maximally coherent state (uniform distribution)
        >>> schmidt = np.ones(16) / 4  # Normalized: sum(λ²) = 1
        >>> cfs = compute_cfs(schmidt)
        >>> print(f"CFS: {cfs:.2f}")  # Should be close to 16

        >>> # Decoherent state (single dominant value)
        >>> schmidt = np.array([1.0, 0.0, 0.0, 0.0])
        >>> cfs = compute_cfs(schmidt)
        >>> print(f"CFS: {cfs:.2f}")  # Should be close to 1

    Implementation Details:
        [Edge Cases]
        - Empty input: raises ValueError
        - Single value: returns F = 1 (minimum coherence)
        - Near-zero values: handled by eps in safe_divide
        - All zeros: returns F = 1 (fallback to minimum)

        [Numerical Stability]
        - Normalization ensures valid quantum state (Σλᵢ² = 1)
        - Safe division prevents inf from zero purity
        - Log-space computation could be added for extreme values
    """
    # Input validation
    if schmidt_values is None or len(schmidt_values) == 0:
        raise ValueError("schmidt_values cannot be empty")

    # Convert to numpy for consistent computation
    if isinstance(schmidt_values, torch.Tensor):
        schmidt_values = schmidt_values.detach().cpu().numpy()
    elif isinstance(schmidt_values, list):
        schmidt_values = np.array(schmidt_values)

    # Ensure 1D array
    schmidt_values = np.asarray(schmidt_values).flatten()

    # Handle edge case: all zeros
    if np.allclose(schmidt_values, 0.0, atol=1e-12):
        logger.warning("All Schmidt values are zero, returning CFS = 1.0")
        return (1.0, 1.0) if return_purity else 1.0

    # Optional normalization (recommended for numerical stability)
    if normalize:
        schmidt_values = normalize_schmidt_values(schmidt_values, eps=eps)

    # Compute purity: P = Σλᵢ⁴
    purity = compute_purity(schmidt_values)

    # Compute CFS: F = 1 / P
    cfs = safe_divide(
        numerator=1.0,
        denominator=purity,
        eps=eps,
        replace_nan=True,
        nan_value=1.0,  # Fallback to minimum coherence
    )

    # Ensure CFS is within valid range [1, χ]
    chi = len(schmidt_values)
    cfs = float(np.clip(cfs, 1.0, chi))

    logger.debug(
        f"CFS computed: F={cfs:.4f}, P={purity:.4f}, "
        f"χ={chi}, spectrum=[{schmidt_values[0]:.4f}, ..., {schmidt_values[-1]:.4f}]"
    )

    if return_purity:
        return cfs, purity
    return cfs


def compute_cfs_from_mps(mps_sequence, **kwargs) -> float:
    """
    Convenience function: compute CFS directly from MPSSequence.

    Args:
        mps_sequence: MPSSequence instance
        **kwargs: Additional arguments passed to compute_cfs

    Returns:
        CFS value (float)

    Example:
        >>> from tnad import MPSSequence, compute_cfs_from_mps
        >>> mps = MPSSequence(bond_dim=16, embedding_dim=768)
        >>> # ... add tokens ...
        >>> cfs = compute_cfs_from_mps(mps)
    """
    schmidt_values = mps_sequence.get_schmidt_values()
    return compute_cfs(schmidt_values, **kwargs)


def compute_cfs_batch(
    schmidt_values_list: list[Union[np.ndarray, torch.Tensor]],
    **kwargs
) -> np.ndarray:
    """
    Compute CFS for multiple Schmidt value arrays (useful for beam search).

    Args:
        schmidt_values_list: List of Schmidt value arrays
        **kwargs: Additional arguments passed to compute_cfs

    Returns:
        Array of CFS values

    Example:
        >>> # Multiple beams in beam search
        >>> schmidt_list = [beam.get_schmidt_values() for beam in beams]
        >>> cfs_scores = compute_cfs_batch(schmidt_list)
    """
    return np.array([compute_cfs(sv, **kwargs) for sv in schmidt_values_list])


def analyze_coherence_spectrum(
    schmidt_values: Union[np.ndarray, torch.Tensor],
    normalize: bool = True,
) -> dict:
    """
    Detailed analysis of Schmidt spectrum and coherence properties.

    Provides multiple metrics for understanding the quantum state structure:
    - CFS and purity
    - Effective rank (participation ratio)
    - Entropy measures
    - Spectrum uniformity

    Args:
        schmidt_values: Schmidt coefficients
        normalize: If True, normalize before analysis

    Returns:
        Dictionary with analysis metrics:
            - 'cfs': Coherence Fidelity Score
            - 'purity': Quantum purity (Σλᵢ⁴)
            - 'effective_rank': Participation ratio (1/Σλᵢ⁴)
            - 'entropy': Von Neumann entropy (-Σλᵢ² log λᵢ²)
            - 'max_schmidt': Largest Schmidt value
            - 'uniformity': Measure of spectrum flatness

    Example:
        >>> schmidt = mps.get_schmidt_values()
        >>> analysis = analyze_coherence_spectrum(schmidt)
        >>> print(f"Effective rank: {analysis['effective_rank']:.2f}/{len(schmidt)}")
    """
    # Convert to numpy
    if isinstance(schmidt_values, torch.Tensor):
        schmidt_values = schmidt_values.detach().cpu().numpy()
    schmidt_values = np.asarray(schmidt_values).flatten()

    # Normalize
    if normalize:
        schmidt_values = normalize_schmidt_values(schmidt_values)

    # Compute CFS and purity
    cfs, purity = compute_cfs(schmidt_values, normalize=False, return_purity=True)

    # Effective rank (inverse purity)
    effective_rank = 1.0 / (purity + 1e-10)

    # Von Neumann entropy: S = -Σλᵢ² log(λᵢ²)
    lambda_sq = schmidt_values ** 2
    # Avoid log(0)
    nonzero_mask = lambda_sq > 1e-12
    entropy = -np.sum(lambda_sq[nonzero_mask] * np.log(lambda_sq[nonzero_mask] + 1e-10))

    # Max Schmidt value (dominance)
    max_schmidt = np.max(schmidt_values)

    # Uniformity: how close to uniform distribution
    # Perfect uniform: λᵢ = 1/√χ for all i
    chi = len(schmidt_values)
    uniform_value = 1.0 / np.sqrt(chi)
    uniformity = 1.0 - np.std(schmidt_values) / uniform_value

    return {
        'cfs': float(cfs),
        'purity': float(purity),
        'effective_rank': float(effective_rank),
        'entropy': float(entropy),
        'max_schmidt': float(max_schmidt),
        'uniformity': float(uniformity),
        'bond_dim': int(chi),
    }


def coherence_trajectory(
    mps_sequence_list: list,
    **kwargs
) -> list[float]:
    """
    Compute CFS trajectory over a sequence of MPS states.

    Useful for visualizing how coherence evolves during generation.

    Args:
        mps_sequence_list: List of MPSSequence objects at different steps
        **kwargs: Additional arguments for compute_cfs

    Returns:
        List of CFS values, one per MPS state

    Example:
        >>> mps_history = []
        >>> for token in tokens:
        >>>     mps.add_token(token)
        >>>     mps_history.append(mps.copy())
        >>> cfs_trajectory = coherence_trajectory(mps_history)
        >>> plt.plot(cfs_trajectory)
    """
    return [compute_cfs_from_mps(mps, **kwargs) for mps in mps_sequence_list]


# Type alias for clarity
CoherenceScore = float
