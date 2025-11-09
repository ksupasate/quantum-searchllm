"""
Utility Functions for TNAD

Provides helper functions for numerical stability, logging,
and common tensor operations used throughout the framework.
"""

import torch
import numpy as np
from typing import Optional, Union

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


def log_normalize(
    logits: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Normalize logits in log-space using log-sum-exp trick for numerical stability.

    This prevents overflow/underflow when working with very large or small probabilities.
    Given log probabilities log(p_i), computes log(p_i / sum(p_i)).

    Args:
        logits: Input log-probabilities tensor
        dim: Dimension along which to normalize (default: -1)
        eps: Small epsilon for numerical stability

    Returns:
        Log-normalized probabilities

    Example:
        >>> logits = torch.tensor([1.0, 2.0, 3.0])
        >>> log_probs = log_normalize(logits)
        >>> probs = torch.exp(log_probs)
        >>> assert torch.allclose(probs.sum(), torch.tensor(1.0))
    """
    # log-sum-exp trick: log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    max_logit = torch.max(logits, dim=dim, keepdim=True)[0]
    shifted_logits = logits - max_logit
    log_sum_exp = max_logit + torch.log(
        torch.sum(torch.exp(shifted_logits), dim=dim, keepdim=True) + eps
    )
    return logits - log_sum_exp


def safe_divide(
    numerator: Union[float, torch.Tensor, np.ndarray],
    denominator: Union[float, torch.Tensor, np.ndarray],
    eps: float = 1e-10,
    replace_nan: bool = True,
    nan_value: float = 0.0,
) -> Union[float, torch.Tensor, np.ndarray]:
    """
    Perform safe division with numerical stability.

    Prevents division by zero and optionally handles NaN values.
    Critical for CFS computation where purity can approach zero.

    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        eps: Small epsilon added to denominator (default: 1e-10)
        replace_nan: If True, replace NaN results with nan_value
        nan_value: Value to use for NaN replacement

    Returns:
        Safe division result of same type as inputs

    Example:
        >>> safe_divide(1.0, 0.0)  # Returns finite value instead of inf
        >>> safe_divide(torch.tensor([1., 2.]), torch.tensor([0., 2.]))
    """
    result = numerator / (denominator + eps)

    if replace_nan:
        if isinstance(result, torch.Tensor):
            result = torch.nan_to_num(result, nan=nan_value)
        elif isinstance(result, np.ndarray):
            result = np.nan_to_num(result, nan=nan_value)
        elif np.isnan(result):
            result = nan_value

    return result


def truncate_svd(
    matrix: torch.Tensor,
    max_bond_dim: int,
    threshold: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform SVD with bond dimension truncation.

    Computes matrix = U @ diag(S) @ V^T and truncates to maximum bond dimension.
    Used in MPS construction to control entanglement complexity.

    Args:
        matrix: Input matrix to decompose (shape: [m, n])
        max_bond_dim: Maximum number of singular values to keep (χ)
        threshold: Optional threshold for singular value cutoff

    Returns:
        Tuple of (U, S, V) where:
            - U: Left singular vectors (shape: [m, χ'])
            - S: Singular values (shape: [χ'])
            - V: Right singular vectors (shape: [χ', n])
            where χ' = min(χ, rank(matrix))

    Implementation Details:
        - Uses torch.linalg.svd for GPU acceleration
        - Truncates based on max_bond_dim and optional threshold
        - Preserves most significant entanglement structure
    """
    original_device = matrix.device
    work_matrix = matrix
    moved_to_cpu = False

    if original_device.type == "mps":
        work_matrix = matrix.detach().to("cpu")
        moved_to_cpu = True

    try:
        U, S, Vh = torch.linalg.svd(work_matrix, full_matrices=False)
    except RuntimeError as exc:
        if not moved_to_cpu:
            work_matrix = matrix.detach().to("cpu")
            moved_to_cpu = True
            U, S, Vh = torch.linalg.svd(work_matrix, full_matrices=False)
        else:
            raise

    # Determine truncation point
    if threshold is not None:
        # Keep singular values above threshold
        above_threshold = S > threshold
        chi = min(above_threshold.sum().item(), max_bond_dim)
    else:
        chi = min(len(S), max_bond_dim)

    # Truncate
    U_trunc = U[:, :chi]
    S_trunc = S[:chi]
    V_trunc = Vh[:chi, :]

    if moved_to_cpu:
        U_trunc = U_trunc.to(original_device)
        S_trunc = S_trunc.to(original_device)
        V_trunc = V_trunc.to(original_device)

    return U_trunc, S_trunc, V_trunc


def normalize_schmidt_values(
    schmidt_values: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-10,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize Schmidt values to satisfy Σλ_i^2 = 1.

    Schmidt decomposition requires normalized coefficients representing
    a valid quantum state (trace-1 density matrix).

    Args:
        schmidt_values: Raw Schmidt coefficients from SVD
        eps: Small epsilon for numerical stability

    Returns:
        Normalized Schmidt values

    Example:
        >>> raw = torch.tensor([3.0, 4.0])  # ||raw||_2 = 5
        >>> normalized = normalize_schmidt_values(raw)
        >>> assert torch.allclose((normalized**2).sum(), torch.tensor(1.0))
    """
    if isinstance(schmidt_values, torch.Tensor):
        norm = torch.sqrt(torch.sum(schmidt_values ** 2) + eps)
    else:
        norm = np.sqrt(np.sum(schmidt_values ** 2) + eps)

    return schmidt_values / norm


def setup_logger(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure loguru logger for TNAD framework.

    Note: Only removes handlers if explicitly reconfiguring. Does not affect
    other loguru handlers to avoid breaking other modules.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
    """
    import sys

    # Only remove handlers that were previously added by this function
    # by checking if default handler exists
    try:
        logger.remove(0)  # Remove only the default handler with ID 0
    except ValueError:
        pass  # Default handler already removed or doesn't exist

    # Console handler with color using sys.stdout instead of lambda
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
    )

    # File handler if specified
    if log_file:
        logger.add(
            log_file,
            rotation="10 MB",
            retention="1 week",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            level=log_level,
        )

    return logger


def compute_purity(schmidt_values: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute purity from Schmidt values: Σλ_i^4.

    Purity measures how close a quantum state is to being pure (vs mixed).
    For normalized Schmidt values: 1/χ ≤ purity ≤ 1
    - purity = 1: maximally pure (single Schmidt value)
    - purity = 1/χ: maximally mixed (uniform distribution)

    Args:
        schmidt_values: Normalized Schmidt coefficients

    Returns:
        Purity value (float)

    Note:
        CFS is computed as F = 1 / purity, so low purity → high fidelity
    """
    if isinstance(schmidt_values, torch.Tensor):
        purity = torch.sum(schmidt_values ** 4).item()
    else:
        purity = np.sum(schmidt_values ** 4)

    return float(purity)


def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert PyTorch tensor to NumPy array (handles GPU tensors)."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def to_torch(array: Union[np.ndarray, torch.Tensor], device: str = "cpu") -> torch.Tensor:
    """Convert NumPy array to PyTorch tensor."""
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).to(device)
    return array.to(device)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get best available device (CUDA > MPS > CPU).

    Args:
        prefer_gpu: If True, prefer GPU over CPU when available

    Returns:
        torch.device object
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")
