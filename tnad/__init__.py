"""
TNAD: Tensor Network-Augmented Decoding

A quantum-inspired inference framework for improving logical coherence
in Large Language Model reasoning through real-time structural monitoring.

Core Components:
- MPSSequence: Matrix Product State representation of token sequences
- compute_cfs: Coherence Fidelity Score calculation
- FidelityGuidedBeamSearcher: FGBS algorithm implementation

Author: Professional AI Research Implementation
"""

from tnad.mps_manager import MPSSequence
from tnad.coherence_score import compute_cfs
from tnad.fgbs_searcher import FidelityGuidedBeamSearcher
from tnad.utils import log_normalize, safe_divide

__version__ = "0.1.0"
__all__ = [
    "MPSSequence",
    "compute_cfs",
    "FidelityGuidedBeamSearcher",
    "log_normalize",
    "safe_divide",
]
