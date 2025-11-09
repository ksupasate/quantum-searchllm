"""
TNAD: Tensor Network-Augmented Decoding

A quantum-inspired inference framework for improving logical coherence
in Large Language Model reasoning through real-time structural monitoring.

Core Components:
- MPSSequence: Matrix Product State representation of token sequences
- compute_cfs: Coherence Fidelity Score calculation
- FidelityGuidedBeamSearcher: FGBS algorithm implementation

Extended Features (v1.0):
- EncoderDecoderFGBS: Support for T5, BART, and other seq2seq models
- DistributedFGBS: Multi-GPU distributed beam search
- vLLMFGBS: Production deployment with vLLM integration
- StreamingFGBS: Real-time token-by-token generation
- CoherenceRewardTrainer: Fine-tuning with coherence rewards
- API Server: REST API and WebSocket support
- Web Demo: Interactive Gradio interface

Author: Professional AI Research Implementation
"""

# Core components
from tnad.mps_manager import MPSSequence
from tnad.coherence_score import compute_cfs, compute_cfs_from_mps
from tnad.fgbs_searcher import FidelityGuidedBeamSearcher, BeamHypothesis
from tnad.utils import log_normalize, safe_divide

# Extended features
from tnad.encoder_decoder_fgbs import EncoderDecoderFGBS, EncoderDecoderBeamHypothesis
from tnad.distributed_fgbs import (
    DistributedFGBS,
    DataParallelFGBS,
    setup_distributed,
    cleanup_distributed,
)
from tnad.vllm_integration import vLLMFGBS, CoherenceLogitsProcessor
from tnad.streaming_fgbs import (
    StreamingFGBS,
    BufferedStreamingFGBS,
    StreamingToken,
    create_sse_stream,
    create_websocket_stream,
)
from tnad.finetuning_pipeline import (
    CoherenceReward,
    CoherenceFilteredDataset,
    CoherenceRewardTrainer,
    CoherenceRLTrainer,
    evaluate_coherence,
)

__version__ = "1.0.0"
__all__ = [
    # Core
    "MPSSequence",
    "compute_cfs",
    "compute_cfs_from_mps",
    "FidelityGuidedBeamSearcher",
    "BeamHypothesis",
    "log_normalize",
    "safe_divide",
    # Encoder-Decoder
    "EncoderDecoderFGBS",
    "EncoderDecoderBeamHypothesis",
    # Distributed
    "DistributedFGBS",
    "DataParallelFGBS",
    "setup_distributed",
    "cleanup_distributed",
    # vLLM
    "vLLMFGBS",
    "CoherenceLogitsProcessor",
    # Streaming
    "StreamingFGBS",
    "BufferedStreamingFGBS",
    "StreamingToken",
    "create_sse_stream",
    "create_websocket_stream",
    # Fine-tuning
    "CoherenceReward",
    "CoherenceFilteredDataset",
    "CoherenceRewardTrainer",
    "CoherenceRLTrainer",
    "evaluate_coherence",
]
