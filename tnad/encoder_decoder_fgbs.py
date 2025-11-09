"""
Encoder-Decoder Fidelity-Guided Beam Search (ED-FGBS)

Extends FGBS to support encoder-decoder architectures (T5, BART, mBART, etc.)

Key Differences from Decoder-Only FGBS:
    1. Encoder processes input once, generating contextual representations
    2. Decoder generates output autoregressively using encoder states
    3. Cross-attention creates dependencies between encoder and decoder
    4. MPS tracking focuses on decoder sequence coherence

Algorithm:
    1. Encode input text to get encoder hidden states
    2. Initialize decoder with BOS token
    3. For each generation step:
        a. Decoder forward pass with encoder states (cross-attention)
        b. Get next-token log probabilities
        c. Expand beams with top-k candidates
        d. Update MPS for decoder sequence
        e. Compute CFS and composite scores
        f. Select top-B beams
    4. Return best hypothesis

Supported Models:
    - T5 (t5-small, t5-base, t5-large, t5-3b, t5-11b)
    - BART (facebook/bart-base, facebook/bart-large)
    - mBART (facebook/mbart-large-50, facebook/mbart-large-cc25)
    - mT5 (google/mt5-small, google/mt5-base, google/mt5-large)
"""

import gc
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

try:
    from loguru import logger
except ImportError:
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

from tqdm import tqdm

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
)

from tnad.coherence_score import compute_cfs_from_mps
from tnad.mps_manager import MPSSequence
from tnad.utils import get_device


@dataclass
class EncoderDecoderBeamHypothesis:
    """
    Single hypothesis for encoder-decoder beam search.

    Attributes:
        decoder_token_ids: Generated decoder token sequence [L]
        log_prob: Cumulative log probability from model
        log_cfs: Cumulative log CFS (coherence score) for decoder
        composite_score: α·log_prob + (1-α)·log_cfs
        mps: MPS representation of decoder sequence
        encoder_outputs: Cached encoder hidden states (shared across beams)
        is_finished: Whether sequence ends with EOS token
    """

    decoder_token_ids: List[int]
    log_prob: float
    log_cfs: float
    composite_score: float
    mps: MPSSequence
    encoder_outputs: Any  # BaseModelOutput from transformers
    is_finished: bool = False

    def __repr__(self) -> str:
        return (
            f"EDBeamHypothesis(len={len(self.decoder_token_ids)}, "
            f"score={self.composite_score:.4f}, "
            f"log_p={self.log_prob:.4f}, log_F={self.log_cfs:.4f})"
        )


class EncoderDecoderFGBS:
    """
    Fidelity-Guided Beam Search for Encoder-Decoder Models.

    Supports T5, BART, mBART, and mT5 architectures for sequence-to-sequence
    generation with coherence-aware decoding.

    Example Usage:
        >>> from transformers import T5ForConditionalGeneration, T5Tokenizer
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-base")
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-base")
        >>>
        >>> searcher = EncoderDecoderFGBS(
        >>>     model=model,
        >>>     tokenizer=tokenizer,
        >>>     beam_width=5,
        >>>     alpha=0.5,
        >>>     bond_dim=16,
        >>> )
        >>>
        >>> # Summarization
        >>> input_text = "summarize: Large language models have shown remarkable capabilities..."
        >>> result = searcher.generate(input_text, max_length=100)
        >>> print(result['text'])
        >>>
        >>> # Translation
        >>> input_text = "translate English to French: Hello, how are you?"
        >>> result = searcher.generate(input_text, max_length=50)
        >>> print(result['text'])
    """

    def __init__(
        self,
        model: Union[T5ForConditionalGeneration, BartForConditionalGeneration, PreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        beam_width: int = 5,
        alpha: float = 0.5,
        bond_dim: int = 16,
        top_k: int = 50,
        temperature: float = 1.0,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize Encoder-Decoder FGBS searcher.

        Args:
            model: Pretrained encoder-decoder model (T5, BART, etc.)
            tokenizer: Corresponding tokenizer
            beam_width: Number of parallel beams B (typical: 5-10)
            alpha: Fluency vs coherence balance ∈ [0,1]
                   α=1: pure model probability (standard beam search)
                   α=0: pure coherence (may sacrifice fluency)
                   α=0.5: balanced (recommended)
            bond_dim: MPS bond dimension χ (typical: 8-32)
            top_k: Number of top tokens to consider per beam
            temperature: Sampling temperature (1.0 = no scaling)
            device: Compute device ('cpu', 'cuda', 'mps')
            normalize_embeddings: Normalize token embeddings in MPS

        Raises:
            ValueError: If hyperparameters are invalid
            TypeError: If model/tokenizer types are incorrect
        """
        # Input validation
        if not isinstance(model, PreTrainedModel):
            raise TypeError(f"model must be PreTrainedModel, got {type(model)}")
        if not isinstance(tokenizer, PreTrainedTokenizer):
            raise TypeError(
                f"tokenizer must be PreTrainedTokenizer, got {type(tokenizer)}"
            )

        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if bond_dim < 1:
            raise ValueError(f"bond_dim must be >= 1, got {bond_dim}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        # Check if model is encoder-decoder
        if not hasattr(model, "get_encoder") or not hasattr(model, "get_decoder"):
            raise TypeError(
                "model must be an encoder-decoder architecture (T5, BART, etc.)"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.alpha = alpha
        self.bond_dim = bond_dim
        self.top_k = top_k
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings

        # Detect quantization
        self._is_quantized = bool(
            getattr(self.model, "is_loaded_in_8bit", False)
            or getattr(self.model, "is_loaded_in_4bit", False)
        )

        # Device setup
        if device is None:
            if self._is_quantized:
                # Get device from model's decoder embeddings
                embed_weight = self.model.get_decoder().get_input_embeddings().weight
                self.device = torch.device(embed_weight.device)
            else:
                self.device = get_device(prefer_gpu=True)
        else:
            self.device = torch.device(device)

        if not self._is_quantized:
            self.model = self.model.to(self.device)
        else:
            embed_weight = self.model.get_decoder().get_input_embeddings().weight
            self.device = torch.device(embed_weight.device)

        self.model.eval()  # Inference mode

        # Extract decoder embedding layer for MPS construction
        self.decoder_embedding_layer = self.model.get_decoder().get_input_embeddings()
        self.embedding_dim = self.decoder_embedding_layer.embedding_dim

        # Special tokens
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id
        self.bos_token_id = (
            self.tokenizer.bos_token_id or self.tokenizer.pad_token_id
        )
        self.decoder_start_token_id = (
            self.model.config.decoder_start_token_id
            or self.bos_token_id
            or self.pad_token_id
        )

        logger.info(
            f"Initialized ED-FGBS: model={model.__class__.__name__}, "
            f"B={beam_width}, α={alpha}, χ={bond_dim}, "
            f"d={self.embedding_dim}, device={self.device}"
        )

    def generate(
        self,
        input_text: str,
        max_length: int = 100,
        min_length: int = 10,
        return_details: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate text using Encoder-Decoder FGBS algorithm.

        Algorithm:
            1. Encode input text to get encoder_outputs
            2. Initialize beams with decoder_start_token
            3. For each step t ∈ [1, max_length]:
                a. For each active beam:
                   - Decoder forward pass with encoder_outputs
                   - Get next-token log probabilities
                   - Select top-K candidate tokens
                b. For each (beam, candidate) pair:
                   - Create new decoder sequence
                   - Update MPS for decoder tokens
                   - Compute CFS from Schmidt values
                   - Calculate composite score
                c. Select top-B hypotheses by composite score
                d. Check termination
            4. Return best hypothesis

        Args:
            input_text: Input text to encode
            max_length: Maximum decoder generation length (tokens)
            min_length: Minimum generation length before allowing EOS
            return_details: If True, return detailed generation info
            show_progress: Show progress bar

        Returns:
            Dictionary containing:
                - 'text': Generated text (str)
                - 'token_ids': Decoder token IDs (List[int])
                - 'log_prob': Final log probability (float)
                - 'log_cfs': Final log CFS (float)
                - 'composite_score': Final composite score (float)
                If return_details=True, also includes:
                - 'all_beams': All beam hypotheses at final step
                - 'cfs_trajectory': CFS values over generation
                - 'score_trajectory': Composite scores over generation

        Raises:
            ValueError: If input_text is empty or length parameters are invalid
            TypeError: If input_text is not a string

        Example:
            >>> # Summarization with T5
            >>> result = searcher.generate(
            >>>     "summarize: The quick brown fox jumps over the lazy dog.",
            >>>     max_length=50,
            >>>     return_details=True
            >>> )
            >>> print(result['text'])
            >>> print(f"CFS: {result['log_cfs']:.2f}")
        """
        # Input validation
        if not isinstance(input_text, str):
            raise TypeError(f"input_text must be string, got {type(input_text)}")
        if not input_text or not input_text.strip():
            raise ValueError("input_text cannot be empty")
        if len(input_text) > 50000:
            raise ValueError(
                f"Input too long: {len(input_text)} characters (max 50000)"
            )
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")
        if min_length < 0:
            raise ValueError(f"min_length must be >= 0, got {min_length}")
        if min_length > max_length:
            raise ValueError(
                f"min_length ({min_length}) cannot exceed max_length ({max_length})"
            )

        # Tokenize and encode input
        logger.info(f"Encoding input text: {len(input_text)} chars")
        encoded_input = self.tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = encoded_input.input_ids.to(self.device)
        attention_mask = encoded_input.attention_mask.to(self.device)

        # Run encoder once (shared across all beams)
        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(
                input_ids=input_ids, attention_mask=attention_mask
            )

        logger.info(f"Encoder output shape: {encoder_outputs.last_hidden_state.shape}")

        # Initialize beams with decoder_start_token
        beams = self._initialize_beams(encoder_outputs)

        # Track trajectories
        cfs_trajectory = []
        score_trajectory = []

        # Generation loop
        progress_bar = tqdm(
            range(max_length), desc="Generating", disable=not show_progress
        )

        for step in progress_bar:
            # Check if all beams finished
            if all(beam.is_finished for beam in beams):
                logger.info(f"All beams finished at step {step}")
                break

            # FGBS step
            beams = self._fgbs_step(beams, min_length, step, attention_mask)

            # Track best beam
            best_beam = max(beams, key=lambda b: b.composite_score)
            cfs_trajectory.append(math.exp(best_beam.log_cfs))
            score_trajectory.append(best_beam.composite_score)

            # Memory management
            if step % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Update progress
            progress_bar.set_postfix(
                {
                    "length": len(best_beam.decoder_token_ids),
                    "score": f"{best_beam.composite_score:.2f}",
                    "cfs": f"{math.exp(best_beam.log_cfs):.2f}",
                }
            )

        # Select best beam
        best_beam = max(beams, key=lambda b: b.composite_score)

        # Decode tokens
        generated_text = self.tokenizer.decode(
            best_beam.decoder_token_ids, skip_special_tokens=True
        )

        # Prepare result
        result = {
            "text": generated_text,
            "token_ids": best_beam.decoder_token_ids,
            "log_prob": best_beam.log_prob,
            "log_cfs": best_beam.log_cfs,
            "composite_score": best_beam.composite_score,
        }

        if return_details:
            result.update(
                {
                    "all_beams": beams,
                    "cfs_trajectory": cfs_trajectory,
                    "score_trajectory": score_trajectory,
                }
            )

        logger.info(
            f"Generation complete: length={len(best_beam.decoder_token_ids)}, "
            f"score={best_beam.composite_score:.4f}"
        )

        return result

    def _initialize_beams(
        self, encoder_outputs: Any
    ) -> List[EncoderDecoderBeamHypothesis]:
        """
        Initialize beam search with decoder_start_token.

        Args:
            encoder_outputs: Output from encoder

        Returns:
            List containing single initial beam
        """
        # Create MPS for decoder sequence
        mps = MPSSequence(
            bond_dim=self.bond_dim,
            embedding_dim=self.embedding_dim,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        )

        # Add decoder_start_token to MPS
        with torch.no_grad():
            start_token_tensor = torch.tensor(
                [self.decoder_start_token_id], device=self.device
            )
            start_embedding = self.decoder_embedding_layer(start_token_tensor)[0]
            mps.add_token(start_embedding)

        # Compute initial CFS
        initial_cfs = compute_cfs_from_mps(mps)
        log_cfs = math.log(max(initial_cfs, 1e-12))

        # Create initial beam
        initial_beam = EncoderDecoderBeamHypothesis(
            decoder_token_ids=[self.decoder_start_token_id],
            log_prob=0.0,
            log_cfs=log_cfs,
            composite_score=self.alpha * 0.0 + (1 - self.alpha) * log_cfs,
            mps=mps,
            encoder_outputs=encoder_outputs,
            is_finished=False,
        )

        return [initial_beam]

    def _fgbs_step(
        self,
        beams: List[EncoderDecoderBeamHypothesis],
        min_length: int,
        current_step: int,
        encoder_attention_mask: torch.Tensor,
    ) -> List[EncoderDecoderBeamHypothesis]:
        """
        Single FGBS step for encoder-decoder model.

        Args:
            beams: Current beam hypotheses
            min_length: Minimum sequence length before allowing EOS
            current_step: Current generation step
            encoder_attention_mask: Attention mask for encoder outputs

        Returns:
            Updated list of top-B beams
        """
        all_candidates: List[EncoderDecoderBeamHypothesis] = []

        # Get logits for all active beams
        active_indices = [idx for idx, beam in enumerate(beams) if not beam.is_finished]
        logits_per_beam: Dict[int, torch.Tensor] = {}

        if active_indices:
            # Batch decoder forward passes
            batch_size = len(active_indices)
            lengths = [len(beams[idx].decoder_token_ids) for idx in active_indices]
            max_len = max(lengths)

            decoder_input_ids = torch.full(
                (batch_size, max_len),
                self.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            decoder_attention_mask = torch.zeros_like(decoder_input_ids)

            for row, idx in enumerate(active_indices):
                tokens = beams[idx].decoder_token_ids
                length = len(tokens)
                decoder_input_ids[row, :length] = torch.tensor(
                    tokens, device=self.device
                )
                decoder_attention_mask[row, :length] = 1

            # All beams share the same encoder_outputs
            encoder_outputs = beams[0].encoder_outputs

            # Expand encoder outputs to match batch size
            batch_encoder_hidden_states = encoder_outputs.last_hidden_state.expand(
                batch_size, -1, -1
            )
            batch_encoder_attention_mask = encoder_attention_mask.expand(batch_size, -1)

            with torch.no_grad():
                # Decoder forward pass with cross-attention to encoder
                outputs = self.model(
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    encoder_outputs=(batch_encoder_hidden_states,),
                    attention_mask=batch_encoder_attention_mask,
                )
                logits = outputs.logits

            for row, idx in enumerate(active_indices):
                length = len(beams[idx].decoder_token_ids)
                logits_per_beam[idx] = logits[row, length - 1, :].detach()

            # Clear intermediate tensors
            del outputs
            del logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Pre-compute token embeddings
        unique_token_ids = set()
        beam_candidates = {}

        for idx, beam in enumerate(beams):
            if beam.is_finished:
                all_candidates.append(beam)
                continue

            beam_logits = logits_per_beam.get(idx)
            if beam_logits is None:
                all_candidates.append(beam)
                continue

            # Apply temperature and get top-k
            logits_adjusted = (
                beam_logits / self.temperature
                if self.temperature != 1.0
                else beam_logits
            )
            log_probs = torch.log_softmax(logits_adjusted, dim=-1)

            current_top_k = min(self.top_k, log_probs.shape[-1])
            topk_log_probs, topk_token_ids = torch.topk(log_probs, current_top_k)

            beam_candidates[idx] = (beam, topk_log_probs, topk_token_ids)
            for token_id_tensor in topk_token_ids:
                unique_token_ids.add(int(token_id_tensor.item()))

        # Batch compute embeddings
        token_embeddings_cache = {}
        if unique_token_ids:
            unique_tokens_list = list(unique_token_ids)
            with torch.no_grad():
                tokens_tensor = torch.tensor(unique_tokens_list, device=self.device)
                embeddings_batch = self.decoder_embedding_layer(tokens_tensor).detach()
                token_embeddings_cache = dict(zip(unique_tokens_list, embeddings_batch))

        # Expand beams
        for idx, (beam, topk_log_probs, topk_token_ids) in beam_candidates.items():
            for log_prob_tensor, token_id_tensor in zip(topk_log_probs, topk_token_ids):
                token_id = int(token_id_tensor.item())
                log_prob = float(log_prob_tensor.item())

                is_eos = token_id == self.eos_token_id
                if is_eos and len(beam.decoder_token_ids) < min_length:
                    continue

                new_decoder_token_ids = beam.decoder_token_ids + [token_id]
                new_log_prob = beam.log_prob + log_prob

                # Update MPS
                new_mps = beam.mps.copy()
                token_embedding = token_embeddings_cache[token_id]
                new_mps.add_token(token_embedding)

                new_cfs = compute_cfs_from_mps(new_mps)
                new_log_cfs = math.log(max(new_cfs, 1e-12))

                composite_score = (
                    self.alpha * new_log_prob + (1.0 - self.alpha) * new_log_cfs
                )

                candidate = EncoderDecoderBeamHypothesis(
                    decoder_token_ids=new_decoder_token_ids,
                    log_prob=new_log_prob,
                    log_cfs=new_log_cfs,
                    composite_score=composite_score,
                    mps=new_mps,
                    encoder_outputs=beam.encoder_outputs,  # Share encoder outputs
                    is_finished=is_eos,
                )
                all_candidates.append(candidate)

        if not all_candidates:
            return beams

        all_candidates.sort(key=lambda x: x.composite_score, reverse=True)
        top_beams = all_candidates[: self.beam_width]

        # Memory cleanup
        del all_candidates[self.beam_width :]
        del beams

        return top_beams

    def compare_with_baseline(
        self, input_text: str, max_length: int = 100
    ) -> Dict[str, Any]:
        """
        Generate with both FGBS and standard beam search for comparison.

        Args:
            input_text: Input text
            max_length: Maximum generation length

        Returns:
            Dictionary with both results:
                - 'fgbs': FGBS generation result
                - 'baseline': Standard beam search result
                - 'cfs_comparison': CFS values for both
        """
        logger.info("Running ED-FGBS generation...")
        fgbs_result = self.generate(
            input_text, max_length=max_length, return_details=True, show_progress=False
        )

        logger.info("Running baseline (standard beam search)...")
        original_alpha = self.alpha
        self.alpha = 1.0

        baseline_result = self.generate(
            input_text, max_length=max_length, return_details=True, show_progress=False
        )

        self.alpha = original_alpha

        return {
            "fgbs": fgbs_result,
            "baseline": baseline_result,
            "cfs_comparison": {
                "fgbs_final_cfs": math.exp(fgbs_result["log_cfs"]),
                "baseline_final_cfs": math.exp(baseline_result["log_cfs"]),
                "cfs_improvement": (
                    math.exp(fgbs_result["log_cfs"])
                    - math.exp(baseline_result["log_cfs"])
                ),
            },
        }

    def __repr__(self) -> str:
        return (
            f"EncoderDecoderFGBS("
            f"model={self.model.__class__.__name__}, "
            f"B={self.beam_width}, α={self.alpha}, χ={self.bond_dim})"
        )


__all__ = ["EncoderDecoderFGBS", "EncoderDecoderBeamHypothesis"]
