"""
Fidelity-Guided Beam Search (FGBS)

Implements the core TNAD algorithm for logically coherent text generation.

Algorithm Overview:
    Standard beam search ranks candidates by: Score(S) = log P(S)
    FGBS uses composite scoring: Score(S) = α·log P(S) + (1-α)·log F(S)

    where:
    - P(S): LLM probability (fluency)
    - F(S): Coherence Fidelity Score (structural integrity)
    - α ∈ [0,1]: Balance parameter

Key Innovation:
    Real-time pruning of incoherent reasoning paths during generation,
    bridging local probabilistic decoding with global logical constraints.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

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

from tqdm import tqdm

from transformers import PreTrainedModel, PreTrainedTokenizer

from tnad.coherence_score import compute_cfs_from_mps
from tnad.mps_manager import MPSSequence
from tnad.utils import get_device


@dataclass
class BeamHypothesis:
    """
    Single hypothesis in beam search.

    Attributes:
        token_ids: Generated token sequence [L]
        log_prob: Cumulative log probability from LLM
        log_cfs: Cumulative log CFS (coherence score)
        composite_score: α·log_prob + (1-α)·log_cfs
        mps: MPS representation of the sequence
        is_finished: Whether sequence ends with EOS token
    """
    token_ids: List[int]
    log_prob: float
    log_cfs: float
    composite_score: float
    mps: MPSSequence
    is_finished: bool = False

    def __repr__(self) -> str:
        return (
            f"BeamHypothesis(len={len(self.token_ids)}, "
            f"score={self.composite_score:.4f}, "
            f"log_p={self.log_prob:.4f}, log_F={self.log_cfs:.4f})"
        )


class FidelityGuidedBeamSearcher:
    """
    FGBS implementation for LLM generation.

    Maintains B parallel hypotheses, each with:
    - Token sequence (standard beam search)
    - MPS representation (tensor network state)
    - Composite score (fluency + coherence)

    Example Usage:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        >>>
        >>> searcher = FidelityGuidedBeamSearcher(
        >>>     model=model,
        >>>     tokenizer=tokenizer,
        >>>     beam_width=5,
        >>>     alpha=0.5,
        >>>     bond_dim=16,
        >>> )
        >>>
        >>> prompt = "Solve: If x + 2 = 5, then x = ?"
        >>> result = searcher.generate(prompt, max_length=100)
        >>> print(result['text'])
    """

    def __init__(
        self,
        model: PreTrainedModel,
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
        Initialize FGBS searcher.

        Args:
            model: Pretrained LLM (e.g., Llama, Mistral)
            tokenizer: Corresponding tokenizer
            beam_width: Number of parallel beams B (typical: 5-10)
            alpha: Fluency vs coherence balance ∈ [0,1]
                   α=1: pure LLM (standard beam search)
                   α=0: pure coherence (may sacrifice fluency)
                   α=0.5: balanced (recommended)
            bond_dim: MPS bond dimension χ (typical: 8-32)
            top_k: Number of top tokens to consider per beam
            temperature: Sampling temperature (1.0 = no scaling)
            device: Compute device ('cpu', 'cuda', 'mps')
            normalize_embeddings: Normalize token embeddings in MPS

        Implementation Details:
            [Model Setup]
            - Model set to eval mode (no gradient computation)
            - Embedding layer extracted for MPS construction
            - Device placement handled automatically

            [Hyperparameter Guidelines]
            - Small α (0.3): Prioritize coherence, good for reasoning tasks
            - Large α (0.7): Prioritize fluency, good for creative writing
            - Small χ (8): Fast but limited logical tracking
            - Large χ (32): Slower but better coherence monitoring
        """
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.alpha = alpha
        self.bond_dim = bond_dim
        self.top_k = top_k
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings

        if device is None:
            self.device = get_device(prefer_gpu=True)
        else:
            self.device = torch.device(device)

        self._is_quantized = False
        try:
            self.model = self.model.to(self.device)
        except RuntimeError as exc:
            message = str(exc)
            if "8-bit" in message or "4-bit" in message:
                self._is_quantized = True
                embed_weight = self.model.get_input_embeddings().weight
                self.device = torch.device(embed_weight.device)
            else:
                raise
        self.model.eval()  # Inference mode

        # Extract embedding layer for MPS construction
        self.embedding_layer = self.model.get_input_embeddings()
        self.embedding_dim = self.embedding_layer.embedding_dim

        # Special tokens
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id

        logger.info(
            f"Initialized FGBS: B={beam_width}, α={alpha}, χ={bond_dim}, "
            f"d={self.embedding_dim}, device={self.device}"
        )

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        min_length: int = 10,
        return_details: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate text using FGBS algorithm.

        Algorithm (from paper):
            1. Initialize B beams with prompt tokens
            2. For each step t ∈ [1, max_length]:
                a. For each beam b ∈ [1, B]:
                   - Get next-token log probs from LLM
                   - Select top-K candidate tokens
                b. For each (beam, candidate) pair:
                   - Create hypothetical new MPS by adding token
                   - Compute CFS from Schmidt values
                   - Calculate composite score
                c. Select top-B hypotheses by composite score
                d. Check termination (all beams finished or max_length)
            3. Return best hypothesis

        Args:
            prompt: Input text prompt
            max_length: Maximum generation length (tokens)
            min_length: Minimum generation length before allowing EOS
            return_details: If True, return detailed generation info
            show_progress: Show progress bar

        Returns:
            Dictionary containing:
                - 'text': Generated text (str)
                - 'token_ids': Token IDs (List[int])
                - 'log_prob': Final log probability (float)
                - 'log_cfs': Final log CFS (float)
                - 'composite_score': Final composite score (float)
                If return_details=True, also includes:
                - 'all_beams': All beam hypotheses at final step
                - 'cfs_trajectory': CFS values over generation
                - 'score_trajectory': Composite scores over generation

        Example:
            >>> result = searcher.generate(
            >>>     "What is 2+2?",
            >>>     max_length=50,
            >>>     return_details=True
            >>> )
            >>> print(result['text'])
            >>> print(f"Final CFS: {result['log_cfs']:.2f}")
        """
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        logger.info(f"Starting FGBS generation: prompt_length={len(prompt_ids)}")

        # Initialize beams with prompt
        beams = self._initialize_beams(prompt_ids)

        # Track trajectories for analysis
        cfs_trajectory = []
        score_trajectory = []

        # Generation loop
        progress_bar = tqdm(
            range(max_length),
            desc="Generating",
            disable=not show_progress
        )

        for step in progress_bar:
            # Check if all beams are finished
            if all(beam.is_finished for beam in beams):
                logger.info(f"All beams finished at step {step}")
                break

            # FGBS step: expand and rank
            beams = self._fgbs_step(beams, min_length, step)

            # Track best beam
            best_beam = max(beams, key=lambda b: b.composite_score)
            cfs_trajectory.append(math.exp(best_beam.log_cfs))
            score_trajectory.append(best_beam.composite_score)

            # Update progress bar
            progress_bar.set_postfix({
                'length': len(best_beam.token_ids),
                'score': f"{best_beam.composite_score:.2f}",
                'cfs': f"{math.exp(best_beam.log_cfs):.2f}",
            })

        # Select best beam
        best_beam = max(beams, key=lambda b: b.composite_score)

        # Decode tokens
        generated_text = self.tokenizer.decode(
            best_beam.token_ids,
            skip_special_tokens=True
        )

        # Prepare result
        result = {
            'text': generated_text,
            'token_ids': best_beam.token_ids,
            'log_prob': best_beam.log_prob,
            'log_cfs': best_beam.log_cfs,
            'composite_score': best_beam.composite_score,
        }

        if return_details:
            result.update({
                'all_beams': beams,
                'cfs_trajectory': cfs_trajectory,
                'score_trajectory': score_trajectory,
            })

        logger.info(
            f"Generation complete: length={len(best_beam.token_ids)}, "
            f"score={best_beam.composite_score:.4f}"
        )

        return result

    def _initialize_beams(self, prompt_ids: List[int]) -> List[BeamHypothesis]:
        """
        Initialize beam search with prompt tokens.

        Creates a single beam containing the prompt, with MPS state
        constructed from prompt token embeddings.

        Args:
            prompt_ids: Tokenized prompt

        Returns:
            List containing single initial beam
        """
        # Create MPS for prompt
        mps = MPSSequence(
            bond_dim=self.bond_dim,
            embedding_dim=self.embedding_dim,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        )

        # Add prompt tokens to MPS
        with torch.no_grad():
            prompt_tensor = torch.tensor(prompt_ids, device=self.device)
            prompt_embeddings = self.embedding_layer(prompt_tensor)

            for embedding in prompt_embeddings:
                mps.add_token(embedding)

        # Compute initial CFS
        initial_cfs = compute_cfs_from_mps(mps)
        log_cfs = math.log(max(initial_cfs, 1e-12))

        # Create initial beam
        initial_beam = BeamHypothesis(
            token_ids=prompt_ids.copy(),
            log_prob=0.0,  # No generation yet
            log_cfs=log_cfs,
            composite_score=self.alpha * 0.0 + (1 - self.alpha) * log_cfs,
            mps=mps,
            is_finished=False,
        )

        return [initial_beam]

    def _fgbs_step(
        self,
        beams: List[BeamHypothesis],
        min_length: int,
        current_step: int,
    ) -> List[BeamHypothesis]:
        """
        Single FGBS step: expand beams and select top-B by composite score.

        Args:
            beams: Current beam hypotheses
            min_length: Minimum sequence length before allowing EOS
            current_step: Current generation step

        Returns:
            Updated list of top-B beams
        """
        all_candidates: List[BeamHypothesis] = []

        active_indices = [idx for idx, beam in enumerate(beams) if not beam.is_finished]
        logits_per_beam: Dict[int, torch.Tensor] = {}

        if active_indices:
            lengths = [len(beams[idx].token_ids) for idx in active_indices]
            max_len = max(lengths)

            batch_size = len(active_indices)
            input_ids = torch.full(
                (batch_size, max_len),
                self.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)

            for row, idx in enumerate(active_indices):
                tokens = beams[idx].token_ids
                length = len(tokens)
                input_ids[row, :length] = torch.tensor(tokens, device=self.device)
                attention_mask[row, :length] = 1

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            for row, idx in enumerate(active_indices):
                length = len(beams[idx].token_ids)
                logits_per_beam[idx] = logits[row, length - 1, :]

        for idx, beam in enumerate(beams):
            if beam.is_finished:
                all_candidates.append(beam)
                continue

            beam_logits = logits_per_beam.get(idx)
            if beam_logits is None:
                all_candidates.append(beam)
                continue

            logits_adjusted = beam_logits / self.temperature if self.temperature != 1.0 else beam_logits
            log_probs = torch.log_softmax(logits_adjusted, dim=-1)

            current_top_k = min(self.top_k, log_probs.shape[-1])
            topk_log_probs, topk_token_ids = torch.topk(log_probs, current_top_k)

            for log_prob_tensor, token_id_tensor in zip(topk_log_probs, topk_token_ids):
                token_id = int(token_id_tensor.item())
                log_prob = float(log_prob_tensor.item())

                is_eos = token_id == self.eos_token_id
                if is_eos and len(beam.token_ids) < min_length:
                    continue

                new_token_ids = beam.token_ids + [token_id]
                new_log_prob = beam.log_prob + log_prob

                new_mps = beam.mps.copy()
                with torch.no_grad():
                    token_embedding = self.embedding_layer(
                        torch.tensor([token_id], device=self.device)
                    )[0]
                    new_mps.add_token(token_embedding)

                new_cfs = compute_cfs_from_mps(new_mps)
                new_log_cfs = math.log(max(new_cfs, 1e-12))

                composite_score = (
                    self.alpha * new_log_prob + (1.0 - self.alpha) * new_log_cfs
                )

                candidate = BeamHypothesis(
                    token_ids=new_token_ids,
                    log_prob=new_log_prob,
                    log_cfs=new_log_cfs,
                    composite_score=composite_score,
                    mps=new_mps,
                    is_finished=is_eos,
                )
                all_candidates.append(candidate)

        if not all_candidates:
            return beams

        all_candidates.sort(key=lambda x: x.composite_score, reverse=True)
        top_beams = all_candidates[: self.beam_width]

        return top_beams

    def compare_with_baseline(
        self,
        prompt: str,
        max_length: int = 100,
    ) -> Dict[str, Any]:
        """
        Generate with both FGBS and standard beam search for comparison.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length

        Returns:
            Dictionary with both results:
                - 'fgbs': FGBS generation result
                - 'baseline': Standard beam search result
                - 'cfs_comparison': CFS values for both
        """
        logger.info("Running FGBS generation...")
        fgbs_result = self.generate(
            prompt,
            max_length=max_length,
            return_details=True,
            show_progress=False,
        )

        logger.info("Running baseline (standard beam search)...")
        # Temporarily set alpha=1 (pure LLM probability)
        original_alpha = self.alpha
        self.alpha = 1.0

        baseline_result = self.generate(
            prompt,
            max_length=max_length,
            return_details=True,
            show_progress=False,
        )

        # Restore alpha
        self.alpha = original_alpha

        return {
            'fgbs': fgbs_result,
            'baseline': baseline_result,
            'cfs_comparison': {
                'fgbs_final_cfs': math.exp(fgbs_result['log_cfs']),
                'baseline_final_cfs': math.exp(baseline_result['log_cfs']),
                'cfs_improvement': (
                    math.exp(fgbs_result['log_cfs']) -
                    math.exp(baseline_result['log_cfs'])
                ),
            },
        }

    def __repr__(self) -> str:
        return (
            f"FidelityGuidedBeamSearcher("
            f"model={self.model.__class__.__name__}, "
            f"B={self.beam_width}, α={self.alpha}, χ={self.bond_dim})"
        )
