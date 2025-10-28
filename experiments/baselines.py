"""
Baseline Methods for Comparison

Implements:
- Greedy Decoding
- Standard Beam Search
- Self-Consistency Sampling

These baselines are used for comparison with FGBS in experiments.
"""

import sys
from pathlib import Path

# Add parent directory to path to import tnad
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm


class GreedyDecoder:
    """
    Greedy decoding baseline (argmax at each step).

    Equivalent to FGBS with α=1.0, B=1.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if device is not None:
            self.device = torch.device(device)
        else:
            if hasattr(model, "device") and isinstance(model.device, torch.device):
                self.device = model.device
            else:
                self.device = model.get_input_embeddings().weight.device
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        min_length: int = 10,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        """Generate using greedy decoding."""
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Greedy generation
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_length,
                min_length=input_ids.shape[1] + min_length,
                do_sample=False,  # Greedy
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            'text': generated_text,
            'token_ids': output_ids[0].tolist(),
            'method': 'greedy',
        }


class StandardBeamSearch:
    """
    Standard beam search baseline (LLM probability only).

    Equivalent to FGBS with α=1.0 (no coherence score).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        beam_width: int = 5,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        if device is not None:
            self.device = torch.device(device)
        else:
            if hasattr(model, "device") and isinstance(model.device, torch.device):
                self.device = model.device
            else:
                self.device = model.get_input_embeddings().weight.device
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        min_length: int = 10,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        """Generate using standard beam search."""
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Beam search generation
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_length,
                min_length=input_ids.shape[1] + min_length,
                do_sample=False,
                num_beams=self.beam_width,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            'text': generated_text,
            'token_ids': output_ids[0].tolist(),
            'method': f'beam_search_B{self.beam_width}',
        }


class SelfConsistency:
    """
    Self-Consistency baseline.

    Samples N diverse reasoning paths and takes majority vote on final answer.
    From Wang et al. (2022): "Self-Consistency Improves Chain of Thought Reasoning"
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_samples: int = 10,
        temperature: float = 0.7,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.temperature = temperature
        if device is not None:
            self.device = torch.device(device)
        else:
            if hasattr(model, "device") and isinstance(model.device, torch.device):
                self.device = model.device
            else:
                self.device = model.get_input_embeddings().weight.device
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        min_length: int = 10,
        show_progress: bool = False,
        answer_extractor = None,  # Function to extract answer from text
    ) -> Dict[str, Any]:
        """
        Generate using self-consistency.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            min_length: Minimum generation length
            show_progress: Show progress bar
            answer_extractor: Function to extract answer from generated text
                             If None, returns all samples

        Returns:
            Dictionary with majority answer and all samples
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate N diverse samples
        samples = []

        iterator = range(self.num_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Self-Consistency Sampling")

        for _ in iterator:
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_length,
                    min_length=input_ids.shape[1] + min_length,
                    do_sample=True,  # Sampling for diversity
                    temperature=self.temperature,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            samples.append(text)

        # Extract answers if extractor provided
        if answer_extractor is not None:
            answers = [answer_extractor(sample) for sample in samples]

            # Majority vote
            from collections import Counter
            answer_counts = Counter(answers)
            majority_answer = answer_counts.most_common(1)[0][0]
            confidence = answer_counts[majority_answer] / len(answers)

            return {
                'text': samples[0],  # Return first sample as representative
                'all_samples': samples,
                'all_answers': answers,
                'majority_answer': majority_answer,
                'confidence': confidence,
                'answer_counts': dict(answer_counts),
                'method': f'self_consistency_N{self.num_samples}',
            }
        else:
            # No answer extraction, just return samples
            return {
                'text': samples[0],
                'all_samples': samples,
                'method': f'self_consistency_N{self.num_samples}',
            }


def run_baseline_comparison(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 100,
    beam_width: int = 5,
    sc_num_samples: int = 10,
    answer_extractor = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run all baselines on a single prompt for comparison.

    Args:
        prompt: Input prompt
        model: LLM model
        tokenizer: Tokenizer
        max_length: Maximum generation length
        beam_width: Beam width for beam search
        sc_num_samples: Number of samples for self-consistency
        answer_extractor: Function to extract answer (for self-consistency)

    Returns:
        Dictionary mapping method name to results
    """
    device = model.device

    results = {}

    # Greedy
    greedy = GreedyDecoder(model, tokenizer, device)
    results['greedy'] = greedy.generate(prompt, max_length=max_length)

    # Standard Beam Search
    beam_search = StandardBeamSearch(model, tokenizer, beam_width, device)
    results['beam_search'] = beam_search.generate(prompt, max_length=max_length)

    # Self-Consistency
    self_consistency = SelfConsistency(
        model, tokenizer, sc_num_samples, device=device
    )
    results['self_consistency'] = self_consistency.generate(
        prompt,
        max_length=max_length,
        answer_extractor=answer_extractor,
    )

    return results
