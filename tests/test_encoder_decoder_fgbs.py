"""
Tests for Encoder-Decoder FGBS (T5, BART support)
"""

import pytest
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)

from tnad.encoder_decoder_fgbs import EncoderDecoderFGBS, EncoderDecoderBeamHypothesis


@pytest.fixture
def t5_model_and_tokenizer():
    """Load T5-small for testing."""
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return model, tokenizer


@pytest.fixture
def bart_model_and_tokenizer():
    """Load BART-base for testing (CPU only for CI)."""
    try:
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        return model, tokenizer
    except Exception:
        pytest.skip("BART model not available")


class TestEncoderDecoderFGBS:
    """Test suite for EncoderDecoderFGBS."""

    def test_initialization_t5(self, t5_model_and_tokenizer):
        """Test initialization with T5 model."""
        model, tokenizer = t5_model_and_tokenizer

        searcher = EncoderDecoderFGBS(
            model=model,
            tokenizer=tokenizer,
            beam_width=3,
            alpha=0.5,
            bond_dim=8,
            device="cpu",
        )

        assert searcher.beam_width == 3
        assert searcher.alpha == 0.5
        assert searcher.bond_dim == 8
        assert searcher.device.type == "cpu"

    def test_invalid_model_type(self):
        """Test that non-encoder-decoder models are rejected."""
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        with pytest.raises(TypeError, match="encoder-decoder"):
            EncoderDecoderFGBS(
                model=model, tokenizer=tokenizer, device="cpu"
            )

    def test_generation_t5(self, t5_model_and_tokenizer):
        """Test T5 generation with FGBS."""
        model, tokenizer = t5_model_and_tokenizer

        searcher = EncoderDecoderFGBS(
            model=model,
            tokenizer=tokenizer,
            beam_width=3,
            alpha=0.5,
            bond_dim=8,
            device="cpu",
        )

        # T5 requires task prefix
        prompt = "translate English to French: Hello, how are you?"
        result = searcher.generate(prompt, max_length=20, show_progress=False)

        assert "text" in result
        assert "token_ids" in result
        assert "log_prob" in result
        assert "log_cfs" in result
        assert len(result["text"]) > 0
        assert len(result["token_ids"]) > 0

    def test_generation_with_details(self, t5_model_and_tokenizer):
        """Test generation with detailed metrics."""
        model, tokenizer = t5_model_and_tokenizer

        searcher = EncoderDecoderFGBS(
            model=model,
            tokenizer=tokenizer,
            beam_width=2,
            alpha=0.5,
            bond_dim=8,
            device="cpu",
        )

        prompt = "summarize: The quick brown fox jumps over the lazy dog."
        result = searcher.generate(
            prompt, max_length=15, return_details=True, show_progress=False
        )

        assert "cfs_trajectory" in result
        assert "score_trajectory" in result
        assert len(result["cfs_trajectory"]) > 0
        assert len(result["score_trajectory"]) > 0

    def test_alpha_parameter_effect(self, t5_model_and_tokenizer):
        """Test that alpha affects generation."""
        model, tokenizer = t5_model_and_tokenizer
        prompt = "translate English to German: Hello world"

        # Pure LLM (alpha=1.0)
        searcher_llm = EncoderDecoderFGBS(
            model=model, tokenizer=tokenizer, alpha=1.0, beam_width=2, device="cpu"
        )
        result_llm = searcher_llm.generate(prompt, max_length=10, show_progress=False)

        # Balanced (alpha=0.5)
        searcher_balanced = EncoderDecoderFGBS(
            model=model, tokenizer=tokenizer, alpha=0.5, beam_width=2, device="cpu"
        )
        result_balanced = searcher_balanced.generate(
            prompt, max_length=10, show_progress=False
        )

        # Results should differ (though not always guaranteed with small models)
        # At minimum, scores should be computed differently
        assert result_llm["composite_score"] != result_balanced["composite_score"]

    def test_compare_with_baseline(self, t5_model_and_tokenizer):
        """Test baseline comparison."""
        model, tokenizer = t5_model_and_tokenizer

        searcher = EncoderDecoderFGBS(
            model=model,
            tokenizer=tokenizer,
            beam_width=2,
            alpha=0.5,
            bond_dim=8,
            device="cpu",
        )

        prompt = "summarize: This is a test."
        comparison = searcher.compare_with_baseline(prompt, max_length=15)

        assert "fgbs" in comparison
        assert "baseline" in comparison
        assert "cfs_comparison" in comparison
        assert "fgbs_final_cfs" in comparison["cfs_comparison"]
        assert "baseline_final_cfs" in comparison["cfs_comparison"]

    def test_input_validation(self, t5_model_and_tokenizer):
        """Test input validation."""
        model, tokenizer = t5_model_and_tokenizer

        searcher = EncoderDecoderFGBS(
            model=model, tokenizer=tokenizer, device="cpu"
        )

        # Empty prompt
        with pytest.raises(ValueError, match="empty"):
            searcher.generate("", max_length=10)

        # Invalid max_length
        with pytest.raises(ValueError, match="max_length"):
            searcher.generate("test", max_length=0)

        # Invalid min_length
        with pytest.raises(ValueError, match="min_length"):
            searcher.generate("test", max_length=10, min_length=20)

    def test_encoder_decoder_beam_hypothesis(self, t5_model_and_tokenizer):
        """Test EncoderDecoderBeamHypothesis dataclass."""
        from tnad.mps_manager import MPSSequence

        mps = MPSSequence(bond_dim=8, embedding_dim=512, device="cpu")

        hypothesis = EncoderDecoderBeamHypothesis(
            decoder_token_ids=[1, 2, 3],
            log_prob=-1.5,
            log_cfs=-0.5,
            composite_score=-1.0,
            mps=mps,
            encoder_outputs=None,
            is_finished=False,
        )

        assert len(hypothesis.decoder_token_ids) == 3
        assert hypothesis.log_prob == -1.5
        assert hypothesis.is_finished is False
        assert "EDBeamHypothesis" in str(hypothesis)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
class TestEncoderDecoderFGBSGPU:
    """GPU-specific tests."""

    def test_generation_on_gpu(self, t5_model_and_tokenizer):
        """Test generation on GPU."""
        model, tokenizer = t5_model_and_tokenizer

        searcher = EncoderDecoderFGBS(
            model=model,
            tokenizer=tokenizer,
            beam_width=3,
            alpha=0.5,
            bond_dim=8,
            device="cuda",
        )

        prompt = "translate English to French: Hello"
        result = searcher.generate(prompt, max_length=10, show_progress=False)

        assert len(result["text"]) > 0
        assert searcher.device.type == "cuda"
