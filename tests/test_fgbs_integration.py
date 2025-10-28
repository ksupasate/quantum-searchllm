"""
Integration Tests for FidelityGuidedBeamSearcher

Tests the end-to-end FGBS pipeline with small models.
These tests are marked as slow and can be skipped with: pytest -m "not slow"
"""

import pytest
import torch

# Skip if transformers not available
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from tnad.fgbs_searcher import FidelityGuidedBeamSearcher, BeamHypothesis
from tnad.mps_manager import MPSSequence


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestFGBSIntegration:
    """Integration tests for FGBS with real models."""

    @pytest.fixture(scope="class")
    def tiny_model_and_tokenizer(self):
        """Load a tiny GPT-2 model for testing (124M params)."""
        try:
            model_name = "gpt2"  # Smallest available model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            # Set padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            return model, tokenizer
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")

    @pytest.mark.slow
    def test_searcher_initialization(self, tiny_model_and_tokenizer):
        """Test FGBS searcher initialization."""
        model, tokenizer = tiny_model_and_tokenizer

        searcher = FidelityGuidedBeamSearcher(
            model=model,
            tokenizer=tokenizer,
            beam_width=3,
            alpha=0.5,
            bond_dim=8,
        )

        assert searcher.beam_width == 3
        assert searcher.alpha == 0.5
        assert searcher.bond_dim == 8
        assert searcher.embedding_dim == 768  # GPT-2 embedding dim

    @pytest.mark.slow
    def test_basic_generation(self, tiny_model_and_tokenizer):
        """Test basic text generation with FGBS."""
        model, tokenizer = tiny_model_and_tokenizer

        searcher = FidelityGuidedBeamSearcher(
            model=model,
            tokenizer=tokenizer,
            beam_width=3,
            alpha=0.5,
            bond_dim=8,
        )

        # Simple prompt
        prompt = "The answer is"
        result = searcher.generate(
            prompt,
            max_length=20,
            min_length=5,
            show_progress=False,
        )

        # Check result structure
        assert 'text' in result
        assert 'token_ids' in result
        assert 'log_prob' in result
        assert 'log_cfs' in result
        assert 'composite_score' in result

        # Check generated text
        assert isinstance(result['text'], str)
        assert len(result['text']) > len(prompt)
        assert len(result['token_ids']) > len(tokenizer.encode(prompt))

    @pytest.mark.slow
    def test_generation_with_details(self, tiny_model_and_tokenizer):
        """Test generation with detailed output."""
        model, tokenizer = tiny_model_and_tokenizer

        searcher = FidelityGuidedBeamSearcher(
            model=model,
            tokenizer=tokenizer,
            beam_width=2,
            alpha=0.5,
            bond_dim=8,
        )

        result = searcher.generate(
            "Hello",
            max_length=15,
            return_details=True,
            show_progress=False,
        )

        # Check detailed outputs
        assert 'all_beams' in result
        assert 'cfs_trajectory' in result
        assert 'score_trajectory' in result

        # Check trajectories
        assert len(result['cfs_trajectory']) > 0
        assert len(result['score_trajectory']) > 0

        # All CFS values should be >= 1
        assert all(cfs >= 1.0 for cfs in result['cfs_trajectory'])

    @pytest.mark.slow
    def test_different_alpha_values(self, tiny_model_and_tokenizer):
        """Test FGBS with different alpha values."""
        model, tokenizer = tiny_model_and_tokenizer

        prompt = "2 + 2 ="
        alphas = [0.3, 0.5, 0.7, 1.0]  # 1.0 = pure beam search

        results = []
        for alpha in alphas:
            searcher = FidelityGuidedBeamSearcher(
                model=model,
                tokenizer=tokenizer,
                beam_width=3,
                alpha=alpha,
                bond_dim=8,
            )

            result = searcher.generate(
                prompt,
                max_length=15,
                show_progress=False,
            )
            results.append(result)

        # All should produce valid results
        assert all('text' in r for r in results)

        # Results may differ (not guaranteed, but likely)
        texts = [r['text'] for r in results]
        # At least check they all contain the prompt
        assert all(prompt in t for t in texts)

    @pytest.mark.slow
    def test_compare_with_baseline(self, tiny_model_and_tokenizer):
        """Test comparison between FGBS and standard beam search."""
        model, tokenizer = tiny_model_and_tokenizer

        searcher = FidelityGuidedBeamSearcher(
            model=model,
            tokenizer=tokenizer,
            beam_width=3,
            alpha=0.5,
            bond_dim=8,
        )

        prompt = "The result is"
        comparison = searcher.compare_with_baseline(
            prompt,
            max_length=15,
        )

        # Check structure
        assert 'fgbs' in comparison
        assert 'baseline' in comparison
        assert 'cfs_comparison' in comparison

        # Check CFS comparison
        assert 'fgbs_final_cfs' in comparison['cfs_comparison']
        assert 'baseline_final_cfs' in comparison['cfs_comparison']
        assert 'cfs_improvement' in comparison['cfs_comparison']

        # Both should produce valid text
        assert len(comparison['fgbs']['text']) > 0
        assert len(comparison['baseline']['text']) > 0


class TestBeamHypothesis:
    """Test BeamHypothesis dataclass."""

    def test_hypothesis_creation(self):
        """Test creating a beam hypothesis."""
        mps = MPSSequence(bond_dim=8, embedding_dim=64)
        mps.add_token(torch.randn(64))

        hypothesis = BeamHypothesis(
            token_ids=[1, 2, 3],
            log_prob=-5.2,
            log_cfs=2.1,
            composite_score=-1.5,
            mps=mps,
            is_finished=False,
        )

        assert hypothesis.token_ids == [1, 2, 3]
        assert hypothesis.log_prob == -5.2
        assert hypothesis.log_cfs == 2.1
        assert hypothesis.composite_score == -1.5
        assert not hypothesis.is_finished

    def test_hypothesis_repr(self):
        """Test string representation."""
        mps = MPSSequence(bond_dim=8, embedding_dim=64)

        hypothesis = BeamHypothesis(
            token_ids=[1, 2],
            log_prob=-2.0,
            log_cfs=1.5,
            composite_score=-0.5,
            mps=mps,
        )

        repr_str = repr(hypothesis)
        assert "BeamHypothesis" in repr_str
        assert "len=2" in repr_str


class TestFGBSConfiguration:
    """Test FGBS configuration and edge cases."""

    def test_invalid_alpha(self):
        """Test that invalid alpha values are handled."""
        # This test doesn't require a real model
        # Just test configuration validation if implemented
        pass

    def test_invalid_beam_width(self):
        """Test that invalid beam width is handled."""
        pass

    def test_invalid_bond_dim(self):
        """Test that invalid bond dimension is handled."""
        pass


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestFGBSStressTests:
    """Stress tests for FGBS."""

    @pytest.fixture(scope="class")
    def tiny_model_and_tokenizer(self):
        """Load a tiny model for stress testing."""
        try:
            model_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            return model, tokenizer
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")

    @pytest.mark.slow
    @pytest.mark.stress
    def test_long_generation(self, tiny_model_and_tokenizer):
        """Test generation of longer sequences."""
        model, tokenizer = tiny_model_and_tokenizer

        searcher = FidelityGuidedBeamSearcher(
            model=model,
            tokenizer=tokenizer,
            beam_width=2,  # Smaller for speed
            alpha=0.5,
            bond_dim=8,
        )

        result = searcher.generate(
            "Once upon a time",
            max_length=50,  # Longer sequence
            show_progress=False,
        )

        # Should complete without errors
        assert len(result['token_ids']) > 10

    @pytest.mark.slow
    @pytest.mark.stress
    def test_multiple_generations(self, tiny_model_and_tokenizer):
        """Test multiple sequential generations."""
        model, tokenizer = tiny_model_and_tokenizer

        searcher = FidelityGuidedBeamSearcher(
            model=model,
            tokenizer=tokenizer,
            beam_width=3,
            alpha=0.5,
            bond_dim=8,
        )

        prompts = [
            "Hello",
            "The answer is",
            "In conclusion",
        ]

        for prompt in prompts:
            result = searcher.generate(
                prompt,
                max_length=15,
                show_progress=False,
            )
            assert len(result['text']) > 0


if __name__ == "__main__":
    # Run with: pytest test_fgbs_integration.py -v
    # Skip slow tests: pytest test_fgbs_integration.py -v -m "not slow"
    pytest.main([__file__, "-v"])
