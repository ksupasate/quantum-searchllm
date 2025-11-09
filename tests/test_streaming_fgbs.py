"""
Tests for Streaming FGBS
"""

import asyncio
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tnad.streaming_fgbs import (
    StreamingFGBS,
    BufferedStreamingFGBS,
    StreamingToken,
)


@pytest.fixture
def model_and_tokenizer():
    """Load small model for testing."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class TestStreamingFGBS:
    """Test suite for StreamingFGBS."""

    def test_initialization(self, model_and_tokenizer):
        """Test streaming FGBS initialization."""
        model, tokenizer = model_and_tokenizer

        searcher = StreamingFGBS(
            model=model,
            tokenizer=tokenizer,
            beam_width=3,
            alpha=0.5,
            bond_dim=8,
            device="cpu",
        )

        assert searcher.beam_width == 3
        assert searcher.alpha == 0.5

    def test_stream_generation(self, model_and_tokenizer):
        """Test basic streaming generation."""
        model, tokenizer = model_and_tokenizer

        searcher = StreamingFGBS(
            model=model,
            tokenizer=tokenizer,
            beam_width=2,
            alpha=0.5,
            bond_dim=8,
            device="cpu",
        )

        prompt = "Hello"
        tokens = []

        for token_info in searcher.generate_stream(prompt, max_length=10):
            assert isinstance(token_info, StreamingToken)
            assert hasattr(token_info, "token")
            assert hasattr(token_info, "log_cfs")
            assert hasattr(token_info, "position")
            tokens.append(token_info)

            # Break early to save time
            if len(tokens) >= 5:
                break

        assert len(tokens) > 0

    def test_streaming_token_attributes(self, model_and_tokenizer):
        """Test StreamingToken dataclass attributes."""
        model, tokenizer = model_and_tokenizer

        searcher = StreamingFGBS(
            model=model, tokenizer=tokenizer, beam_width=2, device="cpu"
        )

        prompt = "Test"
        for i, token_info in enumerate(searcher.generate_stream(prompt, max_length=5)):
            assert token_info.position == len(tokenizer.encode(prompt)) + i
            assert token_info.token_id >= 0
            assert isinstance(token_info.log_prob, float)
            assert isinstance(token_info.log_cfs, float)
            assert isinstance(token_info.composite_score, float)
            assert token_info.timestamp is not None

            if i >= 2:
                break

    def test_coherence_threshold_early_stopping(self, model_and_tokenizer):
        """Test early stopping based on coherence threshold."""
        model, tokenizer = model_and_tokenizer

        searcher = StreamingFGBS(
            model=model, tokenizer=tokenizer, beam_width=2, device="cpu"
        )

        prompt = "Hello world"
        tokens = []

        # Set very high threshold to trigger early stop
        for token_info in searcher.generate_stream(
            prompt, max_length=20, coherence_threshold=100.0  # Unreachable
        ):
            tokens.append(token_info)

        # Should stop early due to threshold
        assert len(tokens) < 20

    def test_yield_prompt_tokens(self, model_and_tokenizer):
        """Test yielding prompt tokens."""
        model, tokenizer = model_and_tokenizer

        searcher = StreamingFGBS(
            model=model, tokenizer=tokenizer, beam_width=2, device="cpu"
        )

        prompt = "Hello"
        prompt_length = len(tokenizer.encode(prompt))

        tokens = list(
            searcher.generate_stream(prompt, max_length=3, yield_prompt=True)
        )

        # Should include prompt tokens
        assert len(tokens) >= prompt_length

    def test_generate_with_callbacks(self, model_and_tokenizer):
        """Test callback-based generation."""
        model, tokenizer = model_and_tokenizer

        searcher = StreamingFGBS(
            model=model, tokenizer=tokenizer, beam_width=2, device="cpu"
        )

        # Track callbacks
        token_count = 0
        coherence_drops = 0

        def on_token(token):
            nonlocal token_count
            token_count += 1

        def on_coherence_drop(cfs, threshold):
            nonlocal coherence_drops
            coherence_drops += 1

        result = searcher.generate_with_callbacks(
            prompt="Hello",
            max_length=5,
            on_token=on_token,
            on_coherence_drop=on_coherence_drop,
            coherence_threshold=0.5,
        )

        assert token_count > 0
        assert "text" in result
        assert "num_tokens" in result
        assert result["num_tokens"] == token_count


class TestBufferedStreamingFGBS:
    """Test suite for BufferedStreamingFGBS."""

    def test_buffered_streaming(self, model_and_tokenizer):
        """Test buffered streaming with batch yields."""
        model, tokenizer = model_and_tokenizer

        buffer_size = 3
        searcher = BufferedStreamingFGBS(
            model=model,
            tokenizer=tokenizer,
            beam_width=2,
            buffer_size=buffer_size,
            device="cpu",
        )

        prompt = "Hello"
        batches = []

        for batch in searcher.generate_stream(prompt, max_length=10):
            assert isinstance(batch, list)
            assert len(batch) <= buffer_size
            batches.append(batch)

            # Collect a few batches
            if len(batches) >= 2:
                break

        assert len(batches) > 0


@pytest.mark.asyncio
class TestAsyncStreaming:
    """Test async streaming functionality."""

    async def test_async_stream_generation(self, model_and_tokenizer):
        """Test async streaming generation."""
        model, tokenizer = model_and_tokenizer

        searcher = StreamingFGBS(
            model=model,
            tokenizer=tokenizer,
            beam_width=2,
            alpha=0.5,
            bond_dim=8,
            device="cpu",
        )

        prompt = "Hello"
        tokens = []

        async for token_info in searcher.generate_stream_async(
            prompt, max_length=5
        ):
            assert isinstance(token_info, StreamingToken)
            tokens.append(token_info)

        assert len(tokens) > 0

    async def test_async_with_coherence_threshold(self, model_and_tokenizer):
        """Test async streaming with coherence threshold."""
        model, tokenizer = model_and_tokenizer

        searcher = StreamingFGBS(
            model=model, tokenizer=tokenizer, beam_width=2, device="cpu"
        )

        prompt = "Test"
        tokens = []

        async for token_info in searcher.generate_stream_async(
            prompt, max_length=10, coherence_threshold=100.0
        ):
            tokens.append(token_info)

        # Should stop early
        assert len(tokens) < 10


class TestSSEAndWebSocket:
    """Test SSE and WebSocket utilities."""

    def test_sse_stream_format(self, model_and_tokenizer):
        """Test SSE stream formatting."""
        from tnad.streaming_fgbs import create_sse_stream

        model, tokenizer = model_and_tokenizer

        searcher = StreamingFGBS(
            model=model, tokenizer=tokenizer, beam_width=2, device="cpu"
        )

        prompt = "Hello"
        sse_messages = []

        for message in create_sse_stream(searcher, prompt, max_length=5):
            assert isinstance(message, str)
            assert message.startswith("data:") or message.startswith("event:")
            sse_messages.append(message)

            if len(sse_messages) >= 3:
                break

        assert len(sse_messages) > 0

    @pytest.mark.asyncio
    async def test_websocket_stream(self, model_and_tokenizer):
        """Test WebSocket streaming (mock)."""
        from tnad.streaming_fgbs import create_websocket_stream

        model, tokenizer = model_and_tokenizer

        searcher = StreamingFGBS(
            model=model, tokenizer=tokenizer, beam_width=2, device="cpu"
        )

        # Mock WebSocket
        class MockWebSocket:
            def __init__(self):
                self.messages = []

            async def send_text(self, data):
                self.messages.append(data)

        websocket = MockWebSocket()
        prompt = "Hello"

        await create_websocket_stream(searcher, websocket, prompt, max_length=5)

        assert len(websocket.messages) > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
class TestStreamingFGBSGPU:
    """GPU-specific streaming tests."""

    def test_streaming_on_gpu(self, model_and_tokenizer):
        """Test streaming generation on GPU."""
        model, tokenizer = model_and_tokenizer

        searcher = StreamingFGBS(
            model=model,
            tokenizer=tokenizer,
            beam_width=3,
            alpha=0.5,
            bond_dim=8,
            device="cuda",
        )

        prompt = "Hello"
        tokens = list(searcher.generate_stream(prompt, max_length=5))

        assert len(tokens) > 0
        assert searcher.device.type == "cuda"
