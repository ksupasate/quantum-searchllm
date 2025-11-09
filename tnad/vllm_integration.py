"""
vLLM Integration for Production FGBS Deployment

Integrates FGBS with vLLM for high-throughput, low-latency inference.

vLLM Benefits:
    - PagedAttention: Efficient KV cache management (reduces memory by 3-4x)
    - Continuous batching: Process multiple requests simultaneously
    - Optimized kernels: Fused attention, efficient sampling
    - Tensor parallelism: Scale to multi-GPU easily
    - Async API: Non-blocking request handling

Performance Improvements over HuggingFace:
    - 5-10x higher throughput
    - 2-3x lower latency for long sequences
    - 3-4x better GPU utilization
    - Supports larger batch sizes

Integration Strategy:
    Since vLLM controls the generation loop, we adapt FGBS to work with
    vLLM's sampling parameters and logits processors.

    1. Use vLLM's LLM class for model inference
    2. Implement custom LogitsProcessor for coherence scoring
    3. Track MPS state per request using request_id
    4. Return coherence metrics alongside generated text
"""

import asyncio
import math
from typing import Any, Dict, List, Optional, Tuple

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

try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning(
        "vLLM not installed. Install with: pip install vllm\n"
        "Note: vLLM requires CUDA and Linux. For other platforms, "
        "use standard FGBS implementation."
    )
    # Provide stub classes for type hints
    class LLM:
        pass

    class SamplingParams:
        pass

    class LogitsProcessor:
        pass


from tnad.coherence_score import compute_cfs_from_mps
from tnad.mps_manager import MPSSequence


class CoherenceLogitsProcessor(LogitsProcessor):
    """
    Custom logits processor that applies CFS-based reranking.

    This processor intercepts token logits during generation and
    adjusts them based on coherence fidelity scores from MPS.

    Process:
        1. vLLM generates next-token logits
        2. Extract top-k candidate tokens
        3. For each candidate:
           - Simulate adding token to MPS
           - Compute resulting CFS
        4. Combine log_prob and log_CFS with alpha weighting
        5. Return adjusted logits

    Note: This is a stateful processor that maintains MPS per request.
    """

    def __init__(
        self,
        bond_dim: int = 16,
        alpha: float = 0.5,
        top_k: int = 50,
        embedding_layer: Optional[torch.nn.Module] = None,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize coherence logits processor.

        Args:
            bond_dim: MPS bond dimension
            alpha: Balance between fluency (1.0) and coherence (0.0)
            top_k: Number of top tokens to consider for CFS computation
            embedding_layer: Token embedding layer from the model
            normalize_embeddings: Normalize token embeddings
        """
        super().__init__()
        self.bond_dim = bond_dim
        self.alpha = alpha
        self.top_k = top_k
        self.embedding_layer = embedding_layer
        self.normalize_embeddings = normalize_embeddings

        # Track MPS per request
        self.mps_states: Dict[str, MPSSequence] = {}

        logger.info(
            f"Initialized CoherenceLogitsProcessor: "
            f"χ={bond_dim}, α={alpha}, k={top_k}"
        )

    def __call__(
        self,
        token_ids: List[int],
        logits: torch.Tensor,
        request_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Process logits to incorporate coherence scoring.

        Args:
            token_ids: Generated token IDs so far
            logits: Next-token logits [vocab_size]
            request_id: Unique request identifier for tracking MPS

        Returns:
            Adjusted logits [vocab_size]
        """
        if self.alpha >= 0.99:  # Pure LLM mode, no coherence adjustment
            return logits

        # Initialize or retrieve MPS for this request
        if request_id not in self.mps_states:
            mps = MPSSequence(
                bond_dim=self.bond_dim,
                embedding_dim=self.embedding_layer.embedding_dim,
                device=logits.device,
                normalize_embeddings=self.normalize_embeddings,
            )

            # Add existing tokens to MPS
            with torch.no_grad():
                token_tensor = torch.tensor(token_ids, device=logits.device)
                embeddings = self.embedding_layer(token_tensor)
                for emb in embeddings:
                    mps.add_token(emb)

            self.mps_states[request_id] = mps
        else:
            mps = self.mps_states[request_id]

            # Add most recent token (if new)
            if len(token_ids) > mps.get_current_length():
                with torch.no_grad():
                    new_token = torch.tensor([token_ids[-1]], device=logits.device)
                    new_embedding = self.embedding_layer(new_token)[0]
                    mps.add_token(new_embedding)

        # Get top-k candidates
        log_probs = torch.log_softmax(logits, dim=-1)
        current_top_k = min(self.top_k, log_probs.shape[-1])
        topk_log_probs, topk_indices = torch.topk(log_probs, current_top_k)

        # Compute CFS for each candidate
        cfs_scores = []
        with torch.no_grad():
            # Batch get embeddings for all candidates
            candidate_embeddings = self.embedding_layer(topk_indices)

            for emb in candidate_embeddings:
                # Simulate adding token
                test_mps = mps.copy()
                test_mps.add_token(emb)

                # Compute CFS
                cfs = compute_cfs_from_mps(test_mps)
                log_cfs = math.log(max(cfs, 1e-12))
                cfs_scores.append(log_cfs)

        # Combine scores: composite = alpha * log_prob + (1-alpha) * log_cfs
        cfs_tensor = torch.tensor(cfs_scores, device=logits.device, dtype=logits.dtype)
        composite_scores = self.alpha * topk_log_probs + (1 - self.alpha) * cfs_tensor

        # Update logits: set top-k to composite scores, rest to -inf
        adjusted_logits = torch.full_like(logits, float("-inf"))
        adjusted_logits[topk_indices] = composite_scores

        return adjusted_logits

    def cleanup_request(self, request_id: str):
        """Clean up MPS state for completed request."""
        if request_id in self.mps_states:
            del self.mps_states[request_id]


class vLLMFGBS:
    """
    FGBS interface using vLLM for production deployment.

    Provides high-throughput coherence-aware generation using vLLM's
    optimized inference engine.

    Example Usage:
        >>> # Initialize vLLM with FGBS
        >>> fgbs = vLLMFGBS(
        >>>     model_name="meta-llama/Llama-3.1-8B-Instruct",
        >>>     alpha=0.5,
        >>>     bond_dim=16,
        >>>     tensor_parallel_size=2,  # Use 2 GPUs
        >>> )
        >>>
        >>> # Single generation
        >>> result = fgbs.generate(
        >>>     "Solve: If x + 2 = 5, then x = ?",
        >>>     max_tokens=100
        >>> )
        >>> print(result['text'])
        >>> print(f"CFS: {result['cfs']:.2f}")
        >>>
        >>> # Batch generation
        >>> prompts = ["Question 1", "Question 2", "Question 3"]
        >>> results = fgbs.generate_batch(prompts, max_tokens=100)
    """

    def __init__(
        self,
        model_name: str,
        alpha: float = 0.5,
        bond_dim: int = 16,
        top_k: int = 50,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        """
        Initialize vLLM FGBS.

        Args:
            model_name: HuggingFace model name or path
            alpha: Fluency vs coherence balance
            bond_dim: MPS bond dimension
            top_k: Number of candidates for CFS computation
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_model_len: Maximum sequence length (None = model default)
            dtype: Model dtype ('auto', 'float16', 'bfloat16')
            quantization: Quantization method ('awq', 'gptq', None)
            trust_remote_code: Trust remote code in model

        Raises:
            ImportError: If vLLM is not installed
            RuntimeError: If vLLM initialization fails
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm\n"
                "Note: vLLM requires CUDA and Linux."
            )

        self.model_name = model_name
        self.alpha = alpha
        self.bond_dim = bond_dim
        self.top_k = top_k

        logger.info(f"Initializing vLLM FGBS: model={model_name}, α={alpha}, χ={bond_dim}")

        # Initialize vLLM engine
        try:
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype=dtype,
                quantization=quantization,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vLLM: {e}")

        # Get embedding layer for CFS computation
        # Note: vLLM doesn't expose embedding layer directly, so we access internal model
        self.embedding_layer = self.llm.llm_engine.model_executor.driver_worker.model_runner.model.get_input_embeddings()

        # Create coherence logits processor
        self.coherence_processor = CoherenceLogitsProcessor(
            bond_dim=bond_dim,
            alpha=alpha,
            top_k=top_k,
            embedding_layer=self.embedding_layer,
        )

        logger.info("vLLM FGBS initialized successfully")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate text with coherence-aware beam search.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            return_details: Return detailed metrics

        Returns:
            Dictionary with:
                - 'text': Generated text
                - 'cfs': Final coherence fidelity score
                - 'tokens': Number of generated tokens
                If return_details=True:
                - 'finish_reason': Why generation stopped
                - 'prompt_tokens': Number of prompt tokens
        """
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logits_processors=[
                lambda token_ids, logits: self.coherence_processor(
                    token_ids, logits, request_id=prompt[:32]  # Use prompt hash as ID
                )
            ],
        )

        # Generate
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]

        # Extract generated text
        generated_text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids

        # Compute final CFS
        request_id = prompt[:32]
        final_mps = self.coherence_processor.mps_states.get(request_id)
        final_cfs = compute_cfs_from_mps(final_mps) if final_mps else 1.0

        # Cleanup
        self.coherence_processor.cleanup_request(request_id)

        # Prepare result
        result = {
            "text": generated_text,
            "cfs": final_cfs,
            "tokens": len(token_ids),
        }

        if return_details:
            result.update(
                {
                    "finish_reason": output.outputs[0].finish_reason,
                    "prompt_tokens": len(output.prompt_token_ids),
                }
            )

        return result

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate for multiple prompts in parallel.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            **kwargs: Additional sampling parameters

        Returns:
            List of generation results (one per prompt)
        """
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens, temperature=temperature, **kwargs
        )

        # Attach coherence processor for each prompt
        sampling_params.logits_processors = [
            lambda token_ids, logits, pid=p[:32]: self.coherence_processor(
                token_ids, logits, request_id=pid
            )
            for p in prompts
        ]

        # Batch generate
        outputs = self.llm.generate(prompts, sampling_params)

        # Process results
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids

            # Get final CFS
            request_id = prompts[i][:32]
            final_mps = self.coherence_processor.mps_states.get(request_id)
            final_cfs = compute_cfs_from_mps(final_mps) if final_mps else 1.0

            # Cleanup
            self.coherence_processor.cleanup_request(request_id)

            results.append(
                {
                    "text": generated_text,
                    "cfs": final_cfs,
                    "tokens": len(token_ids),
                }
            )

        return results

    async def generate_async(
        self, prompt: str, max_tokens: int = 100, **kwargs
    ) -> Dict[str, Any]:
        """
        Async generation (runs in thread pool to avoid blocking).

        Useful for building async API servers.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generation result dictionary
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self.generate, prompt, max_tokens, **kwargs
        )
        return result


def benchmark_vllm_vs_hf(
    model_name: str,
    prompts: List[str],
    max_tokens: int = 100,
    alpha: float = 0.5,
    bond_dim: int = 16,
) -> Dict[str, Any]:
    """
    Benchmark vLLM vs HuggingFace FGBS implementations.

    Compares throughput, latency, and memory usage.

    Args:
        model_name: Model to benchmark
        prompts: List of test prompts
        max_tokens: Maximum generation length
        alpha: Coherence balance parameter
        bond_dim: MPS bond dimension

    Returns:
        Benchmark results with timing and memory stats
    """
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tnad.fgbs_searcher import FidelityGuidedBeamSearcher

    logger.info("Starting benchmark: vLLM vs HuggingFace")

    # Benchmark vLLM
    if VLLM_AVAILABLE:
        logger.info("Benchmarking vLLM...")
        vllm_fgbs = vLLMFGBS(
            model_name=model_name, alpha=alpha, bond_dim=bond_dim
        )

        start = time.time()
        vllm_results = vllm_fgbs.generate_batch(prompts, max_tokens=max_tokens)
        vllm_time = time.time() - start

        vllm_throughput = len(prompts) / vllm_time
        vllm_avg_latency = vllm_time / len(prompts)
    else:
        vllm_results = None
        vllm_time = float("inf")
        vllm_throughput = 0
        vllm_avg_latency = float("inf")

    # Benchmark HuggingFace
    logger.info("Benchmarking HuggingFace...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    hf_fgbs = FidelityGuidedBeamSearcher(
        model=model, tokenizer=tokenizer, alpha=alpha, bond_dim=bond_dim
    )

    start = time.time()
    hf_results = []
    for prompt in prompts:
        result = hf_fgbs.generate(prompt, max_length=max_tokens, show_progress=False)
        hf_results.append(result)
    hf_time = time.time() - start

    hf_throughput = len(prompts) / hf_time
    hf_avg_latency = hf_time / len(prompts)

    # Compare
    speedup = hf_time / vllm_time if vllm_time < float("inf") else 0

    return {
        "vllm": {
            "total_time": vllm_time,
            "throughput": vllm_throughput,
            "avg_latency": vllm_avg_latency,
            "results": vllm_results,
        },
        "huggingface": {
            "total_time": hf_time,
            "throughput": hf_throughput,
            "avg_latency": hf_avg_latency,
            "results": hf_results,
        },
        "speedup": speedup,
    }


__all__ = ["vLLMFGBS", "CoherenceLogitsProcessor", "benchmark_vllm_vs_hf"]
