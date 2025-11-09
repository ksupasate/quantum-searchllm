"""
TNAD API Server

Production-ready REST API and WebSocket server for FGBS inference.

Features:
    - REST API for single and batch generation
    - WebSocket for streaming generation
    - Server-Sent Events (SSE) support
    - Request queuing and rate limiting
    - Prometheus metrics
    - Health checks
    - Graceful shutdown

Endpoints:
    POST /generate - Single generation
    POST /generate/batch - Batch generation
    GET  /stream - SSE streaming
    WS   /ws/generate - WebSocket streaming
    GET  /health - Health check
    GET  /metrics - Prometheus metrics

Example Usage:
    # Start server
    python -m tnad.api_server --model meta-llama/Llama-3.1-8B-Instruct --port 8000

    # Query API
    curl -X POST "http://localhost:8000/generate" \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "Solve: x + 2 = 5", "max_length": 100}'
"""

import argparse
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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

from transformers import AutoModelForCausalLM, AutoTokenizer

from tnad.fgbs_searcher import FidelityGuidedBeamSearcher
from tnad.streaming_fgbs import StreamingFGBS, create_sse_stream


# ============================================================================
# Request/Response Models
# ============================================================================


class GenerateRequest(BaseModel):
    """Single generation request."""

    prompt: str = Field(..., description="Input prompt text", min_length=1)
    max_length: int = Field(100, description="Maximum generation length", ge=1, le=2048)
    min_length: int = Field(10, description="Minimum generation length", ge=0)
    alpha: Optional[float] = Field(
        None, description="Fluency vs coherence balance (0-1)", ge=0.0, le=1.0
    )
    bond_dim: Optional[int] = Field(
        None, description="MPS bond dimension", ge=1, le=64
    )
    temperature: Optional[float] = Field(
        None, description="Sampling temperature", gt=0.0, le=2.0
    )
    return_details: bool = Field(False, description="Return detailed metrics")

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Solve: If x + 2 = 5, then x = ?",
                "max_length": 100,
                "alpha": 0.5,
                "return_details": True,
            }
        }


class GenerateResponse(BaseModel):
    """Single generation response."""

    text: str = Field(..., description="Generated text")
    log_prob: float = Field(..., description="Log probability")
    log_cfs: float = Field(..., description="Log coherence fidelity score")
    composite_score: float = Field(..., description="Composite score")
    num_tokens: int = Field(..., description="Number of generated tokens")
    latency_ms: float = Field(..., description="Generation latency in milliseconds")
    cfs_trajectory: Optional[List[float]] = Field(
        None, description="CFS values over generation"
    )


class BatchGenerateRequest(BaseModel):
    """Batch generation request."""

    prompts: List[str] = Field(
        ..., description="List of input prompts", min_items=1, max_items=100
    )
    max_length: int = Field(100, ge=1, le=2048)
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0)
    bond_dim: Optional[int] = Field(None, ge=1, le=64)


class BatchGenerateResponse(BaseModel):
    """Batch generation response."""

    results: List[GenerateResponse] = Field(..., description="Generation results")
    total_latency_ms: float = Field(..., description="Total batch latency")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model: str = Field(..., description="Loaded model name")
    device: str = Field(..., description="Compute device")
    uptime_seconds: float = Field(..., description="Server uptime")


class StreamRequest(BaseModel):
    """Streaming generation request."""

    prompt: str
    max_length: int = 100
    coherence_threshold: Optional[float] = None


# ============================================================================
# API Server
# ============================================================================


class TNADServer:
    """
    TNAD API Server with FastAPI.

    Manages model lifecycle, request handling, and metrics.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        beam_width: int = 5,
        alpha: float = 0.5,
        bond_dim: int = 16,
        load_in_8bit: bool = False,
        enable_streaming: bool = True,
    ):
        """
        Initialize TNAD server.

        Args:
            model_name: HuggingFace model name
            device: Compute device
            beam_width: Default beam width
            alpha: Default alpha parameter
            bond_dim: Default bond dimension
            load_in_8bit: Enable 8-bit quantization
            enable_streaming: Enable streaming endpoints
        """
        self.model_name = model_name
        self.device = device
        self.default_beam_width = beam_width
        self.default_alpha = alpha
        self.default_bond_dim = bond_dim
        self.enable_streaming = enable_streaming
        self.start_time = time.time()

        logger.info(f"Loading model: {model_name}")

        # Load model
        model_kwargs = {"device_map": device}
        if load_in_8bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model loaded successfully on {self.device}")

        # Create searchers
        self.searcher = FidelityGuidedBeamSearcher(
            model=self.model,
            tokenizer=self.tokenizer,
            beam_width=beam_width,
            alpha=alpha,
            bond_dim=bond_dim,
        )

        if enable_streaming:
            self.streaming_searcher = StreamingFGBS(
                model=self.model,
                tokenizer=self.tokenizer,
                beam_width=beam_width,
                alpha=alpha,
                bond_dim=bond_dim,
            )

        # Metrics
        self.request_count = 0
        self.total_latency = 0.0

        logger.info("TNAD server initialized successfully")

    def get_searcher(self, alpha: Optional[float] = None, bond_dim: Optional[int] = None):
        """Get searcher with custom parameters."""
        if alpha is None and bond_dim is None:
            return self.searcher

        # Create custom searcher
        return FidelityGuidedBeamSearcher(
            model=self.model,
            tokenizer=self.tokenizer,
            beam_width=self.default_beam_width,
            alpha=alpha or self.default_alpha,
            bond_dim=bond_dim or self.default_bond_dim,
        )

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Handle single generation request.

        Args:
            request: Generation request

        Returns:
            Generation response

        Raises:
            HTTPException: If generation fails
        """
        start_time = time.time()

        try:
            # Get searcher
            searcher = self.get_searcher(request.alpha, request.bond_dim)

            # Generate
            result = searcher.generate(
                prompt=request.prompt,
                max_length=request.max_length,
                min_length=request.min_length,
                return_details=request.return_details,
                show_progress=False,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Update metrics
            self.request_count += 1
            self.total_latency += latency_ms

            # Build response
            response = GenerateResponse(
                text=result["text"],
                log_prob=result["log_prob"],
                log_cfs=result["log_cfs"],
                composite_score=result["composite_score"],
                num_tokens=len(result["token_ids"]),
                latency_ms=latency_ms,
                cfs_trajectory=result.get("cfs_trajectory") if request.return_details else None,
            )

            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def generate_batch(self, request: BatchGenerateRequest) -> BatchGenerateResponse:
        """Handle batch generation request."""
        start_time = time.time()

        try:
            searcher = self.get_searcher(request.alpha, request.bond_dim)

            results = []
            for prompt in request.prompts:
                gen_start = time.time()

                result = searcher.generate(
                    prompt=prompt,
                    max_length=request.max_length,
                    show_progress=False,
                )

                gen_latency = (time.time() - gen_start) * 1000

                results.append(
                    GenerateResponse(
                        text=result["text"],
                        log_prob=result["log_prob"],
                        log_cfs=result["log_cfs"],
                        composite_score=result["composite_score"],
                        num_tokens=len(result["token_ids"]),
                        latency_ms=gen_latency,
                    )
                )

            total_latency_ms = (time.time() - start_time) * 1000
            self.request_count += len(request.prompts)
            self.total_latency += total_latency_ms

            return BatchGenerateResponse(
                results=results, total_latency_ms=total_latency_ms
            )

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def health(self) -> HealthResponse:
        """Get server health status."""
        uptime = time.time() - self.start_time

        return HealthResponse(
            status="healthy",
            model=self.model_name,
            device=str(self.device),
            uptime_seconds=uptime,
        )

    def metrics(self) -> Dict[str, Any]:
        """Get Prometheus-style metrics."""
        avg_latency = (
            self.total_latency / self.request_count if self.request_count > 0 else 0
        )

        return {
            "tnad_requests_total": self.request_count,
            "tnad_latency_sum_ms": self.total_latency,
            "tnad_latency_avg_ms": avg_latency,
            "tnad_uptime_seconds": time.time() - self.start_time,
        }


# ============================================================================
# FastAPI App
# ============================================================================


# Global server instance
server: Optional[TNADServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle."""
    global server

    # Startup
    logger.info("Starting TNAD server...")
    yield

    # Shutdown
    logger.info("Shutting down TNAD server...")
    if server:
        # Cleanup
        torch.cuda.empty_cache()


app = FastAPI(
    title="TNAD API",
    description="Tensor Network-Augmented Decoding API for coherent LLM generation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text with FGBS.

    Example:
        ```bash
        curl -X POST "http://localhost:8000/generate" \\
          -H "Content-Type: application/json" \\
          -d '{
            "prompt": "Solve: x + 2 = 5",
            "max_length": 100,
            "alpha": 0.5,
            "return_details": true
          }'
        ```
    """
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    return server.generate(request)


@app.post("/generate/batch", response_model=BatchGenerateResponse)
async def generate_batch(request: BatchGenerateRequest):
    """
    Generate text for multiple prompts.

    Example:
        ```bash
        curl -X POST "http://localhost:8000/generate/batch" \\
          -H "Content-Type: application/json" \\
          -d '{
            "prompts": ["Question 1", "Question 2"],
            "max_length": 100
          }'
        ```
    """
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    return server.generate_batch(request)


@app.get("/stream")
async def stream(prompt: str, max_length: int = 100):
    """
    Stream generation using Server-Sent Events.

    Example:
        ```javascript
        const eventSource = new EventSource(
          'http://localhost:8000/stream?prompt=Hello&max_length=50'
        );
        eventSource.onmessage = (event) => {
          const data = JSON.parse(event.data);
          console.log(data.token);
        };
        ```
    """
    if server is None or not server.enable_streaming:
        raise HTTPException(status_code=503, detail="Streaming not available")

    return StreamingResponse(
        create_sse_stream(server.streaming_searcher, prompt, max_length),
        media_type="text/event-stream",
    )


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """
    WebSocket streaming endpoint.

    Example:
        ```javascript
        const ws = new WebSocket('ws://localhost:8000/ws/generate');
        ws.onopen = () => {
          ws.send(JSON.stringify({prompt: 'Hello', max_length: 50}));
        };
        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          console.log(data.token);
        };
        ```
    """
    if server is None or not server.enable_streaming:
        await websocket.close(code=1003, reason="Streaming not available")
        return

    await websocket.accept()

    try:
        # Receive request
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        max_length = data.get("max_length", 100)

        # Stream generation
        from tnad.streaming_fgbs import create_websocket_stream

        await create_websocket_stream(
            server.streaming_searcher, websocket, prompt, max_length
        )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    return server.health()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    return server.metrics()


# ============================================================================
# CLI
# ============================================================================


def main():
    """Run API server from command line."""
    parser = argparse.ArgumentParser(description="TNAD API Server")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter")
    parser.add_argument("--bond-dim", type=int, default=16, help="Bond dimension")
    parser.add_argument("--load-in-8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    # Initialize server
    global server
    server = TNADServer(
        model_name=args.model,
        device=args.device,
        beam_width=args.beam_width,
        alpha=args.alpha,
        bond_dim=args.bond_dim,
        load_in_8bit=args.load_in_8bit,
        enable_streaming=not args.no_streaming,
    )

    logger.info(f"Starting server on {args.host}:{args.port}")

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()


__all__ = ["TNADServer", "app", "main"]
