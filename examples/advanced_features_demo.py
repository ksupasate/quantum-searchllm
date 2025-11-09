"""
Advanced Features Demo

Demonstrates all new features in TNAD v1.0:
- Encoder-decoder models (T5, BART)
- Multi-GPU distributed beam search
- vLLM integration
- Streaming generation
- Fine-tuning with coherence rewards
- API server
- Web demo

Run each example independently or all together.
"""

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


# ============================================================================
# Example 1: Encoder-Decoder Models (T5)
# ============================================================================


def example_encoder_decoder():
    """Demonstrate T5 generation with FGBS."""
    print("\n" + "=" * 70)
    print("Example 1: Encoder-Decoder FGBS (T5)")
    print("=" * 70)

    from tnad import EncoderDecoderFGBS

    # Load T5
    print("Loading T5-small model...")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Create searcher
    searcher = EncoderDecoderFGBS(
        model=model,
        tokenizer=tokenizer,
        beam_width=5,
        alpha=0.5,
        bond_dim=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Summarization
    input_text = (
        "summarize: The tower is 324 metres (1,063 ft) tall, about the same height "
        "as an 81-storey building. It was the first structure to reach a height of "
        "300 metres. It is now taller than the Chrysler Building in New York City."
    )

    print(f"\nInput: {input_text}")
    print("\nGenerating with FGBS...")

    result = searcher.generate(input_text, max_length=50, return_details=True)

    print(f"Output: {result['text']}")
    print(f"CFS: {result['log_cfs']:.4f}")
    print(f"Tokens: {len(result['token_ids'])}")


# ============================================================================
# Example 2: Streaming Generation
# ============================================================================


def example_streaming():
    """Demonstrate real-time streaming generation."""
    print("\n" + "=" * 70)
    print("Example 2: Streaming Generation")
    print("=" * 70)

    from tnad import StreamingFGBS

    # Load model
    print("Loading GPT-2 model...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create streaming searcher
    searcher = StreamingFGBS(
        model=model,
        tokenizer=tokenizer,
        beam_width=5,
        alpha=0.5,
        bond_dim=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Stream generation
    prompt = "Once upon a time in a distant galaxy,"
    print(f"\nPrompt: {prompt}")
    print("Streaming output:", end=" ")

    for token_info in searcher.generate_stream(prompt, max_length=30):
        print(token_info.token, end="", flush=True)

        # Show coherence in real-time
        if token_info.position % 10 == 0:
            import math

            cfs = math.exp(token_info.log_cfs)
            print(f" [CFS: {cfs:.2f}]", end="", flush=True)

    print("\n")


# ============================================================================
# Example 3: Distributed Multi-GPU (requires multiple GPUs)
# ============================================================================


def example_distributed():
    """Demonstrate distributed beam search (requires 2+ GPUs)."""
    print("\n" + "=" * 70)
    print("Example 3: Distributed Multi-GPU Beam Search")
    print("=" * 70)

    if torch.cuda.device_count() < 2:
        print("⚠️  Requires 2+ GPUs. Skipping this example.")
        return

    from tnad import DistributedFGBS, setup_distributed, cleanup_distributed

    # Setup distributed
    rank, world_size = setup_distributed()

    if rank == 0:
        print(f"Running on {world_size} GPUs")

    # Load model on each GPU
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", device_map=f"cuda:{rank}"
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create distributed searcher
    searcher = DistributedFGBS(
        model=model,
        tokenizer=tokenizer,
        beam_width=16,  # Split across GPUs
        alpha=0.5,
        bond_dim=16,
        rank=rank,
        world_size=world_size,
    )

    # Generate (same prompt on all ranks)
    prompt = "The future of artificial intelligence is"

    if rank == 0:
        print(f"Prompt: {prompt}")

    result = searcher.generate(prompt, max_length=50)

    # Only rank 0 gets result
    if rank == 0:
        print(f"Output: {result['text']}")
        print(f"CFS: {result['log_cfs']:.4f}")

    cleanup_distributed()


# ============================================================================
# Example 4: Fine-tuning with Coherence Rewards
# ============================================================================


def example_finetuning():
    """Demonstrate fine-tuning with coherence rewards."""
    print("\n" + "=" * 70)
    print("Example 4: Fine-tuning with Coherence Rewards")
    print("=" * 70)

    from tnad import CoherenceFilteredDataset, CoherenceRewardTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    # Load small model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")
    except Exception:
        print("⚠️  Dataset not available. Using dummy data.")
        dataset = [
            {"text": "This is a test sentence."},
            {"text": "Another coherent example."},
            {"text": "Logical reasoning is important."},
        ]

    # Filter by coherence
    print("Filtering dataset by coherence...")
    filtered_dataset = CoherenceFilteredDataset(
        data=dataset,
        tokenizer=tokenizer,
        embedding_layer=model.get_input_embeddings(),
        min_cfs=0.5,
        bond_dim=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"Filtered dataset size: {len(filtered_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./coherence_finetuned",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=100,
        logging_steps=10,
        max_steps=10,  # Short demo
    )

    # Create trainer
    trainer = CoherenceRewardTrainer(
        model=model,
        train_dataset=filtered_dataset,
        args=training_args,
        coherence_weight=0.5,
    )

    print("\nTraining with coherence rewards...")
    trainer.train()
    print("Training complete!")


# ============================================================================
# Example 5: vLLM Integration (Linux + CUDA only)
# ============================================================================


def example_vllm():
    """Demonstrate vLLM integration for production inference."""
    print("\n" + "=" * 70)
    print("Example 5: vLLM Production Deployment")
    print("=" * 70)

    try:
        from tnad import vLLMFGBS
    except ImportError:
        print("⚠️  vLLM not installed. Install with: pip install vllm")
        print("    Note: vLLM requires Linux and CUDA.")
        return

    # Initialize vLLM FGBS
    print("Initializing vLLM with FGBS...")
    fgbs = vLLMFGBS(
        model_name="gpt2",
        alpha=0.5,
        bond_dim=16,
        tensor_parallel_size=1,
    )

    # Single generation
    prompt = "Explain quantum computing in simple terms:"
    print(f"\nPrompt: {prompt}")

    result = fgbs.generate(prompt, max_tokens=50)

    print(f"Output: {result['text']}")
    print(f"CFS: {result['cfs']:.4f}")
    print(f"Latency: {result.get('latency_ms', 'N/A')} ms")

    # Batch generation
    prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "What is deep learning?",
    ]

    print("\nBatch generation...")
    results = fgbs.generate_batch(prompts, max_tokens=30)

    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['text'][:100]}...")
        print(f"   CFS: {result['cfs']:.4f}")


# ============================================================================
# Example 6: Web Demo (Gradio)
# ============================================================================


def example_web_demo():
    """Launch interactive web demo."""
    print("\n" + "=" * 70)
    print("Example 6: Web Demo (Gradio)")
    print("=" * 70)

    from tnad.web_demo import TNADDemo

    # Create demo
    print("Creating web demo...")
    demo = TNADDemo(
        model_name="gpt2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    interface = demo.create_interface()

    print("\nLaunching demo...")
    print("Open http://localhost:7860 in your browser")

    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)


# ============================================================================
# Example 7: API Server (FastAPI)
# ============================================================================


def example_api_server():
    """Launch production API server."""
    print("\n" + "=" * 70)
    print("Example 7: API Server (FastAPI)")
    print("=" * 70)

    from tnad.api_server import main as api_main

    print("\nStarting API server on http://localhost:8000")
    print("\nEndpoints:")
    print("  POST /generate - Single generation")
    print("  POST /generate/batch - Batch generation")
    print("  GET  /stream - SSE streaming")
    print("  WS   /ws/generate - WebSocket streaming")
    print("  GET  /health - Health check")
    print("  GET  /metrics - Metrics")
    print("\nPress Ctrl+C to stop")

    # This will block - run in separate process for production
    api_main()


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="TNAD Advanced Features Demo")
    parser.add_argument(
        "--example",
        type=str,
        choices=[
            "encoder-decoder",
            "streaming",
            "distributed",
            "finetuning",
            "vllm",
            "web-demo",
            "api-server",
            "all",
        ],
        default="all",
        help="Which example to run",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TNAD v1.0 - Advanced Features Demo")
    print("=" * 70)

    examples = {
        "encoder-decoder": example_encoder_decoder,
        "streaming": example_streaming,
        "distributed": example_distributed,
        "finetuning": example_finetuning,
        "vllm": example_vllm,
        "web-demo": example_web_demo,
        "api-server": example_api_server,
    }

    if args.example == "all":
        # Run all except web-demo and api-server (which block)
        for name, func in examples.items():
            if name not in ["web-demo", "api-server"]:
                try:
                    func()
                except Exception as e:
                    print(f"\n⚠️  Example '{name}' failed: {e}")
    else:
        examples[args.example]()

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
