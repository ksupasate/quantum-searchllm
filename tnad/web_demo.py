"""
TNAD Web Demo using Gradio

Interactive web interface for testing TNAD (Tensor Network-Augmented Decoding).

Features:
    - User-friendly interface for FGBS generation
    - Real-time coherence monitoring
    - Parameter tuning sliders
    - Comparison with baseline models
    - Visualization of coherence trajectories
    - Export results to JSON/CSV

Example Usage:
    # Launch demo locally
    python -m tnad.web_demo --model meta-llama/Llama-3.1-8B-Instruct

    # Launch with custom settings
    python -m tnad.web_demo \\
        --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --share \\
        --port 7860

    # Launch with authentication
    python -m tnad.web_demo \\
        --model gpt2 \\
        --auth username:password
"""

import argparse
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
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

from transformers import AutoModelForCausalLM, AutoTokenizer

from tnad.fgbs_searcher import FidelityGuidedBeamSearcher


class TNADDemo:
    """
    TNAD Gradio Demo Interface.

    Provides interactive web UI for FGBS generation with coherence monitoring.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        load_in_8bit: bool = False,
    ):
        """
        Initialize demo.

        Args:
            model_name: HuggingFace model name
            device: Compute device
            load_in_8bit: Enable 8-bit quantization
        """
        self.model_name = model_name

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

        logger.info("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_length: int,
        beam_width: int,
        alpha: float,
        bond_dim: int,
        temperature: float,
        show_comparison: bool,
    ) -> Tuple[str, str, str, Optional[plt.Figure]]:
        """
        Generate text with FGBS.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            beam_width: Number of beams
            alpha: Fluency vs coherence balance
            bond_dim: MPS bond dimension
            temperature: Sampling temperature
            show_comparison: Compare with baseline

        Returns:
            Tuple of (generated_text, metrics_json, comparison_text, plot)
        """
        if not prompt or not prompt.strip():
            return "Error: Prompt cannot be empty", "", "", None

        try:
            # Create searcher
            searcher = FidelityGuidedBeamSearcher(
                model=self.model,
                tokenizer=self.tokenizer,
                beam_width=beam_width,
                alpha=alpha,
                bond_dim=bond_dim,
                temperature=temperature,
            )

            # Generate
            logger.info(f"Generating with FGBS: α={alpha}, χ={bond_dim}, B={beam_width}")
            result = searcher.generate(
                prompt,
                max_length=max_length,
                return_details=True,
                show_progress=True,
            )

            generated_text = result["text"]
            cfs_trajectory = result["cfs_trajectory"]
            score_trajectory = result["score_trajectory"]

            # Metrics
            metrics = {
                "final_cfs": math.exp(result["log_cfs"]),
                "log_prob": result["log_prob"],
                "composite_score": result["composite_score"],
                "num_tokens": len(result["token_ids"]),
                "avg_cfs": sum(cfs_trajectory) / len(cfs_trajectory) if cfs_trajectory else 0,
            }
            metrics_json = json.dumps(metrics, indent=2)

            # Comparison with baseline
            comparison_text = ""
            if show_comparison:
                logger.info("Running baseline comparison...")
                comparison = searcher.compare_with_baseline(prompt, max_length)

                fgbs_cfs = comparison["cfs_comparison"]["fgbs_final_cfs"]
                baseline_cfs = comparison["cfs_comparison"]["baseline_final_cfs"]
                improvement = comparison["cfs_comparison"]["cfs_improvement"]

                comparison_text = f"""
**FGBS (α={alpha})**
- Text: {comparison['fgbs']['text'][:200]}...
- CFS: {fgbs_cfs:.4f}

**Baseline (Standard Beam Search)**
- Text: {comparison['baseline']['text'][:200]}...
- CFS: {baseline_cfs:.4f}

**Improvement**: {improvement:+.4f} ({(improvement/baseline_cfs*100):+.1f}%)
"""

            # Plot coherence trajectory
            fig = self._plot_trajectories(cfs_trajectory, score_trajectory)

            return generated_text, metrics_json, comparison_text, fig

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}", "", "", None

    def _plot_trajectories(
        self, cfs_trajectory: List[float], score_trajectory: List[float]
    ) -> plt.Figure:
        """Plot CFS and score trajectories."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        # CFS trajectory
        ax1.plot(cfs_trajectory, marker="o", linewidth=2, markersize=4)
        ax1.set_xlabel("Generation Step")
        ax1.set_ylabel("Coherence Fidelity Score (CFS)")
        ax1.set_title("Coherence Evolution During Generation")
        ax1.grid(True, alpha=0.3)

        # Score trajectory
        ax2.plot(
            score_trajectory, marker="s", color="green", linewidth=2, markersize=4
        )
        ax2.set_xlabel("Generation Step")
        ax2.set_ylabel("Composite Score")
        ax2.set_title("Composite Score (α·log P + (1-α)·log F)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface.

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="TNAD: Tensor Network-Augmented Decoding",
            theme=gr.themes.Soft(),
        ) as demo:
            gr.Markdown(
                """
                # 🧠 TNAD: Tensor Network-Augmented Decoding

                **Quantum-inspired inference framework for coherent LLM reasoning**

                This demo showcases Fidelity-Guided Beam Search (FGBS), which balances:
                - **Fluency** (standard LLM probability)
                - **Coherence** (quantum-inspired fidelity score)

                Adjust the **alpha (α)** parameter to control the balance:
                - α = 1.0: Pure fluency (standard beam search)
                - α = 0.5: Balanced (recommended)
                - α = 0.0: Pure coherence (may sacrifice fluency)
                """
            )

            with gr.Row():
                with gr.Column(scale=2):
                    # Input section
                    gr.Markdown("### 📝 Input")
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3,
                        value="Solve step by step: If x + 2 = 5, then what is x?",
                    )

                    # Parameters
                    with gr.Accordion("⚙️ FGBS Parameters", open=True):
                        max_length_slider = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=100,
                            step=10,
                            label="Max Length",
                        )
                        beam_width_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Beam Width (B)",
                        )
                        alpha_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Alpha (α) - Fluency vs Coherence",
                        )
                        bond_dim_slider = gr.Slider(
                            minimum=4,
                            maximum=64,
                            value=16,
                            step=4,
                            label="Bond Dimension (χ)",
                        )
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Temperature",
                        )

                    show_comparison_checkbox = gr.Checkbox(
                        label="Compare with Baseline",
                        value=False,
                    )

                    generate_button = gr.Button("🚀 Generate", variant="primary")

                with gr.Column(scale=3):
                    # Output section
                    gr.Markdown("### 📤 Output")
                    output_text = gr.Textbox(
                        label="Generated Text",
                        lines=8,
                        interactive=False,
                    )

                    with gr.Tabs():
                        with gr.Tab("📊 Metrics"):
                            metrics_output = gr.JSON(label="Generation Metrics")

                        with gr.Tab("📈 Coherence Plot"):
                            plot_output = gr.Plot(label="Trajectories")

                        with gr.Tab("⚖️ Comparison"):
                            comparison_output = gr.Markdown()

            # Examples
            gr.Markdown("### 💡 Example Prompts")
            gr.Examples(
                examples=[
                    [
                        "Solve step by step: If x + 2 = 5, then what is x?",
                        100,
                        5,
                        0.5,
                        16,
                        1.0,
                        False,
                    ],
                    [
                        "Explain quantum computing in simple terms.",
                        150,
                        5,
                        0.3,
                        16,
                        1.0,
                        True,
                    ],
                    [
                        "If all cats are animals, and some animals can fly, can all cats fly?",
                        120,
                        5,
                        0.5,
                        16,
                        1.0,
                        False,
                    ],
                    [
                        "Write a coherent story about a time traveler.",
                        200,
                        5,
                        0.7,
                        32,
                        1.0,
                        False,
                    ],
                ],
                inputs=[
                    prompt_input,
                    max_length_slider,
                    beam_width_slider,
                    alpha_slider,
                    bond_dim_slider,
                    temperature_slider,
                    show_comparison_checkbox,
                ],
            )

            # Event handlers
            generate_button.click(
                fn=self.generate,
                inputs=[
                    prompt_input,
                    max_length_slider,
                    beam_width_slider,
                    alpha_slider,
                    bond_dim_slider,
                    temperature_slider,
                    show_comparison_checkbox,
                ],
                outputs=[output_text, metrics_output, comparison_output, plot_output],
            )

            # Footer
            gr.Markdown(
                f"""
                ---
                **Model**: `{self.model_name}`

                **Citation**: If you use TNAD in your research, please cite:
                ```bibtex
                @software{{tnad2024,
                  title={{TNAD: Tensor Network-Augmented Decoding for Coherent LLM Reasoning}},
                  author={{Supasate Vorathammathorn}},
                  year={{2025}},
                  url={{https://github.com/ksupasate/quantum-search-llm}}
                }}
                ```
                """
            )

        return demo


def main():
    """Launch Gradio demo from command line."""
    parser = argparse.ArgumentParser(description="TNAD Web Demo")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device (auto, cuda, cpu, mps)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Enable 8-bit quantization",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="Server name (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
    )
    parser.add_argument(
        "--auth",
        type=str,
        default=None,
        help="Authentication (username:password)",
    )

    args = parser.parse_args()

    # Parse authentication
    auth = None
    if args.auth:
        parts = args.auth.split(":")
        if len(parts) == 2:
            auth = (parts[0], parts[1])
        else:
            logger.warning("Invalid auth format. Use username:password")

    # Create demo
    demo_app = TNADDemo(
        model_name=args.model,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
    )

    interface = demo_app.create_interface()

    # Launch
    logger.info(
        f"Launching Gradio demo on {args.server_name}:{args.server_port}"
    )

    interface.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        auth=auth,
    )


if __name__ == "__main__":
    main()


__all__ = ["TNADDemo", "main"]
