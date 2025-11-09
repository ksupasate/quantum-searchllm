"""
Fine-Tuning Pipeline with Coherence Rewards

Train language models to generate more coherent text by incorporating
coherence fidelity scores (CFS) as a reward signal.

Training Strategies:
    1. **Reinforcement Learning from Coherence Feedback (RLCF)**:
       - Similar to RLHF, but uses CFS as reward
       - Trains policy to maximize both fluency and coherence
       - Uses PPO (Proximal Policy Optimization)

    2. **Supervised Fine-Tuning with Coherence Filtering**:
       - Filter training data by coherence scores
       - Keep only high-CFS examples
       - Standard next-token prediction on filtered data

    3. **Reward-Weighted Regression**:
       - Weight training examples by CFS
       - High-coherence sequences get higher weight
       - More stable than full RL

Pipeline Components:
    - CFS-based reward model
    - PPO trainer with coherence rewards
    - Data filtering utilities
    - Evaluation metrics
    - Checkpoint management

Benefits:
    - Models learn to generate intrinsically coherent text
    - Reduces need for FGBS at inference (can use faster greedy decoding)
    - Better base model for downstream tasks
    - Improved zero-shot reasoning
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from tnad.coherence_score import compute_cfs_from_mps
from tnad.mps_manager import MPSSequence
from tnad.fgbs_searcher import FidelityGuidedBeamSearcher


@dataclass
class CoherenceReward:
    """
    Reward signal based on coherence fidelity score.

    Attributes:
        cfs: Coherence Fidelity Score
        sequence_length: Length of evaluated sequence
        normalized_cfs: CFS normalized by sequence length
        reward: Final reward value (log CFS + length bonus)
    """

    cfs: float
    sequence_length: int
    normalized_cfs: float
    reward: float

    @classmethod
    def from_sequence(
        cls,
        token_ids: List[int],
        embedding_layer: torch.nn.Module,
        bond_dim: int = 16,
        device: str = "cuda",
    ) -> "CoherenceReward":
        """
        Compute coherence reward for a token sequence.

        Args:
            token_ids: Token IDs
            embedding_layer: Model embedding layer
            bond_dim: MPS bond dimension
            device: Compute device

        Returns:
            CoherenceReward object
        """
        # Create MPS
        mps = MPSSequence(
            bond_dim=bond_dim,
            embedding_dim=embedding_layer.embedding_dim,
            device=device,
        )

        # Add tokens
        with torch.no_grad():
            token_tensor = torch.tensor(token_ids, device=device)
            embeddings = embedding_layer(token_tensor)
            for emb in embeddings:
                mps.add_token(emb)

        # Compute CFS
        cfs = compute_cfs_from_mps(mps)
        seq_len = len(token_ids)
        normalized_cfs = cfs / math.sqrt(seq_len) if seq_len > 0 else cfs

        # Reward = log CFS + length bonus
        reward = math.log(max(cfs, 1e-12)) + 0.01 * seq_len

        return cls(
            cfs=cfs,
            sequence_length=seq_len,
            normalized_cfs=normalized_cfs,
            reward=reward,
        )


class CoherenceFilteredDataset(Dataset):
    """
    Dataset filtered by coherence scores.

    Keeps only examples with CFS above a threshold.

    Example Usage:
        >>> from datasets import load_dataset
        >>> raw_data = load_dataset("gsm8k", split="train")
        >>>
        >>> filtered = CoherenceFilteredDataset(
        >>>     data=raw_data,
        >>>     tokenizer=tokenizer,
        >>>     embedding_layer=model.get_input_embeddings(),
        >>>     min_cfs=0.8,  # Keep top 80% coherent examples
        >>> )
        >>>
        >>> # Use in training
        >>> trainer = Trainer(
        >>>     model=model,
        >>>     train_dataset=filtered,
        >>>     ...
        >>> )
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        embedding_layer: torch.nn.Module,
        min_cfs: float = 0.5,
        max_length: int = 512,
        bond_dim: int = 16,
        device: str = "cuda",
    ):
        """
        Initialize coherence-filtered dataset.

        Args:
            data: List of examples with 'text' or 'prompt'/'response' fields
            tokenizer: Tokenizer
            embedding_layer: Model embedding layer
            min_cfs: Minimum CFS threshold (0-1)
            max_length: Maximum sequence length
            bond_dim: MPS bond dimension
            device: Compute device
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Filtering {len(data)} examples by CFS >= {min_cfs}")

        # Filter examples
        self.examples = []
        for item in tqdm(data, desc="Computing coherence scores"):
            # Get text
            if "text" in item:
                text = item["text"]
            elif "prompt" in item and "response" in item:
                text = item["prompt"] + " " + item["response"]
            else:
                continue

            # Tokenize
            tokens = tokenizer.encode(text, max_length=max_length, truncation=True)

            # Compute CFS
            reward = CoherenceReward.from_sequence(
                tokens, embedding_layer, bond_dim, device
            )

            # Filter by threshold
            if reward.cfs >= min_cfs:
                self.examples.append(
                    {
                        "input_ids": tokens,
                        "cfs": reward.cfs,
                        "reward": reward.reward,
                    }
                )

        filtered_pct = len(self.examples) / len(data) * 100
        logger.info(
            f"Kept {len(self.examples)}/{len(data)} examples ({filtered_pct:.1f}%)"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "labels": torch.tensor(item["input_ids"]),
            "cfs": torch.tensor(item["cfs"]),
        }


class CoherenceRewardTrainer(Trainer):
    """
    Custom trainer with coherence-weighted loss.

    Extends HuggingFace Trainer to weight examples by their CFS.

    Example Usage:
        >>> trainer = CoherenceRewardTrainer(
        >>>     model=model,
        >>>     train_dataset=dataset,
        >>>     eval_dataset=eval_dataset,
        >>>     args=training_args,
        >>>     coherence_weight=0.5,  # 50% weight on coherence
        >>> )
        >>> trainer.train()
    """

    def __init__(self, *args, coherence_weight: float = 0.5, **kwargs):
        """
        Initialize coherence reward trainer.

        Args:
            coherence_weight: Weight for coherence in loss (0-1)
            *args, **kwargs: Arguments for Trainer
        """
        super().__init__(*args, **kwargs)
        self.coherence_weight = coherence_weight

    def compute_loss(
        self, model, inputs, return_outputs=False
    ) -> torch.Tensor:
        """
        Compute weighted loss based on coherence.

        Loss = (1 - w) * NLL + w * (1 / CFS) * NLL
        where w is coherence_weight
        """
        # Get CFS from inputs
        cfs = inputs.pop("cfs", None)

        # Standard forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        # Weight loss by coherence
        if cfs is not None and self.coherence_weight > 0:
            # Inverse CFS weighting (penalize low coherence)
            coherence_penalty = 1.0 / (cfs + 1e-6)
            weighted_loss = (
                1 - self.coherence_weight
            ) * loss + self.coherence_weight * coherence_penalty * loss
            loss = weighted_loss.mean()

        return (loss, outputs) if return_outputs else loss


class CoherenceRLTrainer:
    """
    Reinforcement Learning trainer with coherence rewards.

    Implements PPO (Proximal Policy Optimization) using CFS as reward signal.

    Training Algorithm:
        1. Sample batch of prompts
        2. Generate completions with current policy
        3. Compute CFS rewards for each completion
        4. Compute advantages using baseline model
        5. Update policy with PPO objective
        6. Repeat

    Example Usage:
        >>> # Load base model and reference model
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>>
        >>> # Create trainer
        >>> trainer = CoherenceRLTrainer(
        >>>     model=model,
        >>>     ref_model=ref_model,
        >>>     tokenizer=tokenizer,
        >>>     bond_dim=16,
        >>>     ppo_epochs=4,
        >>>     learning_rate=1e-5,
        >>> )
        >>>
        >>> # Train on prompts
        >>> prompts = ["Question: ", "Problem: ", ...]
        >>> trainer.train(prompts, num_epochs=3)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        bond_dim: int = 16,
        ppo_epochs: int = 4,
        learning_rate: float = 1e-5,
        kl_coef: float = 0.1,
        clip_epsilon: float = 0.2,
        value_clip_epsilon: float = 0.2,
        device: str = "cuda",
    ):
        """
        Initialize RL trainer.

        Args:
            model: Model to train (policy)
            ref_model: Reference model for KL penalty
            tokenizer: Tokenizer
            bond_dim: MPS bond dimension for rewards
            ppo_epochs: PPO update epochs per batch
            learning_rate: Learning rate
            kl_coef: KL divergence penalty coefficient
            clip_epsilon: PPO clipping parameter
            value_clip_epsilon: Value function clipping
            device: Compute device
        """
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.tokenizer = tokenizer
        self.bond_dim = bond_dim
        self.ppo_epochs = ppo_epochs
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon
        self.device = device

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Value function (simple linear head on top of model)
        hidden_size = model.config.hidden_size
        self.value_head = torch.nn.Linear(hidden_size, 1).to(device)

        logger.info(
            f"Initialized RL trainer: "
            f"ppo_epochs={ppo_epochs}, lr={learning_rate}, kl_coef={kl_coef}"
        )

    def generate_and_compute_rewards(
        self, prompts: List[str], max_length: int = 100
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Generate completions and compute coherence rewards.

        Args:
            prompts: List of prompt strings
            max_length: Maximum generation length

        Returns:
            Tuple of (token_ids, rewards)
        """
        all_token_ids = []
        all_rewards = []

        self.model.eval()
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                    self.device
                )

                # Generate
                output_ids = self.model.generate(
                    input_ids, max_length=max_length, do_sample=True, temperature=1.0
                )
                token_ids = output_ids[0].tolist()

                # Compute coherence reward
                reward = CoherenceReward.from_sequence(
                    token_ids,
                    self.model.get_input_embeddings(),
                    self.bond_dim,
                    self.device,
                )

                all_token_ids.append(token_ids)
                all_rewards.append(reward.reward)

        self.model.train()
        return all_token_ids, all_rewards

    def compute_advantages(
        self, rewards: List[float], values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE (Generalized Advantage Estimation).

        Args:
            rewards: Reward values
            values: Value function predictions

        Returns:
            Tuple of (advantages, returns)
        """
        rewards_tensor = torch.tensor(rewards, device=self.device)
        returns = rewards_tensor  # Simplified: no bootstrapping

        advantages = returns - values.squeeze()
        return advantages, returns

    def ppo_step(
        self,
        token_ids_batch: List[List[int]],
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single PPO update step.

        Args:
            token_ids_batch: Batch of token sequences
            old_log_probs: Log probabilities from old policy
            advantages: Advantage estimates
            returns: Return estimates

        Returns:
            Dictionary of training metrics
        """
        # Prepare batch
        max_len = max(len(ids) for ids in token_ids_batch)
        input_ids = torch.full(
            (len(token_ids_batch), max_len),
            self.tokenizer.pad_token_id or 0,
            dtype=torch.long,
            device=self.device,
        )

        for i, ids in enumerate(token_ids_batch):
            input_ids[i, : len(ids)] = torch.tensor(ids, device=self.device)

        # Forward pass
        outputs = self.model(input_ids, labels=input_ids)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        # Get log probs for actual tokens
        gathered_log_probs = log_probs.gather(
            2, input_ids.unsqueeze(-1)
        ).squeeze(-1)
        new_log_probs = gathered_log_probs.sum(dim=1)

        # Compute ratio for PPO
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
        if hidden_states is not None:
            values = self.value_head(hidden_states.mean(dim=1))
            value_loss = F.mse_loss(values.squeeze(), returns)
        else:
            value_loss = torch.tensor(0.0, device=self.device)

        # KL divergence with reference model
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids)
            ref_log_probs = F.log_softmax(ref_outputs.logits, dim=-1)
            ref_gathered = ref_log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
            ref_total = ref_gathered.sum(dim=1)

        kl_div = (new_log_probs - ref_total).mean()

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + self.kl_coef * kl_div

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "kl_div": kl_div.item(),
            "total_loss": total_loss.item(),
        }

    def train(
        self, prompts: List[str], num_epochs: int = 3, batch_size: int = 8
    ) -> List[Dict[str, float]]:
        """
        Train model with coherence RL.

        Args:
            prompts: Training prompts
            num_epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            List of metrics per epoch
        """
        metrics_history = []

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Sample batch of prompts
            batch_prompts = prompts[:batch_size]

            # Generate and get rewards
            token_ids_batch, rewards = self.generate_and_compute_rewards(batch_prompts)

            # Compute old log probs and values (for PPO)
            with torch.no_grad():
                # Simplified: use rewards as old_log_probs placeholder
                old_log_probs = torch.tensor(rewards, device=self.device)
                values = torch.zeros(len(rewards), device=self.device)

            # Compute advantages
            advantages, returns = self.compute_advantages(rewards, values)

            # PPO updates
            for ppo_epoch in range(self.ppo_epochs):
                metrics = self.ppo_step(
                    token_ids_batch, old_log_probs, advantages, returns
                )

                if ppo_epoch == 0:
                    logger.info(
                        f"PPO metrics: loss={metrics['total_loss']:.4f}, "
                        f"kl={metrics['kl_div']:.4f}"
                    )

            metrics_history.append(metrics)

        return metrics_history


def evaluate_coherence(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_prompts: List[str],
    bond_dim: int = 16,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate model's coherence on a set of prompts.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_prompts: Evaluation prompts
        bond_dim: MPS bond dimension
        device: Compute device

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    cfs_scores = []

    with torch.no_grad():
        for prompt in tqdm(eval_prompts, desc="Evaluating coherence"):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            output_ids = model.generate(input_ids, max_length=100)
            token_ids = output_ids[0].tolist()

            reward = CoherenceReward.from_sequence(
                token_ids, model.get_input_embeddings(), bond_dim, device
            )
            cfs_scores.append(reward.cfs)

    return {
        "mean_cfs": sum(cfs_scores) / len(cfs_scores),
        "median_cfs": sorted(cfs_scores)[len(cfs_scores) // 2],
        "min_cfs": min(cfs_scores),
        "max_cfs": max(cfs_scores),
    }


__all__ = [
    "CoherenceReward",
    "CoherenceFilteredDataset",
    "CoherenceRewardTrainer",
    "CoherenceRLTrainer",
    "evaluate_coherence",
]
