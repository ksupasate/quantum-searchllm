"""
Coherence Metrics Evaluation

Implements metrics from the paper (Table 2):
- Negation Invariance Rate
- Transitivity Violation Rate

These metrics directly measure logical consistency,
independent of final answer accuracy.
"""

import sys
from pathlib import Path

# Add parent directory to path to import tnad
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


class CoherenceEvaluator:
    """
    Evaluates logical coherence of LLM generations.

    Implements metrics from paper Section 5.4:
    - Negation Invariance: Response consistency under logical negation
    - Transitivity: A>B, B>C => A>C
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        generator: Any,  # Can be FGBS, GreedyDecoder, etc.
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = generator

    def evaluate_negation_invariance(
        self,
        test_cases: List[Dict[str, str]],
        max_length: int = 100,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate negation invariance.

        Test cases should have format:
        {
            'positive': "Is the sky blue?",
            'negative': "Is the sky not blue?",
            'expected_relation': 'opposite'  # or 'same' for double negation
        }

        A coherent model should give opposite answers for logically negated prompts.

        Args:
            test_cases: List of positive/negative statement pairs
            max_length: Maximum generation length
            show_progress: Show progress bar

        Returns:
            Dictionary with invariance metrics
        """
        violations = 0
        results = []

        iterator = test_cases
        if show_progress:
            iterator = tqdm(test_cases, desc="Negation Invariance")

        for case in iterator:
            # Generate for positive statement
            pos_result = self.generator.generate(
                case['positive'],
                max_length=max_length,
                show_progress=False,
            )

            # Generate for negative statement
            neg_result = self.generator.generate(
                case['negative'],
                max_length=max_length,
                show_progress=False,
            )

            # Extract yes/no or determine stance
            pos_answer = self._extract_binary_stance(pos_result['text'])
            neg_answer = self._extract_binary_stance(neg_result['text'])

            # Check if answers are appropriately opposite
            expected_relation = case.get('expected_relation', 'opposite')

            if expected_relation == 'opposite':
                is_consistent = (pos_answer != neg_answer)
            else:  # 'same' for double negation
                is_consistent = (pos_answer == neg_answer)

            if not is_consistent:
                violations += 1

            results.append({
                'positive_prompt': case['positive'],
                'negative_prompt': case['negative'],
                'positive_answer': pos_answer,
                'negative_answer': neg_answer,
                'consistent': is_consistent,
                'positive_text': pos_result['text'][:200],
                'negative_text': neg_result['text'][:200],
            })

        violation_rate = violations / len(test_cases) if test_cases else 0.0

        return {
            'violation_rate': violation_rate,
            'total_cases': len(test_cases),
            'violations': violations,
            'consistency_rate': 1.0 - violation_rate,
            'detailed_results': results,
        }

    def evaluate_transitivity(
        self,
        test_cases: List[Dict[str, str]],
        max_length: int = 100,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate transitivity violations.

        Test cases should have format:
        {
            'premise1': "A is greater than B",
            'premise2': "B is greater than C",
            'question': "Is A greater than C?",
            'expected_answer': True
        }

        A coherent model should respect transitive properties.

        Args:
            test_cases: List of transitivity test cases
            max_length: Maximum generation length
            show_progress: Show progress bar

        Returns:
            Dictionary with transitivity metrics
        """
        violations = 0
        results = []

        iterator = test_cases
        if show_progress:
            iterator = tqdm(test_cases, desc="Transitivity")

        for case in iterator:
            # Construct prompt with premises
            prompt = f"{case['premise1']} {case['premise2']} {case['question']}"

            # Generate answer
            result = self.generator.generate(
                prompt,
                max_length=max_length,
                show_progress=False,
            )

            # Extract answer
            predicted_answer = self._extract_binary_stance(result['text'])
            expected_answer = case['expected_answer']

            # Check if transitivity is violated
            is_correct = (predicted_answer == expected_answer)

            if not is_correct:
                violations += 1

            results.append({
                'premise1': case['premise1'],
                'premise2': case['premise2'],
                'question': case['question'],
                'expected_answer': expected_answer,
                'predicted_answer': predicted_answer,
                'correct': is_correct,
                'generated_text': result['text'][:200],
            })

        violation_rate = violations / len(test_cases) if test_cases else 0.0

        return {
            'violation_rate': violation_rate,
            'total_cases': len(test_cases),
            'violations': violations,
            'accuracy': 1.0 - violation_rate,
            'detailed_results': results,
        }

    def _extract_binary_stance(self, text: str) -> Optional[bool]:
        """
        Extract binary stance (yes/no, true/false, positive/negative) from text.

        Returns:
            True for positive stance, False for negative, None if unclear
        """
        text_lower = text.lower()

        # Look for explicit yes/no
        yes_patterns = [
            r'\byes\b', r'\btrue\b', r'\bcorrect\b', r'\bagree\b',
            r'\bpositive\b', r'\baffirmative\b'
        ]
        no_patterns = [
            r'\bno\b', r'\bfalse\b', r'\bincorrect\b', r'\bdisagree\b',
            r'\bnegative\b', r'\bnot\b.*\btrue\b'
        ]

        yes_count = sum(len(re.findall(p, text_lower)) for p in yes_patterns)
        no_count = sum(len(re.findall(p, text_lower)) for p in no_patterns)

        if yes_count > no_count:
            return True
        elif no_count > yes_count:
            return False
        else:
            return None


def generate_negation_test_cases() -> List[Dict[str, str]]:
    """
    Generate standard negation invariance test cases.

    Returns test cases for common reasoning patterns.
    """
    test_cases = [
        # Basic facts
        {
            'positive': 'The sky is blue. Is the sky blue?',
            'negative': 'The sky is blue. Is the sky not blue?',
            'expected_relation': 'opposite'
        },
        {
            'positive': 'Water boils at 100°C. Does water boil at 100°C?',
            'negative': 'Water boils at 100°C. Does water not boil at 100°C?',
            'expected_relation': 'opposite'
        },
        # Logical statements
        {
            'positive': 'If A then B. A is true. Is B true?',
            'negative': 'If A then B. A is true. Is B not true?',
            'expected_relation': 'opposite'
        },
        {
            'positive': 'All cats are mammals. Felix is a cat. Is Felix a mammal?',
            'negative': 'All cats are mammals. Felix is a cat. Is Felix not a mammal?',
            'expected_relation': 'opposite'
        },
        # Comparisons
        {
            'positive': '5 is greater than 3. Is 5 greater than 3?',
            'negative': '5 is greater than 3. Is 5 not greater than 3?',
            'expected_relation': 'opposite'
        },
        # Math
        {
            'positive': '2 + 2 = 4. Is 2 + 2 equal to 4?',
            'negative': '2 + 2 = 4. Is 2 + 2 not equal to 4?',
            'expected_relation': 'opposite'
        },
    ]

    return test_cases


def generate_transitivity_test_cases() -> List[Dict[str, str]]:
    """
    Generate standard transitivity test cases.

    Returns test cases for transitive reasoning.
    """
    test_cases = [
        # Greater than
        {
            'premise1': 'A is greater than B.',
            'premise2': 'B is greater than C.',
            'question': 'Is A greater than C?',
            'expected_answer': True
        },
        {
            'premise1': 'John is taller than Mary.',
            'premise2': 'Mary is taller than Susan.',
            'question': 'Is John taller than Susan?',
            'expected_answer': True
        },
        # Numbers
        {
            'premise1': '10 > 5.',
            'premise2': '5 > 2.',
            'question': 'Is 10 > 2?',
            'expected_answer': True
        },
        # Logical implication
        {
            'premise1': 'If P then Q.',
            'premise2': 'If Q then R.',
            'question': 'If P is true, is R true?',
            'expected_answer': True
        },
        # Subset relations
        {
            'premise1': 'All dogs are mammals.',
            'premise2': 'All mammals are animals.',
            'question': 'Are all dogs animals?',
            'expected_answer': True
        },
        # Temporal ordering
        {
            'premise1': 'Event A happened before Event B.',
            'premise2': 'Event B happened before Event C.',
            'question': 'Did Event A happen before Event C?',
            'expected_answer': True
        },
    ]

    return test_cases


def run_coherence_evaluation(
    generator: Any,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_negation_cases: int = 20,
    num_transitivity_cases: int = 20,
) -> Dict[str, Any]:
    """
    Run comprehensive coherence evaluation.

    Args:
        generator: Generation method (FGBS, greedy, etc.)
        model: LLM model
        tokenizer: Tokenizer
        num_negation_cases: Number of negation test cases
        num_transitivity_cases: Number of transitivity test cases

    Returns:
        Dictionary with all coherence metrics
    """
    evaluator = CoherenceEvaluator(model, tokenizer, generator)

    # Generate test cases
    negation_cases = generate_negation_test_cases()
    if len(negation_cases) > num_negation_cases:
        negation_cases = negation_cases[:num_negation_cases]

    transitivity_cases = generate_transitivity_test_cases()
    if len(transitivity_cases) > num_transitivity_cases:
        transitivity_cases = transitivity_cases[:num_transitivity_cases]

    # Evaluate
    print("Evaluating Negation Invariance...")
    negation_results = evaluator.evaluate_negation_invariance(negation_cases)

    print("Evaluating Transitivity...")
    transitivity_results = evaluator.evaluate_transitivity(transitivity_cases)

    return {
        'negation_invariance': negation_results,
        'transitivity': transitivity_results,
        'summary': {
            'negation_violation_rate': negation_results['violation_rate'],
            'transitivity_violation_rate': transitivity_results['violation_rate'],
            'overall_coherence_score': 1.0 - (
                (negation_results['violation_rate'] +
                 transitivity_results['violation_rate']) / 2
            ),
        }
    }
