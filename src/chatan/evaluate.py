"""Evaluation functions for synthetic data quality assessment."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    SEMANTIC_SIMILARITY_AVAILABLE = True
except ImportError:
    SEMANTIC_SIMILARITY_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import sentence_bleu

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class BaseEvaluator(ABC):
    """Base class for evaluation metrics."""

    @abstractmethod
    def compute(self, predictions: List[Any], targets: List[Any], **kwargs) -> float:
        """Compute the evaluation metric."""
        pass


class ExactMatchEvaluator(BaseEvaluator):
    """Exact string match evaluator."""

    def compute(self, predictions: List[str], targets: List[str], **kwargs) -> float:
        """Compute exact match accuracy."""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")

        matches = sum(1 for p, t in zip(predictions, targets) if str(p) == str(t))
        return matches / len(predictions)


class SemanticSimilarityEvaluator(BaseEvaluator):
    """Semantic similarity evaluator using sentence transformers."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        if not SEMANTIC_SIMILARITY_AVAILABLE:
            raise ImportError(
                "Semantic similarity evaluation requires additional dependencies. "
                "Install with: pip install chatan[eval]"
            )
        self.model_name = model
        self._model = None

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    def compute(self, predictions: List[str], targets: List[str], **kwargs) -> float:
        """Compute mean cosine similarity between predictions and targets."""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")

        self._load_model()

        pred_embeddings = self._model.encode(predictions)
        target_embeddings = self._model.encode(targets)

        # Compute cosine similarities
        similarities = []
        for pred_emb, target_emb in zip(pred_embeddings, target_embeddings):
            sim = cosine_similarity([pred_emb], [target_emb])[0][0]
            similarities.append(sim)

        return float(np.mean(similarities))


class BLEUEvaluator(BaseEvaluator):
    """BLEU score evaluator."""

    def compute(self, predictions: List[str], targets: List[str], **kwargs) -> float:
        """Compute BLEU score."""

        if not NLTK_AVAILABLE:
            raise ImportError(
                "BLEU score evaluation requires additional dependencies. "
                "Install with: pip install chatan[eval]"
            )

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")

        scores = []
        for pred, target in zip(predictions, targets):
            pred_tokens = word_tokenize(str(pred).lower())
            target_tokens = [word_tokenize(str(target).lower())]

            if len(pred_tokens) == 0 or len(target_tokens[0]) == 0:
                scores.append(0.0)
            else:
                score = sentence_bleu(target_tokens, pred_tokens)
                scores.append(score)

        return float(np.mean(scores))


class EditDistanceEvaluator(BaseEvaluator):
    """Normalized edit distance evaluator."""

    def compute(self, predictions: List[str], targets: List[str], **kwargs) -> float:
        """Compute normalized edit distance (lower is better, normalized to 0-1)."""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")

        distances = []
        for pred, target in zip(predictions, targets):
            pred_str, target_str = str(pred), str(target)
            distance = self._levenshtein_distance(pred_str, target_str)
            max_len = max(len(pred_str), len(target_str))
            normalized_distance = distance / max_len if max_len > 0 else 0.0
            # Convert to similarity (1 - distance)
            similarity = 1.0 - normalized_distance
            distances.append(similarity)

        return float(np.mean(distances))

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


class LLMJudgeEvaluator(BaseEvaluator):
    """LLM-as-a-judge evaluator."""

    def __init__(self, generator_client, prompt_template: str = None):
        self.generator = generator_client
        self.prompt_template = prompt_template or (
            "Rate the quality of this response on a scale of 1-10:\n\n"
            "Question: {question}\n"
            "Response: {response}\n\n"
            "Rating (1-10):"
        )

    def compute(
        self,
        predictions: List[str],
        targets: List[str],
        questions: Optional[List[str]] = None,
        **kwargs,
    ) -> float:
        """Compute LLM judge scores."""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")

        scores = []
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            question = questions[i] if questions else f"Question {i+1}"

            prompt = self.prompt_template.format(
                question=question, response=pred, target=target
            )

            try:
                response = self.generator._generator.generate(prompt)
                # Extract numeric score from response
                score = self._extract_score(response)
                scores.append(score)
            except Exception:
                # If generation fails, assign neutral score
                scores.append(0.5)

        return float(np.mean(scores))

    def _extract_score(self, response: str) -> float:
        """Extract numeric score from LLM response."""
        import re

        # Look for numbers in the response
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", response)

        if numbers:
            score = float(numbers[0])
            # Normalize to 0-1 scale if it looks like 1-10 scale
            if score > 1:
                score = score / 10.0
            return min(max(score, 0.0), 1.0)

        return 0.5  # Default neutral score


class EvaluationFunction:
    """Evaluation function for use in evaluation schemas."""

    def __init__(
        self, evaluator: BaseEvaluator, column_a: str, column_b: str, **kwargs
    ):
        self.evaluator = evaluator
        self.column_a = column_a
        self.column_b = column_b
        self.kwargs = kwargs

    def __call__(self, data: pd.DataFrame) -> float:
        """Compute evaluation metric on dataset."""
        predictions = data[self.column_a].tolist()
        targets = data[self.column_b].tolist()
        return self.evaluator.compute(predictions, targets, **self.kwargs)


class DatasetEvaluator:
    """Dataset-specific evaluator for method chaining."""

    def __init__(self, dataset):
        self.dataset = dataset

    def exact_match(self, column_a: str, column_b: str, **kwargs) -> EvaluationFunction:
        """Create exact match evaluation function."""
        return EvaluationFunction(ExactMatchEvaluator(), column_a, column_b, **kwargs)

    def semantic_similarity(
        self, column_a: str, column_b: str, model: str = "all-MiniLM-L6-v2", **kwargs
    ) -> EvaluationFunction:
        """Create semantic similarity evaluation function."""
        evaluator = SemanticSimilarityEvaluator(model)
        return EvaluationFunction(evaluator, column_a, column_b, **kwargs)

    def bleu_score(self, column_a: str, column_b: str, **kwargs) -> EvaluationFunction:
        """Create BLEU score evaluation function."""
        return EvaluationFunction(BLEUEvaluator(), column_a, column_b, **kwargs)

    def edit_distance(
        self, column_a: str, column_b: str, **kwargs
    ) -> EvaluationFunction:
        """Create edit distance evaluation function."""
        return EvaluationFunction(EditDistanceEvaluator(), column_a, column_b, **kwargs)

    def llm_judge(
        self,
        column_a: str,
        column_b: str,
        generator_client,
        prompt_template: str = None,
        **kwargs,
    ) -> EvaluationFunction:
        """Create LLM judge evaluation function."""
        evaluator = LLMJudgeEvaluator(generator_client, prompt_template)
        return EvaluationFunction(evaluator, column_a, column_b, **kwargs)


# Standalone evaluation functions for schema use
class EvalNamespace:
    """Namespace for evaluation functions."""

    @staticmethod
    def exact_match(column_a: str, column_b: str, **kwargs) -> Callable:
        """Exact match evaluation for use in dataset schemas."""

        def eval_func(context: Dict[str, Any]) -> float:
            pred, target = context[column_a], context[column_b]
            return 1.0 if str(pred) == str(target) else 0.0

        return eval_func

    @staticmethod
    def semantic_similarity(
        column_a: str, column_b: str, model: str = "all-MiniLM-L6-v2"
    ) -> Callable:
        """Semantic similarity evaluation for use in dataset schemas."""
        evaluator = SemanticSimilarityEvaluator(model)

        def eval_func(context: Dict[str, Any]) -> float:
            pred, target = context[column_a], context[column_b]
            return evaluator.compute([str(pred)], [str(target)])

        return eval_func

    @staticmethod
    def bleu_score(column_a: str, column_b: str) -> Callable:
        """BLEU score evaluation for use in dataset schemas."""
        evaluator = BLEUEvaluator()

        def eval_func(context: Dict[str, Any]) -> float:
            pred, target = context[column_a], context[column_b]
            return evaluator.compute([str(pred)], [str(target)])

        return eval_func

    @staticmethod
    def edit_distance(column_a: str, column_b: str) -> Callable:
        """Edit distance evaluation for use in dataset schemas."""
        evaluator = EditDistanceEvaluator()

        def eval_func(context: Dict[str, Any]) -> float:
            pred, target = context[column_a], context[column_b]
            return evaluator.compute([str(pred)], [str(target)])

        return eval_func


def evaluate(
    eval_schema: Dict[str, Union[EvaluationFunction, Callable]],
) -> Dict[str, float]:
    """
    Evaluate multiple metrics across datasets.

    Args:
        eval_schema: Dictionary mapping metric names to evaluation functions or callables

    Returns:
        Dictionary of metric names to computed scores
    """
    results = {}
    for name, eval_func in eval_schema.items():
        if isinstance(eval_func, EvaluationFunction):
            # This shouldn't happen in normal usage - EvaluationFunction needs a dataset
            raise ValueError(
                f"EvaluationFunction for '{name}' requires dataset context"
            )
        elif callable(eval_func):
            # Should be a callable that returns the result
            results[name] = eval_func()
        else:
            raise ValueError(
                f"Invalid evaluation function type for '{name}': {type(eval_func)}"
            )
    return results


# Export the eval namespace
eval = EvalNamespace()
