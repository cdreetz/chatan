"""Comprehensive tests for evaluate module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from chatan.evaluate import (
    ExactMatchEvaluator,
    EditDistanceEvaluator,
    LLMJudgeEvaluator,
    EvaluationFunction,
    DatasetEvaluator,
    evaluate,
    eval
)
from chatan.dataset import dataset
from chatan.sampler import ChoiceSampler

# Check for optional dependencies at module level
SEMANTIC_SIMILARITY_AVAILABLE = False
BLEU_AVAILABLE = False

try:
    import sentence_transformers
    import sklearn.metrics.pairwise
    SEMANTIC_SIMILARITY_AVAILABLE = True
except ImportError:
    pass

try:
    import nltk
    BLEU_AVAILABLE = True
except ImportError:
    pass


class TestExactMatchEvaluator:
    """Test ExactMatchEvaluator functionality."""

    def test_perfect_match(self):
        """Test perfect exact match."""
        evaluator = ExactMatchEvaluator()
        predictions = ["hello", "world", "test"]
        targets = ["hello", "world", "test"]
        
        score = evaluator.compute(predictions, targets)
        assert score == 1.0

    def test_no_match(self):
        """Test no exact matches."""
        evaluator = ExactMatchEvaluator()
        predictions = ["hello", "world", "test"]
        targets = ["hi", "earth", "exam"]
        
        score = evaluator.compute(predictions, targets)
        assert score == 0.0

    def test_partial_match(self):
        """Test partial matches."""
        evaluator = ExactMatchEvaluator()
        predictions = ["hello", "world", "test"]
        targets = ["hello", "earth", "test"]
        
        score = evaluator.compute(predictions, targets)
        assert score == 2/3

    def test_case_sensitive(self):
        """Test that matching is case sensitive."""
        evaluator = ExactMatchEvaluator()
        predictions = ["Hello", "WORLD"]
        targets = ["hello", "world"]
        
        score = evaluator.compute(predictions, targets)
        assert score == 0.0

    def test_type_conversion(self):
        """Test that values are converted to strings."""
        evaluator = ExactMatchEvaluator()
        predictions = [1, 2.5, True]
        targets = ["1", "2.5", "True"]
        
        score = evaluator.compute(predictions, targets)
        assert score == 1.0

    def test_length_mismatch_error(self):
        """Test error when lists have different lengths."""
        evaluator = ExactMatchEvaluator()
        predictions = ["a", "b"]
        targets = ["a"]
        
        with pytest.raises(ValueError, match="same length"):
            evaluator.compute(predictions, targets)

    def test_empty_lists(self):
        """Test with empty lists."""
        evaluator = ExactMatchEvaluator()
        predictions = []
        targets = []
        
        with pytest.raises(ZeroDivisionError):
            evaluator.compute(predictions, targets)


@pytest.mark.skipif(not SEMANTIC_SIMILARITY_AVAILABLE, reason="sentence-transformers not available")
class TestSemanticSimilarityEvaluator:
    """Test SemanticSimilarityEvaluator functionality."""

    #def test_semantic_similarity_basic(self):
    #    """Test basic semantic similarity computation."""
    #    from chatan.evaluate import SemanticSimilarityEvaluator
    #    
    #    with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
    #        with patch('sklearn.metrics.pairwise.cosine_similarity') as mock_cosine:
    #            # Mock the transformer
    #            mock_model = Mock()
    #            # Mock encode to return different embeddings for predictions and targets
    #            mock_model.encode.side_effect = [
    #                np.array([[1, 0], [0, 1]]),  # predictions embeddings
    #                np.array([[0.8, 0.6], [0.9, 0.4]])  # targets embeddings
    #            ]
    #            mock_transformer.return_value = mock_model
    #            
    #            # Mock cosine similarity to return specific values for each call
    #            mock_cosine.side_effect = [
    #                np.array([[0.8]]),  # First call
    #                np.array([[0.9]])   # Second call
    #            ]
    #            
    #            evaluator = SemanticSimilarityEvaluator()
    #            predictions = ["hello world", "good morning"]
    #            targets = ["hi earth", "good evening"]
    #            
    #            score = evaluator.compute(predictions, targets)
    #            assert score == pytest.approx(0.85)  # (0.8 + 0.9) / 2

    def test_missing_dependency_error(self):
        """Test error when sentence-transformers not installed."""
        from chatan.evaluate import SemanticSimilarityEvaluator
        
        with patch('chatan.evaluate.SEMANTIC_SIMILARITY_AVAILABLE', False):
            with pytest.raises(ImportError, match="Semantic similarity evaluation requires additional dependencies"):
                evaluator = SemanticSimilarityEvaluator()

    def test_custom_model(self):
        """Test with custom model name."""
        from chatan.evaluate import SemanticSimilarityEvaluator
        
        with patch('sentence_transformers.SentenceTransformer'):
            evaluator = SemanticSimilarityEvaluator(model="custom-model")
            assert evaluator.model_name == "custom-model"

    def test_lazy_loading(self):
        """Test that model is loaded lazily."""
        from chatan.evaluate import SemanticSimilarityEvaluator
        
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
            evaluator = SemanticSimilarityEvaluator()
            assert evaluator._model is None
            
            # Mock the rest of the computation
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[1, 0]])
            mock_transformer.return_value = mock_model
            
            with patch('sklearn.metrics.pairwise.cosine_similarity', return_value=[[0.8]]):
                evaluator.compute(["test"], ["test"])
            
            assert evaluator._model is not None


@pytest.mark.skipif(not BLEU_AVAILABLE, reason="NLTK not available")
class TestBLEUEvaluator:
    """Test BLEUEvaluator functionality."""

    def test_bleu_score_basic(self):
        """Test basic BLEU score computation."""
        from chatan.evaluate import BLEUEvaluator
        
        with patch('chatan.evaluate.word_tokenize') as mock_tokenize:
            with patch('chatan.evaluate.sentence_bleu') as mock_bleu:
                with patch('nltk.download') as mock_download:
                    with patch('nltk.data.find') as mock_find:
                        # Mock NLTK components
                        mock_find.return_value = "mock_path"  # Don't trigger download
                        mock_tokenize.side_effect = [
                            ["hello", "world"], ["hello", "world"],
                            ["good", "morning"], ["good", "morning"]
                        ]
                        mock_bleu.side_effect = [0.8, 0.9]
                        
                        evaluator = BLEUEvaluator()
                        predictions = ["hello world", "good morning"]
                        targets = ["hello world", "good morning"]
                        
                        score = evaluator.compute(predictions, targets)
                        assert score == pytest.approx(0.85)  # (0.8 + 0.9) / 2
                        
                        # Verify the mocks were called correctly
                        assert mock_bleu.call_count == 2

    def test_missing_nltk_error(self):
        """Test error when NLTK not installed."""
        from chatan.evaluate import BLEUEvaluator
        
        with patch('chatan.evaluate.NLTK_AVAILABLE', False):
            with pytest.raises(ImportError, match="BLEU score evaluation requires additional dependencies"):
                evaluator = BLEUEvaluator()
                # Force the error by calling compute
                evaluator.compute(["test"], ["test"])

    def test_empty_tokens_handling(self):
        """Test handling of empty tokenization."""
        from chatan.evaluate import BLEUEvaluator
        
        with patch('chatan.evaluate.word_tokenize') as mock_tokenize:
            with patch('nltk.translate.bleu_score.sentence_bleu') as mock_bleu:
                with patch('nltk.download') as mock_download:
                    with patch('nltk.data.find') as mock_find:
                        mock_find.return_value = "mock_path"  # Don't trigger download
                        mock_tokenize.side_effect = [[], ["word"], ["word"], []]
                        
                        evaluator = BLEUEvaluator()
                        predictions = ["", "word"]
                        targets = ["word", ""]
                        
                        score = evaluator.compute(predictions, targets)
                        assert score == 0.0  # Both should get 0.0 for empty tokens


class TestEditDistanceEvaluator:
    """Test EditDistanceEvaluator functionality."""

    def test_identical_strings(self):
        """Test identical strings have similarity 1.0."""
        evaluator = EditDistanceEvaluator()
        predictions = ["hello", "world"]
        targets = ["hello", "world"]
        
        score = evaluator.compute(predictions, targets)
        assert score == 1.0

    def test_completely_different_strings(self):
        """Test completely different strings."""
        evaluator = EditDistanceEvaluator()
        predictions = ["abc"]
        targets = ["xyz"]
        
        score = evaluator.compute(predictions, targets)
        assert score == 0.0  # All characters need to be changed

    def test_partial_similarity(self):
        """Test strings with partial similarity."""
        evaluator = EditDistanceEvaluator()
        predictions = ["hello"]
        targets = ["hallo"]  # 1 character different out of 5
        
        score = evaluator.compute(predictions, targets)
        assert score == 0.8  # 4/5 similarity

    def test_empty_strings(self):
        """Test empty string handling."""
        evaluator = EditDistanceEvaluator()
        predictions = ["", "hello"]
        targets = ["hello", ""]
        
        score = evaluator.compute(predictions, targets)
        # First: empty to "hello" = 0.0, Second: "hello" to empty = 0.0
        assert score == 0.0

    def test_levenshtein_distance_calculation(self):
        """Test internal Levenshtein distance calculation."""
        evaluator = EditDistanceEvaluator()
        
        # "cat" to "bat" = 1 substitution
        distance = evaluator._levenshtein_distance("cat", "bat")
        assert distance == 1
        
        # "kitten" to "sitting" = 3 operations
        distance = evaluator._levenshtein_distance("kitten", "sitting")
        assert distance == 3


class TestLLMJudgeEvaluator:
    """Test LLMJudgeEvaluator functionality."""

    def test_llm_judge_basic(self):
        """Test basic LLM judge evaluation."""
        mock_generator = Mock()
        mock_generator._generator.generate.side_effect = ["8", "7"]
        
        evaluator = LLMJudgeEvaluator(mock_generator)
        predictions = ["Good response", "OK response"]
        targets = ["Target 1", "Target 2"]
        questions = ["Question 1", "Question 2"]
        
        score = evaluator.compute(predictions, targets, questions=questions)
        assert score == 0.75  # (0.8 + 0.7) / 2

    def test_score_extraction(self):
        """Test score extraction from LLM responses."""
        mock_generator = Mock()
        evaluator = LLMJudgeEvaluator(mock_generator)
        
        # Test various response formats
        assert evaluator._extract_score("The score is 8 out of 10") == 0.8
        assert evaluator._extract_score("Rating: 7.5") == 0.75
        assert evaluator._extract_score("9") == 0.9
        assert evaluator._extract_score("No numbers here") == 0.5

    def test_generation_failure_handling(self):
        """Test handling of generation failures."""
        mock_generator = Mock()
        mock_generator._generator.generate.side_effect = Exception("API Error")
        
        evaluator = LLMJudgeEvaluator(mock_generator)
        predictions = ["Test"]
        targets = ["Target"]
        
        score = evaluator.compute(predictions, targets)
        assert score == 0.5  # Should default to neutral score

    def test_custom_prompt_template(self):
        """Test custom prompt template."""
        mock_generator = Mock()
        mock_generator._generator.generate.return_value = "5"
        
        custom_template = "Rate this: {response} vs {target}"
        evaluator = LLMJudgeEvaluator(mock_generator, custom_template)
        
        evaluator.compute(["response"], ["target"])
        
        # Check that custom template was used
        call_args = mock_generator._generator.generate.call_args[0][0]
        assert "Rate this: response vs target" in call_args


class TestEvaluationFunction:
    """Test EvaluationFunction wrapper."""

    def test_evaluation_function_call(self):
        """Test EvaluationFunction execution."""
        mock_evaluator = Mock()
        mock_evaluator.compute.return_value = 0.85
        
        data = pd.DataFrame({
            "pred": ["hello", "world"],
            "target": ["hi", "earth"]
        })
        
        eval_func = EvaluationFunction(mock_evaluator, "pred", "target")
        result = eval_func(data)
        
        assert result == 0.85
        mock_evaluator.compute.assert_called_once_with(["hello", "world"], ["hi", "earth"])

    def test_evaluation_function_with_kwargs(self):
        """Test EvaluationFunction with additional kwargs."""
        mock_evaluator = Mock()
        mock_evaluator.compute.return_value = 0.9
        
        data = pd.DataFrame({"a": ["test"], "b": ["test"]})
        
        eval_func = EvaluationFunction(mock_evaluator, "a", "b", custom_param=True)
        eval_func(data)
        
        mock_evaluator.compute.assert_called_once_with(["test"], ["test"], custom_param=True)


class TestDatasetEvaluator:
    """Test DatasetEvaluator functionality."""

    def test_exact_match_creation(self):
        """Test exact match evaluation function creation."""
        mock_dataset = Mock()
        evaluator = DatasetEvaluator(mock_dataset)
        
        eval_func = evaluator.exact_match("col_a", "col_b")
        assert isinstance(eval_func, EvaluationFunction)
        assert isinstance(eval_func.evaluator, ExactMatchEvaluator)

    @pytest.mark.skipif(not SEMANTIC_SIMILARITY_AVAILABLE, reason="sentence-transformers not available")
    def test_semantic_similarity_creation(self):
        """Test semantic similarity evaluation function creation."""
        
        with patch('chatan.evaluate.SemanticSimilarityEvaluator') as mock_evaluator_class:
            mock_dataset = Mock()
            evaluator = DatasetEvaluator(mock_dataset)
            
            eval_func = evaluator.semantic_similarity("col_a", "col_b", model="custom-model")
            assert isinstance(eval_func, EvaluationFunction)
            mock_evaluator_class.assert_called_once_with("custom-model")

    @pytest.mark.skipif(not BLEU_AVAILABLE, reason="NLTK not available")
    def test_bleu_score_creation(self):
        """Test BLEU score evaluation function creation."""
        from chatan.evaluate import BLEUEvaluator
        
        mock_dataset = Mock()
        evaluator = DatasetEvaluator(mock_dataset)
        
        eval_func = evaluator.bleu_score("col_a", "col_b")
        assert isinstance(eval_func, EvaluationFunction)
        assert isinstance(eval_func.evaluator, BLEUEvaluator)

    def test_llm_judge_creation(self):
        """Test LLM judge evaluation function creation."""
        mock_dataset = Mock()
        mock_generator = Mock()
        evaluator = DatasetEvaluator(mock_dataset)
        
        eval_func = evaluator.llm_judge("col_a", "col_b", mock_generator)
        assert isinstance(eval_func, EvaluationFunction)
        assert isinstance(eval_func.evaluator, LLMJudgeEvaluator)


class TestEvalNamespace:
    """Test eval namespace functions."""

    def test_exact_match_schema_function(self):
        """Test exact match function for schema use."""
        eval_func = eval.exact_match("pred", "target")
        
        # Test matching case
        result = eval_func({"pred": "hello", "target": "hello"})
        assert result == 1.0
        
        # Test non-matching case
        result = eval_func({"pred": "hello", "target": "hi"})
        assert result == 0.0

    @pytest.mark.skipif(not SEMANTIC_SIMILARITY_AVAILABLE, reason="sentence-transformers not available")
    def test_semantic_similarity_schema_function(self):
        """Test semantic similarity function for schema use."""
        
        with patch('chatan.evaluate.SemanticSimilarityEvaluator') as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_evaluator.compute.return_value = 0.85
            mock_evaluator_class.return_value = mock_evaluator
            
            eval_func = eval.semantic_similarity("pred", "target")
            result = eval_func({"pred": "hello", "target": "hi"})
            
            assert result == 0.85
            mock_evaluator.compute.assert_called_once_with(["hello"], ["hi"])

    @pytest.mark.skipif(not BLEU_AVAILABLE, reason="NLTK not available")
    def test_bleu_score_schema_function(self):
        """Test BLEU score function for schema use."""
        
        with patch('chatan.evaluate.BLEUEvaluator') as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_evaluator.compute.return_value = 0.7
            mock_evaluator_class.return_value = mock_evaluator
            
            eval_func = eval.bleu_score("pred", "target")
            result = eval_func({"pred": "test", "target": "test"})
            
            assert result == 0.7

    def test_edit_distance_schema_function(self):
        """Test edit distance function for schema use."""
        with patch('chatan.evaluate.EditDistanceEvaluator') as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_evaluator.compute.return_value = 0.9
            mock_evaluator_class.return_value = mock_evaluator
            
            eval_func = eval.edit_distance("pred", "target")
            result = eval_func({"pred": "hello", "target": "hallo"})
            
            assert result == 0.9


class TestTopLevelEvaluate:
    """Test top-level evaluate function."""

    def test_evaluate_function_basic(self):
        """Test basic evaluate function usage."""
        mock_eval_func1 = Mock()
        mock_eval_func1.return_value = 0.8
        
        mock_eval_func2 = Mock()
        mock_eval_func2.return_value = 0.9
        
        eval_schema = {
            "metric1": mock_eval_func1,
            "metric2": mock_eval_func2
        }
        
        results = evaluate(eval_schema)
        
        assert results == {"metric1": 0.8, "metric2": 0.9}
        mock_eval_func1.assert_called_once()
        mock_eval_func2.assert_called_once()

    def test_evaluate_with_dataset_evaluator(self):
        """Test evaluate function with DatasetEvaluator objects."""
        # Create mock evaluation function that has dataset attribute
        mock_eval_func = Mock()
        mock_eval_func.dataset._data = pd.DataFrame({"a": [1], "b": [2]})
        mock_eval_func.return_value = 0.75
        
        eval_schema = {"test_metric": mock_eval_func}
        results = evaluate(eval_schema)
        
        assert results == {"test_metric": 0.75}


@pytest.mark.asyncio
class TestDatasetIntegration:
    """Test integration with Dataset class."""

    async def test_dataset_eval_property(self):
        """Test Dataset.eval property."""
        schema = {"col": ChoiceSampler(["A", "B"])}
        ds = dataset(schema, n=5)
        await ds.generate()

        # Should be able to access eval property
        evaluator = ds.eval
        assert isinstance(evaluator, DatasetEvaluator)

    def test_dataset_eval_property_no_data_error(self):
        """Test Dataset.eval property error when no data."""
        schema = {"col": ChoiceSampler(["A"])}
        ds = dataset(schema, n=5)

        with pytest.raises(ValueError, match="Dataset must be generated"):
            _ = ds.eval

    async def test_dataset_evaluate_method(self):
        """Test Dataset.evaluate method."""
        schema = {
            "pred": ChoiceSampler(["hello", "world"]),
            "target": ChoiceSampler(["hello", "world"])
        }
        ds = dataset(schema, n=10)
        await ds.generate()

        # Create mock evaluation functions
        mock_eval1 = Mock()
        mock_eval1.return_value = 0.8
        mock_eval2 = Mock()
        mock_eval2.return_value = 0.9

        eval_schema = {
            "similarity": mock_eval1,
            "exact_match": mock_eval2
        }

        results = ds.evaluate(eval_schema)

        assert results == {"similarity": 0.8, "exact_match": 0.9}
        assert ds._data is not None

    def test_dataset_evaluate_requires_generation(self):
        """Test that Dataset.evaluate requires data to be generated first."""
        schema = {"col": ChoiceSampler(["A"])}
        ds = dataset(schema, n=5)

        assert ds._data is None

        mock_eval = Mock()
        mock_eval.return_value = 1.0

        with pytest.raises(ValueError, match="must be generated"):
            ds.evaluate({"test": mock_eval})


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    async def test_schema_based_evaluation(self):
        """Test schema-based evaluation (Option A)."""
        with patch('chatan.evaluate.ExactMatchEvaluator.compute', return_value=0.8):
            schema = {
                "response": ChoiceSampler(["answer1", "answer2"]),
                "target": ChoiceSampler(["answer1", "answer2"]),
                "exact_match": eval.exact_match("response", "target")
            }

            ds = dataset(schema, n=5)
            df = await ds.generate()

            # Should have evaluation column
            assert "exact_match" in df.columns
            # All values should be either 0.0 or 1.0 (exact match results)
            assert all(val in [0.0, 1.0] for val in df["exact_match"])

    async def test_aggregate_evaluation(self):
        """Test aggregate evaluation (Option B)."""
        schema = {
            "response": ChoiceSampler(["hello", "world"]),
            "target": ChoiceSampler(["hello", "world"])
        }

        ds = dataset(schema, n=10)
        df = await ds.generate()

        # Mock evaluation function
        with patch('chatan.evaluate.ExactMatchEvaluator.compute', return_value=0.7):
            eval_func = ds.eval.exact_match("response", "target")
            result = eval_func(df)
            assert result == 0.7

    async def test_cross_dataset_comparison(self):
        """Test comparing multiple datasets using Dataset.evaluate."""
        schema1 = {"response": ChoiceSampler(["good", "great"])}
        schema2 = {"response": ChoiceSampler(["bad", "worse"])}

        ds1 = dataset(schema1, n=5)
        ds2 = dataset(schema2, n=5)

        # Generate both datasets
        df1 = await ds1.generate()
        df2 = await ds2.generate()

        # Use Dataset.evaluate for individual dataset evaluation
        with patch('chatan.evaluate.ExactMatchEvaluator.compute', return_value=1.0):
            eval_func = ds1.eval.exact_match("response", "response")
            results1 = ds1.evaluate({"self_match": eval_func})
            assert results1["self_match"] == 1.0

        # For cross-dataset comparison, create wrapper functions
        def eval_ds1():
            with patch('chatan.evaluate.ExactMatchEvaluator.compute', return_value=0.9):
                eval_func = ds1.eval.exact_match("response", "response")
                return eval_func(df1)

        def eval_ds2():
            with patch('chatan.evaluate.ExactMatchEvaluator.compute', return_value=0.3):
                eval_func = ds2.eval.exact_match("response", "response")
                return eval_func(df2)

        eval_schema = {
            "ds1_quality": eval_ds1,
            "ds2_quality": eval_ds2
        }

        results = evaluate(eval_schema)

        assert results["ds1_quality"] == 0.9
        assert results["ds2_quality"] == 0.3

    async def test_error_handling_in_evaluation(self):
        """Test error handling during evaluation."""
        schema = {"col": ChoiceSampler(["test"])}
        ds = dataset(schema, n=5)
        await ds.generate()

        # Mock evaluator that raises exception
        mock_eval = Mock()
        mock_eval.side_effect = Exception("Evaluation failed")

        eval_schema = {"failing_metric": mock_eval}

        with pytest.raises(Exception, match="Evaluation failed"):
            ds.evaluate(eval_schema)
