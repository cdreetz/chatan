"""Comprehensive tests for sampler module."""

import pytest
import uuid
from datetime import datetime
from unittest.mock import patch
import pandas as pd
from datasets import Dataset as HFDataset

from chatan.sampler import (
    ChoiceSampler,
    WeightedSampler,
    UUIDSampler,
    DatetimeSampler,
    RangeSampler,
    DatasetSampler,
    sample
)


class TestChoiceSampler:
    """Test ChoiceSampler functionality."""

    def test_list_choices(self):
        """Test sampling from a list."""
        choices = ["A", "B", "C"]
        sampler = ChoiceSampler(choices)
        
        # Test multiple samples are from choices
        results = [sampler() for _ in range(100)]
        assert all(result in choices for result in results)
        
        # Test we get variety (probabilistically should hit all choices)
        unique_results = set(results)
        assert len(unique_results) >= 2  # Should get at least 2 different values

    def test_dict_choices_with_weights(self):
        """Test sampling from dict with weights."""
        choices = {"A": 0.1, "B": 0.9}  # B should be much more frequent
        sampler = ChoiceSampler(choices)
        
        results = [sampler() for _ in range(1000)]
        count_b = sum(1 for r in results if r == "B")
        
        # B should appear roughly 90% of the time (allow some variance)
        assert count_b > 800  # Should be around 900, but allow for randomness
        assert all(result in ["A", "B"] for result in results)

    def test_single_choice(self):
        """Test sampling from single item."""
        sampler = ChoiceSampler(["only"])
        results = [sampler() for _ in range(10)]
        assert all(result == "only" for result in results)

    def test_empty_choices(self):
        """Test error handling for empty choices."""
        with pytest.raises(IndexError):
            sampler = ChoiceSampler([])
            sampler()

    def test_context_ignored(self):
        """Test that context parameter is ignored."""
        sampler = ChoiceSampler(["A", "B"])
        result = sampler({"some": "context"})
        assert result in ["A", "B"]


class TestWeightedSampler:
    """Test WeightedSampler functionality."""

    def test_weighted_distribution(self):
        """Test that weights affect distribution."""
        choices = {"rare": 0.1, "common": 0.9}
        sampler = WeightedSampler(choices)
        
        results = [sampler() for _ in range(1000)]
        common_count = sum(1 for r in results if r == "common")
        
        # Should be roughly 90% common
        assert common_count > 800
        assert all(result in ["rare", "common"] for result in results)

    def test_equal_weights(self):
        """Test equal weights produce roughly equal distribution."""
        choices = {"A": 1.0, "B": 1.0, "C": 1.0}
        sampler = WeightedSampler(choices)
        
        results = [sampler() for _ in range(3000)]
        counts = {choice: sum(1 for r in results if r == choice) for choice in choices}
        
        # Each should appear roughly 1000 times (allow variance)
        for count in counts.values():
            assert 800 < count < 1200

    def test_zero_weight(self):
        """Test that zero weight items never appear."""
        choices = {"never": 0.0, "always": 1.0}
        sampler = WeightedSampler(choices)
        
        results = [sampler() for _ in range(100)]
        assert all(result == "always" for result in results)


class TestUUIDSampler:
    """Test UUIDSampler functionality."""

    def test_generates_valid_uuids(self):
        """Test that generated values are valid UUIDs."""
        sampler = UUIDSampler()
        
        for _ in range(10):
            result = sampler()
            # Should not raise an exception
            uuid.UUID(result)
            assert isinstance(result, str)

    def test_generates_unique_uuids(self):
        """Test that UUIDs are unique."""
        sampler = UUIDSampler()
        results = [sampler() for _ in range(100)]
        
        # All should be unique
        assert len(set(results)) == 100

    def test_context_ignored(self):
        """Test that context is ignored."""
        sampler = UUIDSampler()
        result = sampler({"context": "ignored"})
        uuid.UUID(result)  # Should be valid


class TestDatetimeSampler:
    """Test DatetimeSampler functionality."""

    def test_date_range_sampling(self):
        """Test sampling within date range."""
        sampler = DatetimeSampler("2024-01-01", "2024-01-31")
        
        results = [sampler() for _ in range(100)]
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        
        for result in results:
            assert isinstance(result, datetime)
            assert start <= result <= end

    def test_single_day_range(self):
        """Test sampling from single day."""
        sampler = DatetimeSampler("2024-01-01", "2024-01-01")
        
        results = [sampler() for _ in range(10)]
        expected = datetime(2024, 1, 1)
        
        assert all(result == expected for result in results)

    def test_custom_format(self):
        """Test custom date format."""
        sampler = DatetimeSampler("01/01/2024", "01/31/2024", format="%m/%d/%Y")
        
        result = sampler()
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1

    def test_invalid_date_format(self):
        """Test error handling for invalid date format."""
        with pytest.raises(ValueError):
            DatetimeSampler("invalid", "2024-01-01")

    def test_distribution_across_range(self):
        """Test that dates are distributed across the range."""
        sampler = DatetimeSampler("2024-01-01", "2024-12-31")
        results = [sampler() for _ in range(1000)]
        
        months = [r.month for r in results]
        unique_months = set(months)
        
        # Should hit multiple months across the year
        assert len(unique_months) > 6


class TestRangeSampler:
    """Test RangeSampler functionality."""

    def test_integer_range(self):
        """Test integer range sampling."""
        sampler = RangeSampler(1, 10)
        
        results = [sampler() for _ in range(100)]
        
        for result in results:
            assert isinstance(result, int)
            assert 1 <= result <= 10

    def test_float_range(self):
        """Test float range sampling."""
        sampler = RangeSampler(1.0, 10.0)
        
        results = [sampler() for _ in range(100)]
        
        for result in results:
            assert isinstance(result, float)
            assert 1.0 <= result <= 10.0

    def test_mixed_type_range(self):
        """Test mixed int/float range."""
        sampler = RangeSampler(1, 10.0)  # Mixed types
        
        results = [sampler() for _ in range(100)]
        
        for result in results:
            assert isinstance(result, float)  # Should be float
            assert 1.0 <= result <= 10.0

    def test_negative_range(self):
        """Test negative number range."""
        sampler = RangeSampler(-10, -1)
        
        results = [sampler() for _ in range(100)]
        
        for result in results:
            assert isinstance(result, int)
            assert -10 <= result <= -1

    def test_zero_range(self):
        """Test range of just zero."""
        sampler = RangeSampler(0, 0)
        
        results = [sampler() for _ in range(10)]
        assert all(result == 0 for result in results)

    def test_distribution_coverage(self):
        """Test that range is well covered."""
        sampler = RangeSampler(1, 100)
        results = [sampler() for _ in range(1000)]
        
        # Should hit a good portion of the range
        unique_values = set(results)
        assert len(unique_values) > 50


class TestDatasetSampler:
    """Test DatasetSampler functionality."""

    def test_pandas_dataframe_sampling(self):
        """Test sampling from pandas DataFrame."""
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        sampler = DatasetSampler(df, "col")
        
        results = [sampler() for _ in range(100)]
        assert all(result in [1, 2, 3, 4, 5] for result in results)

    def test_huggingface_dataset_sampling(self):
        """Test sampling from HuggingFace dataset."""
        data = {"col": [1, 2, 3, 4, 5]}
        hf_dataset = HFDataset.from_dict(data)
        sampler = DatasetSampler(hf_dataset, "col")
        
        results = [sampler() for _ in range(100)]
        assert all(result in [1, 2, 3, 4, 5] for result in results)

    def test_dict_sampling(self):
        """Test sampling from dictionary."""
        data = {"col": [1, 2, 3, 4, 5]}
        sampler = DatasetSampler(data, "col")
        
        results = [sampler() for _ in range(100)]
        assert all(result in [1, 2, 3, 4, 5] for result in results)

    def test_empty_dataset_with_default(self):
        """Test empty dataset falls back to default."""
        df = pd.DataFrame({"col": []})
        default_sampler = ChoiceSampler(["default"])
        sampler = DatasetSampler(df, "col", default=default_sampler)
        
        result = sampler()
        assert result == "default"

    def test_empty_dataset_without_default(self):
        """Test empty dataset without default raises error."""
        df = pd.DataFrame({"col": []})
        sampler = DatasetSampler(df, "col")
        
        with pytest.raises(IndexError):
            sampler()

    def test_missing_column(self):
        """Test error handling for missing column."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        
        with pytest.raises(KeyError):
            DatasetSampler(df, "missing_col")

    def test_unsupported_dataset_type(self):
        """Test error for unsupported dataset type."""
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            DatasetSampler("not_a_dataset", "col")


class TestSampleNamespace:
    """Test the sample namespace and its factory functions."""

    def test_choice_factory(self):
        """Test choice factory function."""
        sampler = sample.choice(["A", "B", "C"])
        assert isinstance(sampler, ChoiceSampler)
        
        result = sampler()
        assert result in ["A", "B", "C"]

    def test_weighted_factory(self):
        """Test weighted factory function."""
        sampler = sample.weighted({"A": 0.3, "B": 0.7})
        assert isinstance(sampler, WeightedSampler)
        
        result = sampler()
        assert result in ["A", "B"]

    def test_uuid_factory(self):
        """Test UUID factory function."""
        sampler = sample.uuid()
        assert isinstance(sampler, UUIDSampler)
        
        result = sampler()
        uuid.UUID(result)  # Should be valid UUID

    def test_datetime_factory(self):
        """Test datetime factory function."""
        sampler = sample.datetime("2024-01-01", "2024-01-31")
        assert isinstance(sampler, DatetimeSampler)
        
        result = sampler()
        assert isinstance(result, datetime)

    def test_range_factory(self):
        """Test range factory function."""
        sampler = sample.range(1, 100)
        assert isinstance(sampler, RangeSampler)
        
        result = sampler()
        assert 1 <= result <= 100

    def test_from_dataset_factory(self):
        """Test from_dataset factory function."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        sampler = sample.from_dataset(df, "col")
        assert isinstance(sampler, DatasetSampler)
        
        result = sampler()
        assert result in [1, 2, 3]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_large_range(self):
        """Test sampling from very large range."""
        sampler = RangeSampler(0, 1000000)
        results = [sampler() for _ in range(1000)]
        
        # Should still work and give variety
        unique_values = set(results)
        assert len(unique_values) > 500

    def test_string_choices(self):
        """Test string value choices."""
        choices = ["hello", "world", "test"]
        sampler = ChoiceSampler(choices)
        
        results = [sampler() for _ in range(100)]
        assert all(isinstance(result, str) for result in results)
        assert all(result in choices for result in results)

    def test_mixed_type_choices(self):
        """Test choices with mixed types."""
        choices = [1, "string", 3.14, True]
        sampler = ChoiceSampler(choices)
        
        results = [sampler() for _ in range(100)]
        assert all(result in choices for result in results)

    @patch('random.choice')
    def test_randomness_is_used(self, mock_choice):
        """Test that random module is actually called."""
        mock_choice.return_value = "A"
        
        sampler = ChoiceSampler(["A", "B", "C"])
        result = sampler()
        
        mock_choice.assert_called_once_with(["A", "B", "C"])
        assert result == "A"

    def test_context_parameter_consistency(self):
        """Test that all samplers accept context parameter."""
        samplers = [
            sample.choice(["A"]),
            sample.weighted({"A": 1.0}),
            sample.uuid(),
            sample.datetime("2024-01-01", "2024-01-01"),
            sample.range(1, 1),
            sample.from_dataset({"col": [1]}, "col")
        ]
        
        context = {"test": "context"}
        
        # All should accept context without error
        for sampler in samplers:
            result = sampler(context)
            assert result is not None
