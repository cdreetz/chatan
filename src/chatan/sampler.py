"""Sampling functions for synthetic data creation."""

import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from datasets import Dataset as HFDataset


class SampleFunction:
    """Base class for sampling functions."""

    def __call__(self, context: Dict[str, Any] = None) -> Any:
        """Generate a sample value."""
        raise NotImplementedError


class ChoiceSampler(SampleFunction):
    """Sample from a list of choices."""

    def __init__(self, choices: Union[List[Any], Dict[str, Any]]):
        if isinstance(choices, dict):
            self.choices = list(choices.keys())
            self.weights = list(choices.values())
        else:
            self.choices = choices
            self.weights = None

    def __call__(self, context: Dict[str, Any] = None) -> Any:
        if self.weights:
            return random.choices(self.choices, weights=self.weights, k=1)[0]
        return random.choice(self.choices)


class WeightedSampler(SampleFunction):
    """Sample from weighted choices."""

    def __init__(self, choices: Dict[str, float]):
        self.choices = list(choices.keys())
        self.weights = list(choices.values())

    def __call__(self, context: Dict[str, Any] = None) -> Any:
        return random.choices(self.choices, weights=self.weights, k=1)[0]


class UUIDSampler(SampleFunction):
    """Generate UUID strings."""

    def __call__(self, context: Dict[str, Any] = None) -> str:
        return str(uuid.uuid4())


class DatetimeSampler(SampleFunction):
    """Sample random datetimes."""

    def __init__(self, start: str, end: str, format: str = "%Y-%m-%d"):
        self.start = datetime.strptime(start, format)
        self.end = datetime.strptime(end, format)
        self.delta = self.end - self.start

    def __call__(self, context: Dict[str, Any] = None) -> datetime:
        random_days = random.randint(0, self.delta.days)
        return self.start + timedelta(days=random_days)


class RangeSampler(SampleFunction):
    """Sample from numeric ranges."""

    def __init__(
        self,
        start: Union[int, float],
        end: Union[int, float],
        step: Optional[Union[int, float]] = None,
    ):
        self.start = start
        self.end = end
        self.step = step
        self.is_int = isinstance(start, int) and isinstance(end, int)

    def __call__(self, context: Dict[str, Any] = None) -> Union[int, float]:
        if self.is_int:
            return random.randint(self.start, self.end)
        return random.uniform(self.start, self.end)


class DatasetSampler(SampleFunction):
    """Sample from existing dataset columns."""

    def __init__(
        self,
        dataset: Union[pd.DataFrame, HFDataset, Dict],
        column: str,
        default: Optional[SampleFunction] = None,
    ):
        if isinstance(dataset, pd.DataFrame):
            self.values = dataset[column].tolist()
        elif isinstance(dataset, HFDataset):
            self.values = dataset[column]
        elif isinstance(dataset, dict):
            self.values = dataset[column]
        else:
            raise ValueError("Unsupported dataset type")

        self.default = default

    def __call__(self, context: Dict[str, Any] = None) -> Any:
        if not self.values and self.default:
            return self.default(context)
        return random.choice(self.values)


class RowSampler:
    """Sample aligned rows from a dataset - all columns from the same row."""

    def __init__(self, dataset: Union[pd.DataFrame, HFDataset, Dict]):
        if isinstance(dataset, pd.DataFrame):
            self._data = dataset
            self._len = len(dataset)
        elif isinstance(dataset, HFDataset):
            self._data = dataset
            self._len = len(dataset)
        elif isinstance(dataset, dict):
            self._data = dataset
            first_key = next(iter(dataset.keys()))
            self._len = len(dataset[first_key])
        else:
            raise ValueError("Unsupported dataset type")

        self._index_cache: Dict[int, int] = {}

    def _get_index(self, context: Dict[str, Any]) -> int:
        """Get or create the sampled row index for this context."""
        ctx_id = id(context) if context else 0
        if ctx_id not in self._index_cache:
            self._index_cache[ctx_id] = random.randint(0, self._len - 1)
        return self._index_cache[ctx_id]

    def __getitem__(self, column: str) -> "RowColumnSampler":
        """Get a sampler for a specific column."""
        return RowColumnSampler(self, column)

    def clear_cache(self) -> None:
        """Clear the index cache."""
        self._index_cache.clear()


class RowColumnSampler(SampleFunction):
    """Sample a column value from an aligned row."""

    def __init__(self, row_sampler: RowSampler, column: str):
        self._row_sampler = row_sampler
        self._column = column

    def __call__(self, context: Dict[str, Any] = None) -> Any:
        idx = self._row_sampler._get_index(context)
        data = self._row_sampler._data

        if isinstance(data, pd.DataFrame):
            return data.iloc[idx][self._column]
        elif isinstance(data, HFDataset):
            return data[idx][self._column]
        elif isinstance(data, dict):
            return data[self._column][idx]
        else:
            raise ValueError("Unsupported dataset type")


# Factory functions for the sample namespace
class SampleNamespace:
    """Namespace for sampling functions."""

    @staticmethod
    def choice(choices: Union[List[Any], Dict[str, Any]]) -> ChoiceSampler:
        """Sample from choices."""
        return ChoiceSampler(choices)

    @staticmethod
    def weighted(choices: Dict[str, float]) -> WeightedSampler:
        """Sample from weighted choices."""
        return WeightedSampler(choices)

    @staticmethod
    def uuid() -> UUIDSampler:
        """Generate UUIDs."""
        return UUIDSampler()

    @staticmethod
    def datetime(start: str, end: str, format: str = "%Y-%m-%d") -> DatetimeSampler:
        """Sample random datetimes."""
        return DatetimeSampler(start, end, format)

    @staticmethod
    def range(
        start: Union[int, float],
        end: Union[int, float],
        step: Optional[Union[int, float]] = None,
    ) -> RangeSampler:
        """Sample from numeric ranges."""
        return RangeSampler(start, end, step)

    @staticmethod
    def from_dataset(
        dataset: Union[pd.DataFrame, HFDataset, Dict],
        column: str,
        default: Optional[SampleFunction] = None,
    ) -> DatasetSampler:
        """Sample from existing dataset (independent per column)."""
        return DatasetSampler(dataset, column, default)

    @staticmethod
    def row(dataset: Union[pd.DataFrame, HFDataset, Dict]) -> RowSampler:
        """Sample aligned rows from a dataset.

        Returns a RowSampler that ensures all column accesses within
        the same row come from the same source row.

        Example:
            >>> eval_data = load_dataset("squad")
            >>> row = sample.row(eval_data)
            >>> ds = dataset({
            ...     "question": row["question"],
            ...     "context": row["context"],
            ...     "answer": row["answer"],
            ... })
        """
        return RowSampler(dataset)


# Export the sample namespace
sample = SampleNamespace()
