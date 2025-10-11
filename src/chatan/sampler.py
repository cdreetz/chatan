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
        """Sample from existing dataset."""
        return DatasetSampler(dataset, column, default)


# Export the sample namespace
sample = SampleNamespace()
