"""Chatan: Create synthetic datasets with LLM generators and samplers."""

__version__ = "0.3.0"

from .dataset import dataset
from .evaluate import eval, evaluate
from .generator import generator
from .sampler import sample
from .viewer import generate_with_viewer

__all__ = [
    "dataset",
    "generator",
    "sample",
    "generate_with_viewer",
    "evaluate",
    "eval",
]
