"""Dataset creation and manipulation."""

from typing import Dict, Any, Union, Optional, List, Callable
import pandas as pd
from datasets import Dataset as HFDataset
from .generator import GeneratorFunction
from .sampler import SampleFunction


class Dataset:
    """Synthetic dataset generator."""
    
    def __init__(self, schema: Union[Dict[str, Any], str], n: int = 100):
        """
        Initialize dataset with schema.
        
        Args:
            schema: Either a dict mapping column names to generators/samplers,
                   or a string prompt for high-level dataset generation
            n: Number of samples to generate
        """
        if isinstance(schema, str):
            # High-level prompting - we'll implement this later
            raise NotImplementedError("High-level prompting not yet implemented")
        
        self.schema = schema
        self.n = n
        self._data = None
    
    def generate(self, n: Optional[int] = None) -> pd.DataFrame:
        """Generate the dataset."""
        num_samples = n or self.n
        
        # Build dependency graph
        dependencies = self._build_dependency_graph()
        execution_order = self._topological_sort(dependencies)
        
        # Generate data
        data = []
        for i in range(num_samples):
            row = {}
            for column in execution_order:
                value = self._generate_value(column, row)
                row[column] = value
            data.append(row)
        
        self._data = pd.DataFrame(data)
        return self._data
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph from schema."""
        dependencies = {}
        
        for column, func in self.schema.items():
            deps = []
            if isinstance(func, GeneratorFunction):
                # Extract column references from prompt template
                import re
                template = func.prompt_template
                deps = re.findall(r'\{(\w+)\}', template)
            
            dependencies[column] = [dep for dep in deps if dep in self.schema]
        
        return dependencies
    
    def _topological_sort(self, dependencies: Dict[str, List[str]]) -> List[str]:
        """Topologically sort columns by dependencies."""
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(column):
            if column in temp_visited:
                raise ValueError(f"Circular dependency detected involving {column}")
            if column in visited:
                return
            
            temp_visited.add(column)
            for dep in dependencies.get(column, []):
                visit(dep)
            temp_visited.remove(column)
            visited.add(column)
            result.append(column)
        
        for column in self.schema:
            visit(column)
        
        return result
    
    def _generate_value(self, column: str, context: Dict[str, Any]) -> Any:
        """Generate a single value for a column."""
        func = self.schema[column]
        
        if isinstance(func, (GeneratorFunction, SampleFunction)):
            return func(context)
        elif callable(func):
            return func(context)
        else:
            # Static value
            return func
    
    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        if self._data is None:
            self.generate()
        return self._data
    
    def to_huggingface(self) -> HFDataset:
        """Convert to HuggingFace Dataset."""
        if self._data is None:
            self.generate()
        return HFDataset.from_pandas(self._data)
    
    def save(self, path: str, format: str = "parquet") -> None:
        """Save dataset to file."""
        if self._data is None:
            self.generate()
        
        if format == "parquet":
            self._data.to_parquet(path)
        elif format == "csv":
            self._data.to_csv(path, index=False)
        elif format == "json":
            self._data.to_json(path, orient="records")
        else:
            raise ValueError(f"Unsupported format: {format}")


def dataset(schema: Union[Dict[str, Any], str], n: int = 100) -> Dataset:
    """Create a synthetic dataset."""
    return Dataset(schema, n)
