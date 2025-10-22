"""Async dataset generation with dependency management.

This module provides async versions of Dataset generation that can handle
dependencies efficiently using asyncio.Event coordination.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm

from .generator import AsyncGeneratorFunction
from .sampler import SampleFunction


class AsyncDataset:
    """Async dataset generator with dependency-aware execution."""

    def __init__(self, schema: Union[Dict[str, Any], str], n: int = 100):
        """
        Initialize async dataset with schema.

        Args:
            schema: Either a dict mapping column names to generators/samplers,
                   or a string prompt for high-level dataset generation
            n: Number of samples to generate
        """
        if isinstance(schema, str):
            raise NotImplementedError("High-level prompting not yet implemented")

        self.schema = schema
        self.n = n
        self._data = None

    async def generate(
        self, 
        n: Optional[int] = None, 
        progress: bool = True,
        max_concurrent_rows: int = 10,
        max_concurrent_columns: int = 5
    ) -> pd.DataFrame:
        """Generate the dataset asynchronously with dependency management.

        Args:
            n: Number of samples to generate
            progress: Whether to display a progress bar
            max_concurrent_rows: Maximum number of rows to process concurrently
            max_concurrent_columns: Maximum number of columns to generate concurrently per row

        Returns:
            Generated DataFrame
        """
        num_samples = n or self.n

        # Build dependency graph
        dependencies = self._build_dependency_graph()
        execution_order = self._topological_sort(dependencies)

        # Create semaphore for row concurrency
        row_semaphore = asyncio.Semaphore(max_concurrent_rows)
        
        # Generate all rows concurrently with bounded parallelism
        tasks = []
        for i in range(num_samples):
            task = asyncio.create_task(
                self._generate_row_with_deps(
                    i, 
                    execution_order, 
                    dependencies,
                    row_semaphore,
                    max_concurrent_columns
                )
            )
            tasks.append(task)

        # Wait for all rows with progress bar
        if progress:
            rows = await async_tqdm.gather(*tasks, desc="Generating rows")
        else:
            rows = await asyncio.gather(*tasks)

        self._data = pd.DataFrame(rows)
        return self._data

    async def _generate_row_with_deps(
        self,
        row_index: int,
        execution_order: List[str],
        dependencies: Dict[str, List[str]],
        row_semaphore: asyncio.Semaphore,
        max_concurrent_columns: int
    ) -> Dict[str, Any]:
        """Generate a single row with dependency-aware column generation."""
        
        async with row_semaphore:
            row = {}
            completion_events = {col: asyncio.Event() for col in self.schema}
            
            # Launch all column generators for this row
            column_tasks = []
            for column in self.schema:
                task = asyncio.create_task(
                    self._generate_column_value(
                        column,
                        row,
                        dependencies.get(column, []),
                        completion_events,
                        max_concurrent_columns
                    )
                )
                column_tasks.append((column, task))
            
            # Wait for all columns to complete
            for column, task in column_tasks:
                row[column] = await task
            
            return row

    async def _generate_column_value(
        self,
        column: str,
        row: Dict[str, Any],
        column_deps: List[str],
        completion_events: Dict[str, asyncio.Event],
        max_concurrent: int
    ) -> Any:
        """Generate a single column value, waiting for dependencies first."""
        
        # Wait for all dependencies to complete
        for dep in column_deps:
            await completion_events[dep].wait()
        
        # Generate the value
        func = self.schema[column]
        
        if isinstance(func, AsyncGeneratorFunction):
            # Use async generator
            value = await func(row)
        elif isinstance(func, SampleFunction):
            # Samplers are sync but fast
            value = func(row)
        elif callable(func):
            # Check if it's an async callable
            if asyncio.iscoroutinefunction(func):
                value = await func(row)
            else:
                value = func(row)
        else:
            # Static value
            value = func
        
        # Update row dict (thread-safe within event loop)
        row[column] = value
        
        # Signal completion
        completion_events[column].set()
        
        return value

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph from schema."""
        dependencies = {}

        for column, func in self.schema.items():
            deps = []
            
            # Extract dependencies from generator functions
            if hasattr(func, 'prompt_template'):
                import re
                template = getattr(func, 'prompt_template', '')
                deps = re.findall(r"\{(\w+)\}", template)
            
            # Only include dependencies that are in the schema
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

    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        if self._data is None:
            # Run async generation in a new event loop
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.generate())
            finally:
                loop.close()
        return self._data

    def save(self, path: str, format: str = "parquet") -> None:
        """Save dataset to file."""
        df = self.to_pandas()
        
        if format == "parquet":
            df.to_parquet(path)
        elif format == "csv":
            df.to_csv(path, index=False)
        elif format == "json":
            df.to_json(path, orient="records")
        else:
            raise ValueError(f"Unsupported format: {format}")


def async_dataset(schema: Union[Dict[str, Any], str], n: int = 100) -> AsyncDataset:
    """Create an async synthetic dataset.
    
    Example:
        >>> import asyncio
        >>> from chatan import async_generator, async_dataset, sample
        >>> 
        >>> async def main():
        ...     gen = async_generator("openai", "YOUR_KEY")
        ...     ds = async_dataset({
        ...         "topic": sample.choice(["Python", "JavaScript", "Rust"]),
        ...         "question": gen("Write a question about {topic}"),
        ...         "answer": gen("Answer: {question}")
        ...     })
        ...     df = await ds.generate(100)
        ...     return df
        >>> 
        >>> df = asyncio.run(main())
    """
    return AsyncDataset(schema, n)
