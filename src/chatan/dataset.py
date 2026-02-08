"""Dataset creation and manipulation with async generation."""

import asyncio
import inspect
import re
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
from datasets import Dataset as HFDataset
from tqdm.asyncio import tqdm as async_tqdm

from .evaluate import DatasetEvaluator, EvaluationFunction
from .generator import GeneratorFunction
from .sampler import SampleFunction


def _detect_callable_deps(func, schema_columns: Set[str]) -> List[str]:
    """Auto-detect dependencies by inspecting a callable's source code.

    Detection works in two ways:
    1. Direct string subscripts: row["col"] or row['col']
    2. Closure variable subscripts: row[col_var] where col_var is a
       captured string matching a schema column (enables factory pattern)
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return []

    # 1. Match direct string subscripts: row["col"] or row['col']
    accessed_keys = re.findall(r"""\w+\[['"](\w+)['"]\]""", source)

    # 2. Match variable subscripts: row[var_name] — resolve via closure
    #    This enables factory functions like get_content("file_path")
    #    that return closures doing row[col_param]
    if hasattr(func, "__code__") and hasattr(func, "__closure__") and func.__closure__:
        freevars = func.__code__.co_freevars
        closure_cells = func.__closure__
        # Find variable subscripts (identifiers, not string literals or numbers)
        var_subscripts = re.findall(r"""\w+\[([a-zA-Z_]\w*)\]""", source)
        for var_name in var_subscripts:
            if var_name in freevars:
                idx = freevars.index(var_name)
                if idx < len(closure_cells):
                    try:
                        cell_value = closure_cells[idx].cell_contents
                    except ValueError:
                        continue
                    if isinstance(cell_value, str):
                        accessed_keys.append(cell_value)

    return list(dict.fromkeys(k for k in accessed_keys if k in schema_columns))


class CallableColumn:
    """Wraps a callable with explicit dependency declarations.

    Only needed when auto-detection doesn't work for your use case
    (e.g., dynamic key access). In most cases, just pass a plain
    callable and dependencies are detected automatically.

    Example:
        >>> schema = {
        ...     "filepath": get_filepaths,
        ...     "content": get_content,  # auto-detects dep on filepath
        ... }
    """

    def __init__(self, func, depends_on: Optional[List[str]] = None):
        self.func = func
        self._depends_on = depends_on

    def __call__(self, context: Dict[str, Any]) -> Any:
        if asyncio.iscoroutinefunction(self.func):
            raise RuntimeError(
                "Async CallableColumn must be awaited. This is handled internally."
            )
        return self.func(context)


def column(func, depends_on: Optional[List[str]] = None) -> CallableColumn:
    """Explicitly declare dependencies for a callable column.

    In most cases you don't need this — dependencies are auto-detected
    from source code. Use this only when auto-detection fails (e.g.,
    dynamic key access, C extensions).

    Args:
        func: A callable (sync or async) that takes a row context dict.
        depends_on: List of column names this callable depends on.

    Returns:
        A CallableColumn wrapping the function with dependency metadata.

    Example:
        >>> # Usually just pass the function directly:
        >>> schema = {
        ...     "filepath": get_filepaths,
        ...     "content": get_content,  # auto-detected
        ... }
        >>> # Use column() only if auto-detection doesn't work:
        >>> schema = {
        ...     "content": column(get_content, depends_on=["filepath"]),
        ... }
    """
    return CallableColumn(func, depends_on=depends_on)


class Dataset:
    """Async dataset generator with dependency-aware execution."""

    def __init__(self, schema: Union[Dict[str, Any], str], n: int = 100):
        """
        Initialize dataset with schema.

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

    @property
    def eval(self):
        """Get dataset evaluator for method chaining."""
        if self._data is None:
            raise ValueError("Dataset must be generated before evaluation")
        return DatasetEvaluator(self)

    def evaluate(self, eval_schema: Dict[str, EvaluationFunction]) -> Dict[str, float]:
        """
        Evaluate multiple metrics on this dataset.

        Args:
            eval_schema: Dictionary mapping metric names to evaluation function.

        Returns:
            Dictionary of metric names to computed scores
        """
        if self._data is None:
            raise ValueError("Dataset must be generated before evaluation")

        results = {}
        for name, eval_function in eval_schema.items():
            results[name] = eval_function(self._data)
        return results

    async def generate(
        self,
        n: Optional[int] = None,
        progress: bool = True,
        concurrency: int = 100,
        max_concurrent_columns: int = -1,
    ) -> pd.DataFrame:
        """Generate the dataset asynchronously with dependency management.

        Args:
            n: Number of samples to generate
            progress: Whether to display a progress bar
            concurrency: Maximum number of rows to process concurrently
            max_concurrent_columns: Maximum columns to generate concurrently per row.
                Use -1 for unlimited (default).

        Returns:
            Generated DataFrame
        """
        num_samples = n or self.n

        # Build dependency graph
        dependencies = self._build_dependency_graph()
        execution_order = self._topological_sort(dependencies)

        # Create semaphore for row concurrency
        row_semaphore = asyncio.Semaphore(concurrency)

        # Generate all rows concurrently with bounded parallelism
        tasks = []
        for i in range(num_samples):
            task = asyncio.create_task(
                self._generate_row_with_deps(
                    i,
                    execution_order,
                    dependencies,
                    row_semaphore,
                    max_concurrent_columns,
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
        max_concurrent_columns: int,
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
                        max_concurrent_columns,
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
        max_concurrent: int,
    ) -> Any:
        """Generate a single column value, waiting for dependencies first."""

        # Wait for all dependencies to complete
        for dep in column_deps:
            await completion_events[dep].wait()

        # Generate the value
        func = self.schema[column]

        if isinstance(func, CallableColumn):
            # Unwrap CallableColumn to get the inner function
            inner = func.func
            if asyncio.iscoroutinefunction(inner):
                value = await inner(row)
            else:
                value = inner(row)
        elif isinstance(func, GeneratorFunction):
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
        schema_columns = set(self.schema.keys())

        for column, func in self.schema.items():
            deps = []
            auto_detected = False

            if isinstance(func, CallableColumn):
                if func._depends_on is not None:
                    # Explicit override via column()
                    deps = list(func._depends_on)
                else:
                    # Auto-detect from the wrapped function's source
                    deps = _detect_callable_deps(func.func, schema_columns)
                    auto_detected = True
            elif hasattr(func, "prompt_template"):
                # Extract dependencies from generator functions
                template = getattr(func, "prompt_template", "")
                deps = re.findall(r"\{(\w+)\}", template)
            elif callable(func) and not isinstance(func, SampleFunction):
                # Auto-detect dependencies from callable source code
                deps = _detect_callable_deps(func, schema_columns)
                auto_detected = True

            # Filter to schema columns; also filter self-references
            # for auto-detected deps (source inspection can pick up
            # the column's own name spuriously)
            dependencies[column] = [
                dep
                for dep in deps
                if dep in schema_columns
                and (not auto_detected or dep != column)
            ]

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
            raise ValueError("Dataset must be generated before conversion. Call generate() first.")
        return self._data

    def to_huggingface(self) -> HFDataset:
        """Convert to HuggingFace Dataset."""
        if self._data is None:
            raise ValueError("Dataset must be generated before conversion. Call generate() first.")
        return HFDataset.from_pandas(self._data)

    def save(self, path: str, format: str = "parquet") -> None:
        """Save dataset to file."""
        if self._data is None:
            raise ValueError("Dataset must be generated before saving. Call generate() first.")

        if format == "parquet":
            self._data.to_parquet(path)
        elif format == "csv":
            self._data.to_csv(path, index=False)
        elif format == "json":
            self._data.to_json(path, orient="records")
        else:
            raise ValueError(f"Unsupported format: {format}")


def dataset(schema: Union[Dict[str, Any], str], n: int = 100) -> Dataset:
    """Create a synthetic dataset.

    Example:
        >>> import asyncio
        >>> from chatan import generator, dataset, sample
        >>>
        >>> async def main():
        ...     gen = generator("openai", "YOUR_KEY")
        ...     ds = dataset({
        ...         "topic": sample.choice(["Python", "JavaScript", "Rust"]),
        ...         "question": gen("Write a question about {topic}"),
        ...         "answer": gen("Answer: {question}")
        ...     })
        ...     df = await ds.generate(100)
        ...     return df
        >>>
        >>> df = asyncio.run(main())
    """
    return Dataset(schema, n)
