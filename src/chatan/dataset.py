"""Dataset creation and manipulation with async generation."""

import asyncio
import inspect
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from datasets import Dataset as HFDataset
from tqdm.asyncio import tqdm as async_tqdm

from .evaluate import DatasetEvaluator, EvaluationFunction
from .generator import GeneratorFunction
from .sampler import SampleFunction

# Call modes for callable columns
_INJECT = "inject"  # call with matched column values as keyword args
_CTX = "ctx"  # call with full row dict (legacy lambda ctx: ... pattern)
_NO_ARGS = "no_args"  # call with no arguments


def _resolve_callable(func, schema_columns: Set[str]) -> Tuple[List[str], str]:
    """Figure out how to call a callable and what columns it depends on.

    Returns (deps, call_mode) where call_mode is one of:
      - "inject": function params match column names, call with those values
      - "ctx": single-arg function, call with full row dict
      - "no_args": zero-arg function, call with nothing
    """
    # Try parameter name matching first (the clean path)
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return _detect_source_deps(func, schema_columns), _CTX

    params = list(sig.parameters.values())
    column_params = []
    other_required = []

    for p in params:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.name in schema_columns:
            column_params.append(p.name)
        elif p.default is inspect.Parameter.empty:
            other_required.append(p.name)

    # If params match column names and nothing else is required → inject
    if column_params and not other_required:
        return column_params, _INJECT

    # No params at all, or all non-column params have defaults → no-arg
    if not other_required and not column_params:
        return [], _NO_ARGS

    # Single required param that doesn't match a column → legacy ctx pattern
    if len(other_required) == 1 and not column_params:
        deps = _detect_source_deps(func, schema_columns)
        return deps, _CTX

    # Fallback: source inspection + ctx mode
    deps = _detect_source_deps(func, schema_columns)
    return deps, _CTX


def _detect_source_deps(func, schema_columns: Set[str]) -> List[str]:
    """Fallback: detect deps by inspecting source for dict subscript patterns."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return []

    # Match any_var["key"] or any_var['key']
    accessed_keys = re.findall(r"""\w+\[['"](\w+)['"]\]""", source)

    # Resolve closure variables: row[var] where var is a captured string
    if hasattr(func, "__code__") and hasattr(func, "__closure__") and func.__closure__:
        freevars = func.__code__.co_freevars
        closure_cells = func.__closure__
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


def column(func, depends_on: Optional[List[str]] = None) -> "CallableColumn":
    """Explicitly declare dependencies for a callable column.

    In most cases you don't need this — dependencies are auto-detected
    from function parameter names or source code.

    Use this only when auto-detection fails (e.g., dynamic key access).
    """
    return CallableColumn(func, depends_on=depends_on)


class CallableColumn:
    """Wraps a callable with explicit dependency declarations.

    Only needed when auto-detection doesn't work. In most cases,
    just pass a plain function and name its parameters after columns.
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
        self._call_info: Dict[str, Tuple[str, List[str]]] = {}

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

        # Build dependency graph and call info
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

        func = self.schema[column]
        value = await self._call_func(column, func, row)

        # Update row dict (thread-safe within event loop)
        row[column] = value

        # Signal completion
        completion_events[column].set()

        return value

    async def _call_func(
        self, column: str, func: Any, row: Dict[str, Any]
    ) -> Any:
        """Call a schema function with the right calling convention."""
        if isinstance(func, GeneratorFunction):
            return await func(row)

        if isinstance(func, SampleFunction):
            return func(row)

        # Unwrap CallableColumn
        actual_func = func.func if isinstance(func, CallableColumn) else func

        if not callable(actual_func):
            return actual_func  # static value

        call_mode, inject_params = self._call_info.get(column, (_CTX, []))

        if call_mode == _INJECT:
            kwargs = {p: row[p] for p in inject_params}
            if asyncio.iscoroutinefunction(actual_func):
                return await actual_func(**kwargs)
            return actual_func(**kwargs)
        elif call_mode == _NO_ARGS:
            if asyncio.iscoroutinefunction(actual_func):
                return await actual_func()
            return actual_func()
        else:  # _CTX
            if asyncio.iscoroutinefunction(actual_func):
                return await actual_func(row)
            return actual_func(row)

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph from schema."""
        dependencies = {}
        self._call_info = {}
        schema_columns = set(self.schema.keys())

        for col, func in self.schema.items():
            deps: List[str] = []
            call_mode = _CTX

            if isinstance(func, CallableColumn):
                actual_func = func.func
                if func._depends_on is not None:
                    deps = list(func._depends_on)
                    # Still figure out call mode from signature
                    resolved_deps, call_mode = _resolve_callable(
                        actual_func, schema_columns
                    )
                    inject_params = (
                        resolved_deps if call_mode == _INJECT else []
                    )
                    self._call_info[col] = (call_mode, inject_params)
                else:
                    deps_resolved, call_mode = _resolve_callable(
                        actual_func, schema_columns
                    )
                    deps = deps_resolved
                    self._call_info[col] = (
                        call_mode,
                        deps_resolved if call_mode == _INJECT else [],
                    )
            elif hasattr(func, "prompt_template"):
                template = getattr(func, "prompt_template", "")
                deps = re.findall(r"\{(\w+)\}", template)
            elif isinstance(func, SampleFunction):
                deps = []
            elif callable(func):
                deps, call_mode = _resolve_callable(func, schema_columns)
                self._call_info[col] = (
                    call_mode,
                    deps if call_mode == _INJECT else [],
                )

            # Filter to schema columns; filter self-refs for auto-detected
            # Template deps ({variable} in prompts) are explicit, not auto-detected
            is_template = hasattr(func, "prompt_template")
            auto_detected = (
                call_mode in (_INJECT, _CTX, _NO_ARGS)
                and not (isinstance(func, CallableColumn) and func._depends_on is not None)
                and not is_template
            )
            dependencies[col] = [
                dep
                for dep in deps
                if dep in schema_columns
                and (not auto_detected or dep != col)
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
