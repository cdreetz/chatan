"""Comprehensive tests for dataset module."""

import asyncio

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import Mock
from datasets import Dataset as HFDataset

from chatan.dataset import CallableColumn, Dataset, column, dataset
from chatan.generator import GeneratorFunction, BaseGenerator
from chatan.sampler import ChoiceSampler, UUIDSampler


class TestDatasetInitialization:
    """Test Dataset class initialization."""

    def test_init_with_dict_schema(self):
        """Test initialization with dictionary schema."""
        schema = {"col1": "static", "col2": ChoiceSampler(["A", "B"])}
        ds = Dataset(schema, n=50)

        assert ds.schema == schema
        assert ds.n == 50
        assert ds._data is None

    def test_init_with_string_schema(self):
        """Test initialization with string schema (not implemented)."""
        with pytest.raises(NotImplementedError):
            Dataset("create a QA dataset", n=100)

    def test_default_sample_count(self):
        """Test default sample count."""
        ds = Dataset({"col": "value"})
        assert ds.n == 100


class TestDependencyResolution:
    """Test dependency resolution and topological sorting."""

    def test_no_dependencies(self):
        """Test schema with no dependencies."""
        schema = {
            "col1": ChoiceSampler(["A", "B"]),
            "col2": UUIDSampler(),
            "col3": "static",
        }
        ds = Dataset(schema, n=5)

        dependencies = ds._build_dependency_graph()
        assert dependencies == {"col1": [], "col2": [], "col3": []}

        order = ds._topological_sort(dependencies)
        assert set(order) == {"col1", "col2", "col3"}

    def test_simple_dependency_chain(self):
        """Test simple dependency chain A -> B -> C."""
        mock_gen = Mock(spec=GeneratorFunction)
        mock_gen.prompt_template = "Generate based on {col1}"

        mock_gen2 = Mock(spec=GeneratorFunction)
        mock_gen2.prompt_template = "Generate based on {col2}"

        schema = {"col1": ChoiceSampler(["A", "B"]), "col2": mock_gen, "col3": mock_gen2}
        ds = Dataset(schema, n=5)

        dependencies = ds._build_dependency_graph()
        assert dependencies["col1"] == []
        assert dependencies["col2"] == ["col1"]
        assert dependencies["col3"] == ["col2"]

        order = ds._topological_sort(dependencies)
        assert order.index("col1") < order.index("col2")
        assert order.index("col2") < order.index("col3")

    def test_multiple_dependencies(self):
        """Test column with multiple dependencies."""
        mock_gen = Mock(spec=GeneratorFunction)
        mock_gen.prompt_template = "Combine {col1} and {col2}"

        schema = {
            "col1": ChoiceSampler(["A"]),
            "col2": ChoiceSampler(["B"]),
            "col3": mock_gen,
        }
        ds = Dataset(schema, n=5)

        dependencies = ds._build_dependency_graph()
        assert dependencies["col3"] == ["col1", "col2"]

        order = ds._topological_sort(dependencies)
        col3_index = order.index("col3")
        assert order.index("col1") < col3_index
        assert order.index("col2") < col3_index

    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        mock_gen1 = Mock(spec=GeneratorFunction)
        mock_gen1.prompt_template = "Based on {col2}"

        mock_gen2 = Mock(spec=GeneratorFunction)
        mock_gen2.prompt_template = "Based on {col1}"

        schema = {"col1": mock_gen1, "col2": mock_gen2}
        ds = Dataset(schema, n=5)

        dependencies = ds._build_dependency_graph()
        with pytest.raises(ValueError, match="Circular dependency"):
            ds._topological_sort(dependencies)

    def test_self_dependency_detection(self):
        """Test detection of self-referencing dependencies."""
        mock_gen = Mock(spec=GeneratorFunction)
        mock_gen.prompt_template = "Based on {col1} itself"

        schema = {"col1": mock_gen}
        ds = Dataset(schema, n=5)

        dependencies = ds._build_dependency_graph()
        with pytest.raises(ValueError, match="Circular dependency"):
            ds._topological_sort(dependencies)

    def test_dependency_outside_schema(self):
        """Test dependencies on columns not in schema are ignored."""
        mock_gen = Mock(spec=GeneratorFunction)
        mock_gen.prompt_template = "Based on {col1} and {external_col}"

        schema = {"col1": ChoiceSampler(["A"]), "col2": mock_gen}
        ds = Dataset(schema, n=5)

        dependencies = ds._build_dependency_graph()
        # external_col should be filtered out
        assert dependencies["col2"] == ["col1"]


@pytest.mark.asyncio
class TestDataGeneration:
    """Test actual data generation."""

    async def test_static_value_generation(self):
        """Test generation with static values."""
        schema = {"constant": "hello", "number": 42, "boolean": True}
        ds = Dataset(schema, n=3)
        df = await ds.generate()

        assert len(df) == 3
        assert all(df["constant"] == "hello")
        assert all(df["number"] == 42)
        assert all(df["boolean"] == True)

    async def test_sampler_generation(self):
        """Test generation with samplers."""
        schema = {"choice": ChoiceSampler(["A", "B", "C"]), "uuid": UUIDSampler()}
        ds = Dataset(schema, n=10)
        df = await ds.generate()

        assert len(df) == 10
        assert all(df["choice"].isin(["A", "B", "C"]))

        # UUIDs should be unique
        assert len(df["uuid"].unique()) == 10

    async def test_generator_function_with_context(self):
        """Test generation with GeneratorFunction."""

        class MockGenerator(BaseGenerator):
            async def generate(self, prompt: str, **kwargs) -> str:
                return f"Generated: {prompt}"

        gen_func = GeneratorFunction(MockGenerator(), "Create content for {topic}")

        schema = {"topic": ChoiceSampler(["AI", "ML"]), "content": gen_func}
        ds = Dataset(schema, n=2)
        df = await ds.generate()

        assert len(df) == 2
        assert all(df["content"].str.startswith("Generated: Create content for"))

    async def test_lambda_function_generation(self):
        """Test generation with lambda functions."""
        schema = {
            "base": ChoiceSampler([1, 2, 3]),
            "doubled": lambda ctx: ctx["base"] * 2,
            "sum": lambda ctx: ctx["base"] + ctx["doubled"],
        }
        ds = Dataset(schema, n=5)
        df = await ds.generate()

        assert len(df) == 5
        for _, row in df.iterrows():
            assert row["doubled"] == row["base"] * 2
            assert row["sum"] == row["base"] + row["doubled"]

    async def test_complex_dependency_chain(self):
        """Test complex dependency resolution."""
        schema = {
            "a": ChoiceSampler([1, 2]),
            "b": lambda ctx: ctx["a"] * 10,
            "c": ChoiceSampler([100, 200]),
            "d": lambda ctx: ctx["b"] + ctx["c"],
            "e": lambda ctx: ctx["a"] + ctx["d"],
        }
        ds = Dataset(schema, n=5)
        df = await ds.generate()

        assert len(df) == 5
        for _, row in df.iterrows():
            assert row["b"] == row["a"] * 10
            assert row["d"] == row["b"] + row["c"]
            assert row["e"] == row["a"] + row["d"]

    async def test_override_sample_count(self):
        """Test overriding sample count in generate()."""
        schema = {"col": ChoiceSampler(["A"])}
        ds = Dataset(schema, n=10)

        df = await ds.generate(n=5)
        assert len(df) == 5

        # Original n should be unchanged
        assert ds.n == 10

    async def test_multiple_generate_calls(self):
        """Test multiple calls to generate()."""
        schema = {"col": UUIDSampler()}
        ds = Dataset(schema, n=3)

        df1 = await ds.generate()
        df2 = await ds.generate()

        # Should generate different data each time
        assert len(df1) == 3
        assert len(df2) == 3
        assert not df1["col"].equals(df2["col"])


@pytest.mark.asyncio
class TestDatasetExport:
    """Test dataset export functionality."""

    async def test_to_pandas(self):
        """Test conversion to pandas DataFrame."""
        schema = {"col": ChoiceSampler(["A", "B"])}
        ds = Dataset(schema, n=5)
        await ds.generate()

        df = ds.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    async def test_to_pandas_without_prior_generation(self):
        """Test to_pandas raises error if not generated."""
        schema = {"col": ChoiceSampler(["A", "B"])}
        ds = Dataset(schema, n=5)

        assert ds._data is None
        with pytest.raises(ValueError, match="must be generated"):
            ds.to_pandas()

    async def test_to_huggingface(self):
        """Test conversion to HuggingFace Dataset."""
        schema = {"col": ChoiceSampler(["A", "B"])}
        ds = Dataset(schema, n=5)
        await ds.generate()

        hf_ds = ds.to_huggingface()
        assert isinstance(hf_ds, HFDataset)
        assert len(hf_ds) == 5

    async def test_save_parquet(self):
        """Test saving to parquet format."""
        schema = {"col": ChoiceSampler(["A", "B"])}
        ds = Dataset(schema, n=5)
        await ds.generate()

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            try:
                ds.save(f.name, format="parquet")
                assert os.path.exists(f.name)

                # Read back and verify
                df_loaded = pd.read_parquet(f.name)
                assert len(df_loaded) == 5
                assert "col" in df_loaded.columns
            finally:
                os.unlink(f.name)

    async def test_save_csv(self):
        """Test saving to CSV format."""
        schema = {"col": ChoiceSampler(["A", "B"])}
        ds = Dataset(schema, n=5)
        await ds.generate()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            try:
                ds.save(f.name, format="csv")
                assert os.path.exists(f.name)

                # Read back and verify
                df_loaded = pd.read_csv(f.name)
                assert len(df_loaded) == 5
                assert "col" in df_loaded.columns
            finally:
                os.unlink(f.name)

    async def test_save_json(self):
        """Test saving to JSON format."""
        schema = {"col": ChoiceSampler(["A", "B"])}
        ds = Dataset(schema, n=5)
        await ds.generate()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            try:
                ds.save(f.name, format="json")
                assert os.path.exists(f.name)

                # Read back and verify
                df_loaded = pd.read_json(f.name)
                assert len(df_loaded) == 5
                assert "col" in df_loaded.columns
            finally:
                os.unlink(f.name)

    async def test_save_unsupported_format(self):
        """Test error handling for unsupported formats."""
        schema = {"col": ChoiceSampler(["A"])}
        ds = Dataset(schema, n=5)
        await ds.generate()

        with pytest.raises(ValueError, match="Unsupported format: xml"):
            ds.save("test.xml", format="xml")

    async def test_save_without_prior_generation(self):
        """Test save raises error if not generated."""
        schema = {"col": ChoiceSampler(["A"])}
        ds = Dataset(schema, n=5)

        assert ds._data is None

        with pytest.raises(ValueError, match="must be generated"):
            ds.save("test.parquet")


class TestDatasetFactory:
    """Test dataset factory function."""

    def test_factory_creates_dataset(self):
        """Test factory function creates Dataset instance."""
        schema = {"col": "value"}
        ds = dataset(schema, n=50)

        assert isinstance(ds, Dataset)
        assert ds.schema == schema
        assert ds.n == 50

    def test_factory_default_count(self):
        """Test factory function with default count."""
        schema = {"col": "value"}
        ds = dataset(schema)

        assert ds.n == 100


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling scenarios."""

    async def test_callable_context_error(self):
        """Test error handling when callable fails."""

        def failing_func(ctx):
            raise ValueError("Function failed")

        schema = {"base": ChoiceSampler([1]), "fail": failing_func}
        ds = Dataset(schema, n=1)

        with pytest.raises(ValueError, match="Function failed"):
            await ds.generate()

    async def test_generator_function_context_error(self):
        """Test error handling when GeneratorFunction fails."""

        class FailingGenerator(BaseGenerator):
            async def generate(self, prompt: str, **kwargs) -> str:
                raise Exception("Generation failed")

        gen_func = GeneratorFunction(FailingGenerator(), "Template")

        schema = {"content": gen_func}
        ds = Dataset(schema, n=1)

        with pytest.raises(Exception, match="Generation failed"):
            await ds.generate()

    async def test_missing_context_variable(self):
        """Test error when context variable is missing."""

        class EchoGenerator(BaseGenerator):
            async def generate(self, prompt: str, **kwargs) -> str:
                return prompt

        gen_func = GeneratorFunction(EchoGenerator(), "Template {missing}")

        schema = {"base": ChoiceSampler([1]), "content": gen_func}
        ds = Dataset(schema, n=1)

        with pytest.raises(KeyError):
            await ds.generate()


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests combining multiple components."""

    async def test_realistic_qa_dataset(self):
        """Test realistic QA dataset generation."""
        call_count = 0
        responses = [
            "What is machine learning?",
            "Machine learning is a subset of AI...",
            "Explain neural networks",
            "Neural networks are computational models...",
        ]

        class MockGenerator(BaseGenerator):
            async def generate(self, prompt: str, **kwargs) -> str:
                nonlocal call_count
                result = responses[call_count % len(responses)]
                call_count += 1
                return result

        gen_func_q = GeneratorFunction(MockGenerator(), "Generate a question about {topic}")
        gen_func_a = GeneratorFunction(MockGenerator(), "Answer: {question}")

        schema = {
            "id": UUIDSampler(),
            "topic": ChoiceSampler(["ML", "AI", "DL"]),
            "question": gen_func_q,
            "answer": gen_func_a,
        }

        ds = Dataset(schema, n=2)
        df = await ds.generate()

        assert len(df) == 2
        assert all(df["topic"].isin(["ML", "AI", "DL"]))
        assert len(df["id"].unique()) == 2  # Unique IDs

    async def test_augmentation_pattern(self):
        """Test data augmentation pattern."""
        # Simulate existing dataset
        existing_data = pd.DataFrame({"original": ["Text 1", "Text 2", "Text 3"]})

        class VariationGenerator(BaseGenerator):
            async def generate(self, prompt: str, **kwargs) -> str:
                # Extract the original text from prompt
                return f"Variation of {prompt.split(': ')[1]}"

        from chatan.sampler import DatasetSampler

        gen_func = GeneratorFunction(VariationGenerator(), "Create variation: {original}")

        schema = {
            "original": DatasetSampler(existing_data, "original"),
            "variation": gen_func,
        }

        ds = Dataset(schema, n=3)
        df = await ds.generate()

        assert len(df) == 3
        assert all(df["original"].isin(["Text 1", "Text 2", "Text 3"]))
        assert all(df["variation"].str.startswith("Variation of"))

    async def test_data_mix_pattern(self):
        """Test data mix generation pattern."""
        responses = [
            "Implementation prompt",
            "Implementation response",
            "Explanation prompt",
            "Explanation response",
        ]
        call_count = 0

        class MockGenerator(BaseGenerator):
            async def generate(self, prompt: str, **kwargs) -> str:
                nonlocal call_count
                result = responses[call_count % len(responses)]
                call_count += 1
                return result

        gen_prompt = GeneratorFunction(MockGenerator(), "Generate {task_type} prompt")
        gen_response = GeneratorFunction(MockGenerator(), "Respond to: {prompt}")

        schema = {
            "task_type": ChoiceSampler(["implementation", "explanation"]),
            "prompt": gen_prompt,
            "response": gen_response,
        }

        ds = Dataset(schema, n=2)
        df = await ds.generate()

        assert len(df) == 2
        assert all(df["task_type"].isin(["implementation", "explanation"]))


class TestCallableAutoDetection:
    """Test auto-detection of callable dependencies."""

    def test_auto_detect_single_dep(self):
        """Test auto-detecting a single dependency from source."""
        schema = {
            "a": ChoiceSampler([1, 2]),
            "b": lambda ctx: ctx["a"] * 2,
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert deps["a"] == []
        assert deps["b"] == ["a"]

    def test_auto_detect_chained_deps(self):
        """Test auto-detecting chained callable deps: a -> b -> c."""

        def compute_b(ctx):
            return ctx["a"] * 10

        def compute_c(ctx):
            return ctx["b"] + 1

        schema = {
            "a": ChoiceSampler([1, 2]),
            "b": compute_b,
            "c": compute_c,
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert deps["a"] == []
        assert deps["b"] == ["a"]
        assert deps["c"] == ["b"]

        order = ds._topological_sort(deps)
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_auto_detect_multiple_deps(self):
        """Test auto-detecting multiple dependencies."""

        def combine(ctx):
            return ctx["x"] + ctx["y"]

        schema = {
            "x": ChoiceSampler([1, 2]),
            "y": ChoiceSampler([10, 20]),
            "z": combine,
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert set(deps["z"]) == {"x", "y"}

    def test_auto_detect_callable_on_callable(self):
        """Test auto-detecting deps between callables."""

        def derived(ctx):
            return ctx["base"] * 2

        def final(ctx):
            return ctx["derived"] + ctx["base"]

        schema = {
            "base": ChoiceSampler([1, 2, 3]),
            "derived": derived,
            "final": final,
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert deps["derived"] == ["base"]
        assert set(deps["final"]) == {"derived", "base"}

        order = ds._topological_sort(deps)
        assert order.index("base") < order.index("derived")
        assert order.index("derived") < order.index("final")

    def test_auto_detect_with_generator_dep(self):
        """Test auto-detecting callable dep on a GeneratorFunction column."""
        mock_gen = Mock(spec=GeneratorFunction)
        mock_gen.prompt_template = "Generate for {topic}"

        schema = {
            "topic": ChoiceSampler(["AI", "ML"]),
            "response": mock_gen,
            "word_count": lambda ctx: len(ctx["response"].split()),
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert deps["topic"] == []
        assert deps["response"] == ["topic"]
        assert deps["word_count"] == ["response"]

    def test_circular_callable_deps_detected(self):
        """Test circular dependencies between callables are detected."""

        def func_a(ctx):
            return ctx["b"]

        def func_b(ctx):
            return ctx["a"]

        schema = {"a": func_a, "b": func_b}
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        with pytest.raises(ValueError, match="Circular dependency"):
            ds._topological_sort(deps)

    def test_non_schema_keys_filtered(self):
        """Test that detected keys not in schema are ignored."""

        def func(ctx):
            return ctx["real_col"] + ctx["not_a_column"]

        schema = {
            "real_col": ChoiceSampler([1]),
            "result": func,
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert deps["result"] == ["real_col"]

    def test_self_reference_filtered(self):
        """Test that self-references are filtered from auto-detected deps."""

        def func(ctx):
            return ctx.get("myself", 0) + 1

        schema = {"myself": func}
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert deps["myself"] == []

    def test_samplers_skip_auto_detect(self):
        """Test that SampleFunction instances are not source-inspected."""
        schema = {
            "a": ChoiceSampler([1, 2]),
            "b": ChoiceSampler([3, 4]),
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert deps["a"] == []
        assert deps["b"] == []


class TestCallableColumnExplicitOverride:
    """Test column() as explicit override when auto-detection is insufficient."""

    def test_explicit_override_respected(self):
        """Test that column() with depends_on overrides auto-detection."""
        schema = {
            "a": ChoiceSampler([1]),
            "b": ChoiceSampler([2]),
            "c": column(lambda ctx: ctx["a"], depends_on=["a", "b"]),
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        # Explicit override: both a and b, even though source only shows "a"
        assert set(deps["c"]) == {"a", "b"}

    def test_column_factory_creates_callable_column(self):
        """Test that column() returns a CallableColumn."""
        func = lambda ctx: ctx["a"] * 2
        col = column(func, depends_on=["a"])

        assert isinstance(col, CallableColumn)
        assert col._depends_on == ["a"]
        assert col.func is func


@pytest.mark.asyncio
class TestCallableColumnGeneration:
    """Test actual generation with auto-detected callable dependencies."""

    async def test_simple_callable_chain(self):
        """Test generation with auto-detected callable chain."""
        schema = {
            "base": ChoiceSampler([10, 20, 30]),
            "doubled": lambda ctx: ctx["base"] * 2,
            "tripled": lambda ctx: ctx["base"] * 3,
        }
        ds = Dataset(schema, n=5)
        df = await ds.generate(progress=False)

        assert len(df) == 5
        for _, row in df.iterrows():
            assert row["doubled"] == row["base"] * 2
            assert row["tripled"] == row["base"] * 3

    async def test_chained_callable_columns(self):
        """Test callable columns that depend on other callable columns."""

        def step1(ctx):
            return ctx["seed"] * 10

        def step2(ctx):
            return ctx["step1"] + 5

        def step3(ctx):
            return ctx["step2"] * ctx["seed"]

        schema = {
            "seed": ChoiceSampler([1, 2, 3]),
            "step1": step1,
            "step2": step2,
            "step3": step3,
        }
        ds = Dataset(schema, n=10)
        df = await ds.generate(progress=False)

        assert len(df) == 10
        for _, row in df.iterrows():
            assert row["step1"] == row["seed"] * 10
            assert row["step2"] == row["step1"] + 5
            assert row["step3"] == row["step2"] * row["seed"]

    async def test_filepath_content_pattern(self):
        """Test the filepath -> content pattern (user's use case)."""
        fake_files = {
            "/repo/main.py": "print('hello')",
            "/repo/utils.py": "def helper(): pass",
            "/repo/test.py": "import pytest",
        }

        def get_filepath(ctx):
            import random

            return random.choice(list(fake_files.keys()))

        def get_content(ctx):
            return fake_files[ctx["filepath"]]

        schema = {
            "filepath": get_filepath,
            "content": get_content,
        }
        ds = Dataset(schema, n=5)
        df = await ds.generate(progress=False)

        assert len(df) == 5
        for _, row in df.iterrows():
            assert row["filepath"] in fake_files
            assert row["content"] == fake_files[row["filepath"]]

    async def test_three_level_callable_chain(self):
        """Test a 3-level callable chain: filepath -> content -> analysis."""
        fake_files = {
            "/a.py": "import os\nimport sys",
            "/b.py": "import json",
        }

        def get_filepath(ctx):
            import random

            return random.choice(list(fake_files.keys()))

        def get_content(ctx):
            return fake_files[ctx["filepath"]]

        def count_imports(ctx):
            return ctx["content"].count("import")

        schema = {
            "filepath": get_filepath,
            "content": get_content,
            "import_count": count_imports,
        }
        ds = Dataset(schema, n=6)
        df = await ds.generate(progress=False)

        assert len(df) == 6
        for _, row in df.iterrows():
            expected_count = fake_files[row["filepath"]].count("import")
            assert row["import_count"] == expected_count

    async def test_async_callable_auto_detected(self):
        """Test async callable with auto-detected deps."""

        async def async_process(ctx):
            await asyncio.sleep(0.001)
            return ctx["value"] * 100

        schema = {
            "value": ChoiceSampler([1, 2, 3]),
            "processed": async_process,
        }
        ds = Dataset(schema, n=5)
        df = await ds.generate(progress=False)

        assert len(df) == 5
        for _, row in df.iterrows():
            assert row["processed"] == row["value"] * 100

    async def test_mixed_generator_and_callable_deps(self):
        """Test callable depending on a GeneratorFunction column."""

        class MockGenerator(BaseGenerator):
            async def generate(self, prompt: str, **kwargs) -> str:
                return f"Response about {prompt.split()[-1]}"

        gen_func = GeneratorFunction(MockGenerator(), "Talk about {topic}")

        schema = {
            "topic": ChoiceSampler(["AI", "ML"]),
            "response": gen_func,
            "word_count": lambda ctx: len(ctx["response"].split()),
        }
        ds = Dataset(schema, n=3)
        df = await ds.generate(progress=False)

        assert len(df) == 3
        for _, row in df.iterrows():
            assert row["word_count"] == len(row["response"].split())


class TestParameterInjection:
    """Test parameter name injection — the clean API where function params match columns."""

    def test_param_name_detects_single_dep(self):
        """Function with param matching a column name → detected as dep."""

        def doubled(base):
            return base * 2

        schema = {
            "base": ChoiceSampler([1, 2, 3]),
            "doubled": doubled,
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert deps["base"] == []
        assert deps["doubled"] == ["base"]

    def test_param_name_detects_multiple_deps(self):
        """Function with multiple params matching columns."""

        def total(price, quantity):
            return price * quantity

        schema = {
            "price": ChoiceSampler([10, 20]),
            "quantity": ChoiceSampler([1, 5]),
            "total": total,
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert set(deps["total"]) == {"price", "quantity"}

    def test_param_name_chain(self):
        """Chained deps via param names: a -> b -> c."""

        def step1(seed):
            return seed * 10

        def step2(step1):
            return step1 + 5

        schema = {
            "seed": ChoiceSampler([1, 2]),
            "step1": step1,
            "step2": step2,
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert deps["seed"] == []
        assert deps["step1"] == ["seed"]
        assert deps["step2"] == ["step1"]

        order = ds._topological_sort(deps)
        assert order.index("seed") < order.index("step1")
        assert order.index("step1") < order.index("step2")

    def test_no_arg_function(self):
        """Zero-arg function → no deps, called with no args."""
        import uuid

        def make_id():
            return str(uuid.uuid4())

        schema = {
            "id": make_id,
            "value": ChoiceSampler([1]),
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert deps["id"] == []

    def test_non_column_param_falls_back_to_ctx(self):
        """Single param not matching any column → ctx mode (legacy compat)."""

        def compute(row):
            return row["base"] * 2

        schema = {
            "base": ChoiceSampler([10]),
            "result": compute,
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        # Falls back to source inspection, finds "base" via ctx["base"]
        assert deps["result"] == ["base"]

    def test_mixed_column_and_default_params(self):
        """Params matching columns plus params with defaults → inject mode."""

        def compute(base, multiplier=2):
            return base * multiplier

        schema = {
            "base": ChoiceSampler([5, 10]),
            "result": compute,
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert deps["result"] == ["base"]


@pytest.mark.asyncio
class TestParameterInjectionGeneration:
    """Test actual generation with parameter injection."""

    async def test_simple_param_injection(self):
        """Function params matching column names get injected values."""

        def doubled(base):
            return base * 2

        schema = {
            "base": ChoiceSampler([10, 20, 30]),
            "doubled": doubled,
        }
        ds = Dataset(schema, n=5)
        df = await ds.generate(progress=False)

        assert len(df) == 5
        for _, row in df.iterrows():
            assert row["doubled"] == row["base"] * 2

    async def test_multi_param_injection(self):
        """Multiple params injected from different columns."""

        def total(price, quantity):
            return price * quantity

        schema = {
            "price": ChoiceSampler([10, 20]),
            "quantity": ChoiceSampler([1, 2, 5]),
            "total": total,
        }
        ds = Dataset(schema, n=10)
        df = await ds.generate(progress=False)

        assert len(df) == 10
        for _, row in df.iterrows():
            assert row["total"] == row["price"] * row["quantity"]

    async def test_chained_param_injection(self):
        """Chained callable columns via param injection."""

        def step1(seed):
            return seed * 10

        def step2(step1):
            return step1 + 5

        def step3(step2, seed):
            return step2 * seed

        schema = {
            "seed": ChoiceSampler([1, 2, 3]),
            "step1": step1,
            "step2": step2,
            "step3": step3,
        }
        ds = Dataset(schema, n=10)
        df = await ds.generate(progress=False)

        assert len(df) == 10
        for _, row in df.iterrows():
            assert row["step1"] == row["seed"] * 10
            assert row["step2"] == row["step1"] + 5
            assert row["step3"] == row["step2"] * row["seed"]

    async def test_async_param_injection(self):
        """Async functions work with param injection."""

        async def async_doubled(base):
            await asyncio.sleep(0.001)
            return base * 2

        schema = {
            "base": ChoiceSampler([5, 10]),
            "doubled": async_doubled,
        }
        ds = Dataset(schema, n=5)
        df = await ds.generate(progress=False)

        assert len(df) == 5
        for _, row in df.iterrows():
            assert row["doubled"] == row["base"] * 2

    async def test_no_arg_function_generation(self):
        """Zero-arg functions called correctly."""
        import uuid

        def make_id():
            return str(uuid.uuid4())

        schema = {
            "id": make_id,
            "base": ChoiceSampler([1]),
        }
        ds = Dataset(schema, n=5)
        df = await ds.generate(progress=False)

        assert len(df) == 5
        assert len(df["id"].unique()) == 5  # All unique UUIDs

    async def test_filepath_content_param_injection(self):
        """User's primary use case with param injection."""
        fake_files = {
            "/repo/main.py": "print('hello')",
            "/repo/utils.py": "def helper(): pass",
        }

        def get_filepath():
            import random
            return random.choice(list(fake_files.keys()))

        def get_content(file_path):
            return fake_files[file_path]

        schema = {
            "file_path": get_filepath,
            "file_content": get_content,
        }
        ds = Dataset(schema, n=5)
        df = await ds.generate(progress=False)

        assert len(df) == 5
        for _, row in df.iterrows():
            assert row["file_path"] in fake_files
            assert row["file_content"] == fake_files[row["file_path"]]

    async def test_param_injection_with_generator(self):
        """Param injection works alongside GeneratorFunction columns."""

        class MockGenerator(BaseGenerator):
            async def generate(self, prompt: str, **kwargs) -> str:
                return f"Generated: {prompt}"

        gen_func = GeneratorFunction(MockGenerator(), "write about {file_content}")

        def get_content(topic):
            return f"Content about {topic}"

        schema = {
            "topic": ChoiceSampler(["AI", "ML"]),
            "file_content": get_content,
            "query": gen_func,
        }
        ds = Dataset(schema, n=3)
        df = await ds.generate(progress=False)

        assert len(df) == 3
        for _, row in df.iterrows():
            assert row["file_content"] == f"Content about {row['topic']}"
            assert "Generated:" in row["query"]

    async def test_backward_compat_ctx_pattern(self):
        """Old ctx dict pattern still works alongside param injection."""

        def new_style(base):
            return base * 2

        def old_style(ctx):
            return ctx["base"] * 3

        schema = {
            "base": ChoiceSampler([10, 20]),
            "new": new_style,
            "old": old_style,
        }
        ds = Dataset(schema, n=5)
        df = await ds.generate(progress=False)

        assert len(df) == 5
        for _, row in df.iterrows():
            assert row["new"] == row["base"] * 2
            assert row["old"] == row["base"] * 3


class TestFactoryPatternAutoDetection:
    """Test auto-detection for factory functions that return closures."""

    def test_closure_variable_resolved(self):
        """Test that closure variables referencing column names are detected."""

        def get_content(filepath_col):
            def _generate(row):
                return row[filepath_col]
            return _generate

        schema = {
            "file_path": ChoiceSampler(["/a.py", "/b.py"]),
            "content": get_content("file_path"),
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert deps["file_path"] == []
        assert deps["content"] == ["file_path"]

    def test_factory_chain(self):
        """Test chained factory functions auto-detect deps."""
        import random

        def pick_from(choices):
            def _generate(row):
                return random.choice(choices)
            return _generate

        def read_from(col_name):
            files = {"/a.py": "import os", "/b.py": "import json"}
            def _generate(row):
                return files[row[col_name]]
            return _generate

        def analyze(col_name):
            def _generate(row):
                return len(row[col_name])
            return _generate

        schema = {
            "file_path": pick_from(["/a.py", "/b.py"]),
            "content": read_from("file_path"),
            "length": analyze("content"),
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert deps["file_path"] == []
        assert deps["content"] == ["file_path"]
        assert deps["length"] == ["content"]

        order = ds._topological_sort(deps)
        assert order.index("file_path") < order.index("content")
        assert order.index("content") < order.index("length")

    def test_factory_with_multiple_column_refs(self):
        """Test factory capturing multiple column references."""

        def combine(col_a, col_b):
            def _generate(row):
                return str(row[col_a]) + str(row[col_b])
            return _generate

        schema = {
            "first": ChoiceSampler(["A", "B"]),
            "second": ChoiceSampler(["1", "2"]),
            "combined": combine("first", "second"),
        }
        ds = Dataset(schema, n=5)
        deps = ds._build_dependency_graph()

        assert set(deps["combined"]) == {"first", "second"}


@pytest.mark.asyncio
class TestFactoryPatternGeneration:
    """Test actual generation with factory-style callables."""

    async def test_filepath_content_factory(self):
        """Test the user's primary use case: filepath -> content via factories."""
        import random

        fake_files = {
            "/repo/main.py": "print('hello')",
            "/repo/utils.py": "def helper(): pass",
            "/repo/test.py": "import pytest",
        }

        def get_filepath(repo_path):
            files = list(repo_path.keys())
            def _generate(row):
                return random.choice(files)
            return _generate

        def get_content(filepath_col):
            def _generate(row):
                return fake_files[row[filepath_col]]
            return _generate

        schema = {
            "file_path": get_filepath(fake_files),
            "file_content": get_content("file_path"),
        }
        ds = Dataset(schema, n=5)
        df = await ds.generate(progress=False)

        assert len(df) == 5
        for _, row in df.iterrows():
            assert row["file_path"] in fake_files
            assert row["file_content"] == fake_files[row["file_path"]]

    async def test_three_level_factory_chain(self):
        """Test filepath -> content -> analysis via factory chain."""
        import random

        fake_files = {
            "/a.py": "import os\nimport sys",
            "/b.py": "import json",
        }

        def get_filepath(file_dict):
            paths = list(file_dict.keys())
            def _generate(row):
                return random.choice(paths)
            return _generate

        def get_content(filepath_col):
            def _generate(row):
                return fake_files[row[filepath_col]]
            return _generate

        def count_pattern(content_col, pattern):
            def _generate(row):
                return row[content_col].count(pattern)
            return _generate

        schema = {
            "file_path": get_filepath(fake_files),
            "file_content": get_content("file_path"),
            "import_count": count_pattern("file_content", "import"),
        }
        ds = Dataset(schema, n=6)
        df = await ds.generate(progress=False)

        assert len(df) == 6
        for _, row in df.iterrows():
            expected = fake_files[row["file_path"]].count("import")
            assert row["import_count"] == expected

    async def test_factory_with_generator_mix(self):
        """Test factories mixed with GeneratorFunction columns."""

        class MockGenerator(BaseGenerator):
            async def generate(self, prompt: str, **kwargs) -> str:
                return f"Generated: {prompt}"

        gen = GeneratorFunction(MockGenerator(), "write about {file_content}")

        fake_files = {"main.py": "hello world"}

        def get_filepath(files):
            def _generate(row):
                return list(files.keys())[0]
            return _generate

        def get_content(filepath_col):
            def _generate(row):
                return fake_files[row[filepath_col]]
            return _generate

        schema = {
            "file_path": get_filepath(fake_files),
            "file_content": get_content("file_path"),
            "query": gen,
        }
        ds = Dataset(schema, n=2)
        df = await ds.generate(progress=False)

        assert len(df) == 2
        for _, row in df.iterrows():
            assert row["file_content"] == "hello world"
            assert "Generated:" in row["query"]
