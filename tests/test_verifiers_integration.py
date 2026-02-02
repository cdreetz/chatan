"""Tests for verifiers integration."""

import pytest

pytest.importorskip("verifiers")

import asyncio
from unittest.mock import AsyncMock, MagicMock

from chatan import dataset, sample
from chatan.integrations.verifiers import RolloutGenerator, rollout_generator


@pytest.fixture
def mock_env():
    env = MagicMock()
    env.env_id = "test_env"
    env.run_rollout = AsyncMock(
        return_value={
            "completion": [{"role": "assistant", "content": "42"}],
            "reward": 1.0,
            "metrics": {"accuracy": 1.0},
            "trajectory": [{"prompt": "test", "completion": "42"}],
        }
    )
    return env


@pytest.fixture
def mock_client():
    return MagicMock()


class TestRolloutGenerator:
    @pytest.mark.asyncio
    async def test_extract_completion(self, mock_env, mock_client):
        gen = RolloutGenerator(mock_env, mock_client, "test-model", extract="completion")
        result = await gen.generate("What is 6 * 7?")
        assert result == "42"

    @pytest.mark.asyncio
    async def test_extract_reward(self, mock_env, mock_client):
        gen = RolloutGenerator(mock_env, mock_client, "test-model", extract="reward")
        result = await gen.generate("What is 6 * 7?")
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_extract_metrics(self, mock_env, mock_client):
        gen = RolloutGenerator(mock_env, mock_client, "test-model", extract="metrics")
        result = await gen.generate("What is 6 * 7?")
        assert result == {"accuracy": 1.0}

    @pytest.mark.asyncio
    async def test_extract_trajectory(self, mock_env, mock_client):
        gen = RolloutGenerator(mock_env, mock_client, "test-model", extract="trajectory")
        result = await gen.generate("What is 6 * 7?")
        assert result == [{"prompt": "test", "completion": "42"}]

    @pytest.mark.asyncio
    async def test_extract_full(self, mock_env, mock_client):
        gen = RolloutGenerator(mock_env, mock_client, "test-model", extract="full")
        result = await gen.generate("What is 6 * 7?")
        assert "completion" in result
        assert "reward" in result

    @pytest.mark.asyncio
    async def test_answer_passed_through(self, mock_env, mock_client):
        gen = RolloutGenerator(mock_env, mock_client, "test-model")
        await gen.generate("What is 6 * 7?", answer="42")

        call_args = mock_env.run_rollout.call_args
        input_data = call_args[0][0]
        assert input_data["answer"] == "42"


class TestRolloutGeneratorClient:
    def test_creates_generator_function(self, mock_env, mock_client):
        client = rollout_generator(mock_env, mock_client, "test-model")
        gen_func = client("{question}")

        assert gen_func.prompt_template == "{question}"

    def test_answer_template_resolved(self, mock_env, mock_client):
        client = rollout_generator(mock_env, mock_client, "test-model")
        gen_func = client("{question}", answer="{ground_truth}")

        assert "answer" in gen_func.variables
        ctx = {"ground_truth": "42", "question": "test"}
        assert gen_func.variables["answer"](ctx) == "42"


class TestDatasetIntegration:
    @pytest.mark.asyncio
    async def test_in_dataset_schema(self, mock_env, mock_client):
        rollout = rollout_generator(mock_env, mock_client, "test-model")

        ds = dataset(
            {
                "question": sample.choice(["What is 2+2?", "What is 3+3?"]),
                "answer": rollout("{question}"),
            },
            n=2,
        )

        df = await ds.generate(progress=False)

        assert len(df) == 2
        assert "question" in df.columns
        assert "answer" in df.columns
        assert all(df["answer"] == "42")

    @pytest.mark.asyncio
    async def test_with_dependencies(self, mock_env, mock_client):
        rollout = rollout_generator(mock_env, mock_client, "test-model")

        ds = dataset(
            {
                "question": sample.choice(["What is 2+2?"]),
                "ground_truth": sample.choice(["4"]),
                "model_answer": rollout("{question}", answer="{ground_truth}"),
                "reward": rollout("{question}", extract="reward"),
            },
            n=1,
        )

        df = await ds.generate(progress=False)

        assert df["model_answer"].iloc[0] == "42"
        assert df["reward"].iloc[0] == 1.0
