"""Verifiers integration for using rollout environments as chatan generators."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Literal

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from verifiers.envs.environment import Environment

from chatan.generator import BaseGenerator, GeneratorFunction


@asynccontextmanager
async def _null_semaphore():
    """A no-op async context manager that acts as an unlimited semaphore."""
    yield


ExtractType = Literal["completion", "reward", "metrics", "trajectory", "full"]


class RolloutResult:
    """Lazy rollout result - runs once, extracts many."""

    def __init__(
        self,
        env: Environment,
        client: AsyncOpenAI,
        model: str,
        prompt_template: str,
        answer_template: str | None = None,
        **sampling_args,
    ):
        self.env = env
        self.client = client
        self.model = model
        self.prompt_template = prompt_template
        self.answer_template = answer_template
        self.sampling_args = sampling_args
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._locks: Dict[int, asyncio.Lock] = {}
        self._row_counter = 0

    async def _get_result(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get or compute the rollout result for this context."""
        ctx_id = id(context)

        if ctx_id in self._cache:
            return self._cache[ctx_id]

        # Get or create lock for this context
        if ctx_id not in self._locks:
            self._locks[ctx_id] = asyncio.Lock()

        async with self._locks[ctx_id]:
            # Double-check after acquiring lock
            if ctx_id in self._cache:
                return self._cache[ctx_id]

            prompt = self.prompt_template.format(**context)
            answer = self.answer_template.format(**context) if self.answer_template else None

            input_data = {
                "prompt": [{"role": "user", "content": prompt}],
                "example_id": self._row_counter,
                "task": self.env.env_id or "default",
            }
            if answer is not None:
                input_data["answer"] = answer

            self._row_counter += 1

            result = await self._run_with_retry(input_data)
            self._cache[ctx_id] = result
            return result

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    async def _run_with_retry(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run rollout with retry on transient failures."""
        return await self.env.run_rollout(
            input_data,
            self.client,
            self.model,
            gen_sampling_args=self.sampling_args,
            gen_sem=_null_semaphore(),
        )

    def _extractor(self, extract: ExtractType) -> "RolloutExtractor":
        """Create an extractor for a specific field."""
        return RolloutExtractor(self, extract)

    @property
    def completion(self) -> "RolloutExtractor":
        return self._extractor("completion")

    @property
    def reward(self) -> "RolloutExtractor":
        return self._extractor("reward")

    @property
    def metrics(self) -> "RolloutExtractor":
        return self._extractor("metrics")

    @property
    def trajectory(self) -> "RolloutExtractor":
        return self._extractor("trajectory")

    @property
    def full(self) -> "RolloutExtractor":
        return self._extractor("full")


class RolloutExtractor(BaseGenerator):
    """Extracts a field from a shared RolloutResult."""

    def __init__(self, rollout_result: RolloutResult, extract: ExtractType):
        self._rollout_result = rollout_result
        self._extract = extract

    # Make it usable directly in dataset schema
    prompt_template = ""

    async def __call__(self, context: Dict[str, Any]) -> Any:
        """Called by dataset generation."""
        return await self.generate("", _context=context)

    async def generate(self, prompt: str, **kwargs) -> Any:
        context = kwargs.get("_context", {})
        result = await self._rollout_result._get_result(context)
        return self._extract_field(result)

    def _extract_field(self, result: Dict[str, Any]) -> Any:
        if self._extract == "completion":
            return self._extract_completion_text(result)
        elif self._extract == "reward":
            return result.get("reward", 0.0)
        elif self._extract == "metrics":
            return result.get("metrics", {})
        elif self._extract == "trajectory":
            return result.get("trajectory", [])
        elif self._extract == "full":
            return dict(result)
        else:
            return result.get(self._extract)

    def _extract_completion_text(self, result: Dict[str, Any]) -> str:
        completion = result.get("completion", [])
        if isinstance(completion, list):
            for msg in reversed(completion):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    return content if isinstance(content, str) else str(content)
            return ""
        return str(completion)


def rollout_generator(
    env: Environment,
    client: AsyncOpenAI,
    model: str,
    **sampling_args,
) -> "RolloutClient":
    """Create a rollout client.

    Args:
        env: Verifiers environment to use for rollouts.
        client: AsyncOpenAI client.
        model: Model name to use for generation.
        **sampling_args: Additional sampling arguments passed to the model.

    Returns:
        RolloutClient that can be called to create rollout results.

    Example:
        >>> from chatan import dataset, sample
        >>> from chatan.integrations.verifiers import rollout_generator
        >>> from verifiers.utils.env_utils import load_environment
        >>> from openai import AsyncOpenAI
        >>>
        >>> client = AsyncOpenAI(api_key="...")
        >>> env = load_environment("gsm8k")
        >>> rollout = rollout_generator(env, client, "gpt-4.1-mini")
        >>>
        >>> row = sample.row(env.get_eval_dataset())
        >>> r = rollout("{question}", answer="{ground_truth}")
        >>>
        >>> ds = dataset({
        ...     "question": row["prompt"],
        ...     "ground_truth": row["answer"],
        ...     "model_answer": r.completion,
        ...     "trajectory": r.trajectory,
        ...     "reward": r.reward,
        ... })
    """
    return RolloutClient(env, client, model, **sampling_args)


class RolloutClient:
    """Factory for creating rollout results."""

    def __init__(
        self,
        env: Environment,
        client: AsyncOpenAI,
        model: str,
        **sampling_args,
    ):
        self.env = env
        self.client = client
        self.model = model
        self.sampling_args = sampling_args

    def __call__(
        self,
        prompt_template: str,
        answer: str | None = None,
    ) -> RolloutResult:
        """Create a rollout result that can be used to extract multiple fields.

        Args:
            prompt_template: Template string for the prompt.
            answer: Template string for the ground truth answer.

        Returns:
            RolloutResult with .completion, .trajectory, .reward, .metrics, .full extractors.
        """
        return RolloutResult(
            env=self.env,
            client=self.client,
            model=self.model,
            prompt_template=prompt_template,
            answer_template=answer,
            **self.sampling_args,
        )
