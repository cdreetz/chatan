"""Verifiers integration for using rollout environments as chatan generators."""

from typing import Any, Dict, Literal

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from verifiers.envs.environment import Environment

from chatan.generator import BaseGenerator, GeneratorFunction

ExtractType = Literal["completion", "reward", "metrics", "trajectory", "full"]


class RolloutGenerator(BaseGenerator):
    """Wraps a verifiers environment as a chatan generator."""

    def __init__(
        self,
        env: Environment,
        client: AsyncOpenAI,
        model: str,
        extract: ExtractType = "completion",
        max_retries: int = 3,
        **sampling_args,
    ):

        self.env = env
        self.client = client
        self.model = model
        self.extract = extract
        self.max_retries = max_retries
        self.sampling_args = sampling_args
        self._row_counter = 0

    async def generate(self, prompt: str, **kwargs) -> Any:
        """Run a verifiers rollout and extract the requested field."""

        input_data = {
            "prompt": [{"role": "user", "content": prompt}],
            "example_id": self._row_counter,
            "task": self.env.env_id or "default",
        }

        if "answer" in kwargs and kwargs["answer"] is not None:
            input_data["answer"] = kwargs["answer"]

        self._row_counter += 1

        result = await self._run_with_retry(input_data)

        return self._extract_field(result)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    async def _run_with_retry(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run rollout with retry on transient failures."""
        return await self.env.run_rollout(
            input_data,
            self.client,
            self.model,
            sampling_args=self.sampling_args,
        )

    def _extract_field(self, result: Dict[str, Any]) -> Any:
        if self.extract == "completion":
            return self._extract_completion_text(result)
        elif self.extract == "reward":
            return result.get("reward", 0.0)
        elif self.extract == "metrics":
            return result.get("metrics", {})
        elif self.extract == "trajectory":
            return result.get("trajectory", [])
        elif self.extract == "full":
            return dict(result)
        else:
            return result.get(self.extract)

    def _extract_completion_text(self, result: Dict[str, Any]) -> str:
        completion = result.get("completion", [])
        if isinstance(completion, list):
            for msg in reversed(completion):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    return content if isinstance(content, str) else str(content)
            return ""
        return str(completion)


class RolloutGeneratorClient:
    """Factory for creating rollout generator functions."""

    def __init__(
        self,
        env: Environment,
        client: AsyncOpenAI,
        model: str,
        max_retries: int = 3,
        **default_sampling_args,
    ):
        self.env = env
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.default_sampling_args = default_sampling_args

    def __call__(
        self,
        prompt_template: str,
        extract: ExtractType = "completion",
        answer: str | None = None,
        **variables,
    ) -> GeneratorFunction:
        """Create a generator function that runs a rollout.

        Args:
            prompt_template: Template string for the prompt. Can reference other columns.
            extract: What to extract from the rollout result.
                - "completion": Last assistant message text (default)
                - "reward": Reward score
                - "metrics": Metrics dict
                - "trajectory": Full trajectory list
                - "full": Entire rollout output dict
            answer: Template string for the ground truth answer. Can reference other columns.
                e.g., answer="{ground_truth}" will resolve the ground_truth column value.
            **variables: Additional variables to pass to the template.

        Returns:
            GeneratorFunction that can be used in a chatan dataset schema.
        """
        generator = RolloutGenerator(
            env=self.env,
            client=self.client,
            model=self.model,
            extract=extract,
            max_retries=self.max_retries,
            **self.default_sampling_args,
        )

        if answer is not None:
            variables["answer"] = lambda ctx: answer.format(**ctx)

        return GeneratorFunction(generator, prompt_template, variables)


def rollout_generator(
    env: Environment,
    client: AsyncOpenAI,
    model: str,
    max_retries: int = 3,
    **sampling_args,
) -> RolloutGeneratorClient:
    """Create a rollout generator client.

    Args:
        env: Verifiers environment to use for rollouts.
        client: AsyncOpenAI client.
        model: Model name to use for generation.
        max_retries: Number of retries on transient failures. Default 3.
        **sampling_args: Additional sampling arguments passed to the model.

    Returns:
        RolloutGeneratorClient that can be called to create generator functions.

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
        >>> ds = dataset({
        ...     "question": sample.from_dataset(questions, "question"),
        ...     "ground_truth": sample.from_dataset(questions, "answer"),
        ...     "model_answer": rollout("{question}", answer="{ground_truth}"),
        ...     "trajectory": rollout("{question}", extract="trajectory"),
        ... })
    """
    return RolloutGeneratorClient(
        env=env,
        client=client,
        model=model,
        max_retries=max_retries,
        **sampling_args,
    )
