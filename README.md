## Examples

Prompt a dataset

```
import chatan

gen = chatan.generator.client("YOUR_OPENAI_API_KEY")
ds = chatan.dataset("create a QA dataset for finetuning an LLM on pharmacology")
```

Creating datasets with different data mixes

```
import uuid
from chatan import dataset, generator, mix

gen = generator.client("YOUR_OPENAI_API_KEY")
#generator.client("anthropic", "YOUR_ANTHROPIC_API_KEY")

mix = {
    "implementation": "Can you implement a matmul kernel in Triton",
    "conversion": "Convert this pytorch model to Triton",
    "explanation": "What memory access optimizations are being used here?"
}

ds = dataset({
    "id": uuid,
    "task": sample.choice(mix),
    "prompt": gen("write a prompt for {task}"),
    "response": gen("write a response to {prompt}"),
)}
```

Augment datasets

```
import uuid
from chatan import dataset, generator
from dataset import load_dataset

gen = generator.client("YOUR_OPENAI_API_KEY")
hf_dataset = load_dataset("GPU_MODE/KernelBook")

ds = dataset({
    "id": sample.from_dataset(hf_data, "id", default=sample.uuid()),
    "prompt": sample.from_dataset(hf_data, "prompt", aug=gen("provide a variation of this prompt")),
    "response": gen("write a response to {prompt}")

})

```
