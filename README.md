# Chatan

Create diverse, synthetic datasets. Start from scratch or augment an existing dataset. Simply define your dataset schema as a set of generators, typically being LLMs with a prompt describing what kind of examples you want.

## Installation

Basic installation (includes OpenAI, Anthropic, and core functionality):
```bash
pip install chatan
```

With optional features:
```bash
# For local model support (transformers + PyTorch)
pip install chatan[local]

# For advanced evaluation features (semantic similarity, BLEU score)
pip install chatan[eval]

# For all optional features
pip install chatan[all]
```

## Getting Started

```python
import asyncio
import chatan

async def main():
    # Create a generator
    gen = chatan.generator("openai", "YOUR_API_KEY")

    # Define a dataset schema
    ds = chatan.dataset({
        "topic": chatan.sample.choice(["Python", "JavaScript", "Rust"]),
        "prompt": gen("write a programming question about {topic}"),
        "response": gen("answer this question: {prompt}")
    })

    # Generate the data (async for concurrent API calls)
    df = await ds.generate(n=10)
    return df

df = asyncio.run(main())
```

## Generator Options

### API-based Generators (included in base install)
```python
# OpenAI
gen = chatan.generator("openai", "YOUR_OPENAI_API_KEY")

# Anthropic
gen = chatan.generator("anthropic", "YOUR_ANTHROPIC_API_KEY")
```

### Local Model Support (requires `pip install chatan[local]`)
```python
# HuggingFace Transformers
gen = chatan.generator("transformers", model="microsoft/DialoGPT-medium")
```

## Examples

Create Data Mixes

```python
import asyncio
from chatan import dataset, generator, sample

async def main():
    gen = generator("openai", "YOUR_API_KEY")

    mix = [
        "san antonio, tx",
        "marfa, tx",
        "paris, fr"
    ]

    ds = dataset({
        "id": sample.uuid(),
        "topic": sample.choice(mix),
        "prompt": gen("write an example question about the history of {topic}"),
        "response": gen("respond to: {prompt}"),
    })

    df = await ds.generate(n=100)
    return df

df = asyncio.run(main())
```

Augment datasets

```python
import asyncio
from chatan import generator, dataset, sample
from datasets import load_dataset

async def main():
    gen = generator("openai", "YOUR_API_KEY")
    hf_data = load_dataset("some/dataset")

    ds = dataset({
        "original_prompt": sample.from_dataset(hf_data, "prompt"),
        "variation": gen("rewrite this prompt: {original_prompt}"),
        "response": gen("respond to: {variation}")
    })

    df = await ds.generate(n=100)
    return df

df = asyncio.run(main())
```

## Evaluation

Evaluate rows inline or compute aggregate metrics:

```python
import asyncio
from chatan import dataset, eval, sample

async def main():
    ds = dataset({
        "col1": sample.choice(["a", "a", "b"]),
        "col2": "b",
        "score": eval.exact_match("col1", "col2")
    })

    df = await ds.generate(n=100)

    # Aggregate evaluation
    aggregate = ds.evaluate({
        "exact_match": ds.eval.exact_match("col1", "col2")
    })
    return df, aggregate

df, aggregate = asyncio.run(main())
```

### Advanced Evaluation (requires `pip install chatan[eval]`)
```python
# Semantic similarity using sentence transformers
aggregate = ds.evaluate({
    "semantic_sim": ds.eval.semantic_similarity("col1", "col2")
})

# BLEU score evaluation
aggregate = ds.evaluate({
    "bleu": ds.eval.bleu_score("col1", "col2")
})
```

## Installation Options Summary

| Feature | Install Command | What's Included |
|---------|----------------|-----------------|
| **Basic** | `pip install chatan` | OpenAI, Anthropic, core sampling, basic evaluation |
| **Local Models** | `pip install chatan[local]` | + HuggingFace Transformers, PyTorch |
| **Advanced Eval** | `pip install chatan[eval]` | + Semantic similarity, BLEU scores, NLTK |
| **Everything** | `pip install chatan[all]` | All features above |

## Citation

If you use this code in your research, please cite:

```
@software{reetz2025chatan,
  author = {Reetz, Christian},
  title = {chatan: Create synthetic datasets with LLM generators.},
  url = {https://github.com/cdreetz/chatan},
  year = {2025}
}
```

## Contributing

Community contributions are more than welcome, bug reports, bug fixes, feature requests, feature additions, please refer to the Issues tab.
