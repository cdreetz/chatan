# Chatan Architecture & Workflow

## Overview

Chatan is a Python library for creating diverse, synthetic datasets using LLM generators and samplers.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER CODE                                       │
│                                                                             │
│   ds = chatan.dataset({                                                     │
│       "topic": chatan.sample.choice(["Python", "Rust"]),                    │
│       "question": gen("Write a question about {topic}"),                    │
│       "answer": gen("Answer: {question}")                                   │
│   })                                                                        │
│   df = ds.generate(n=100)                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SCHEMA DEFINITION                                  │
│                                                                             │
│   ┌──────────────┐   ┌──────────────────┐   ┌──────────────────────┐       │
│   │   Samplers   │   │    Generators    │   │    Static Values     │       │
│   │              │   │                  │   │                      │       │
│   │ • choice()   │   │ • OpenAI         │   │ • strings            │       │
│   │ • weighted() │   │ • Anthropic      │   │ • numbers            │       │
│   │ • uuid()     │   │ • Transformers   │   │ • any Python value   │       │
│   │ • datetime() │   │   (local)        │   │                      │       │
│   │ • range()    │   │                  │   │                      │       │
│   │ • from_ds()  │   │                  │   │                      │       │
│   └──────────────┘   └──────────────────┘   └──────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DEPENDENCY ANALYSIS                                   │
│                                                                             │
│   1. Extract {variable} references from prompt templates                    │
│   2. Build dependency graph between columns                                 │
│   3. Topological sort to determine execution order                          │
│                                                                             │
│   Example:  topic (no deps) → question (needs topic) → answer (needs q)    │
│                                                                             │
│            ┌───────┐      ┌───────────┐      ┌────────┐                    │
│            │ topic │ ───▶ │ question  │ ───▶ │ answer │                    │
│            └───────┘      └───────────┘      └────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
┌────────────────────────────────────┐  ┌────────────────────────────────────┐
│         SYNC GENERATION            │  │        ASYNC GENERATION            │
│         (dataset.py)               │  │        (async_dataset.py)          │
│                                    │  │                                    │
│  for row in range(n):              │  │  Concurrent execution with:        │
│    for col in sorted_columns:      │  │  • max_concurrent_rows (10)        │
│      generate_value(col)           │  │  • max_concurrent_columns (5)      │
│                                    │  │  • asyncio.Semaphore              │
│  Sequential, simple                │  │  • asyncio.Event for deps          │
└────────────────────────────────────┘  └────────────────────────────────────┘
                          │                       │
                          └───────────┬───────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VALUE GENERATION                                     │
│                                                                             │
│   For each cell (row, column):                                              │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        Value Type Check                              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│              │              │                │                │              │
│              ▼              ▼                ▼                ▼              │
│      ┌───────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│      │  Static   │  │  Sampler    │  │  Generator  │  │  Callable   │      │
│      │  Value    │  │  Function   │  │  Function   │  │  (custom)   │      │
│      └───────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
│           │              │                │                │               │
│           │              │                ▼                │               │
│           │              │   ┌─────────────────────────┐   │               │
│           │              │   │  Template Substitution  │   │               │
│           │              │   │  {topic} → "Python"     │   │               │
│           │              │   └─────────────────────────┘   │               │
│           │              │                │                │               │
│           │              │                ▼                │               │
│           │              │   ┌─────────────────────────┐   │               │
│           │              │   │      LLM API Call       │   │               │
│           │              │   │  ┌─────────────────┐    │   │               │
│           │              │   │  │ OpenAI API      │    │   │               │
│           │              │   │  │ Anthropic API   │    │   │               │
│           │              │   │  │ Local Model     │    │   │               │
│           │              │   │  └─────────────────┘    │   │               │
│           │              │   └─────────────────────────┘   │               │
│           │              │                │                │               │
│           ▼              ▼                ▼                ▼               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        Return Value                                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT LAYER                                      │
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │   pandas    │  │  HuggingFace│  │    File     │  │    Live     │       │
│   │  DataFrame  │  │   Dataset   │  │   Export    │  │   Viewer    │       │
│   │             │  │             │  │             │  │             │       │
│   │ .to_pandas()│  │.to_hf()     │  │ .save()     │  │ HTML+JS     │       │
│   └─────────────┘  └─────────────┘  │ • parquet   │  │ Real-time   │       │
│                                     │ • csv       │  │ updates     │       │
│                                     │ • json      │  └─────────────┘       │
│                                     └─────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION (Optional)                                 │
│                                                                             │
│   ds.evaluate({                                                             │
│       "match": ds.eval.exact_match("col1", "col2"),                         │
│       "similarity": ds.eval.semantic_similarity("col1", "col2")             │
│   })                                                                        │
│                                                                             │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │
│   │   Exact    │ │  Semantic  │ │    BLEU    │ │   Edit     │ │   LLM    │ │
│   │   Match    │ │ Similarity │ │   Score    │ │  Distance  │ │  Judge   │ │
│   └────────────┘ └────────────┘ └────────────┘ └────────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### Generators

```
┌─────────────────────────────────────────────────────────────────┐
│                    Generator Architecture                        │
│                                                                 │
│   chatan.generator("openai", api_key)                           │
│   chatan.async_generator("anthropic", api_key)                  │
│                                                                 │
│           ┌─────────────────────────────────────┐               │
│           │      GeneratorClient (Factory)      │               │
│           └─────────────────────────────────────┘               │
│                            │                                    │
│              ┌─────────────┼─────────────┐                      │
│              ▼             ▼             ▼                      │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│   │   OpenAI     │ │  Anthropic   │ │ Transformers │           │
│   │  Generator   │ │  Generator   │ │  Generator   │           │
│   │              │ │              │ │   (local)    │           │
│   │ GPT-4, 3.5   │ │ Claude 3     │ │ Any HF model │           │
│   └──────────────┘ └──────────────┘ └──────────────┘           │
│                                                                 │
│   Returns: GeneratorFunction (callable with prompt template)    │
│                                                                 │
│   gen = chatan.generator("openai", key)                         │
│   fn = gen("Write about {topic}")  # GeneratorFunction          │
│   result = fn({"topic": "Python"}) # Makes API call             │
└─────────────────────────────────────────────────────────────────┘
```

### Samplers

```
┌─────────────────────────────────────────────────────────────────┐
│                      Sampler Architecture                        │
│                                                                 │
│   Access via: chatan.sample.*                                   │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                   SampleNamespace                         │  │
│   │                                                          │  │
│   │  .choice(items)      → Random selection from list        │  │
│   │  .weighted(dict)     → Weighted random selection         │  │
│   │  .uuid()             → Generate UUID string              │  │
│   │  .datetime(s, e)     → Random date in range              │  │
│   │  .range(min, max)    → Random number in range            │  │
│   │  .from_dataset(df,c) → Sample from existing data         │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Each returns a SampleFunction that's called during generation │
└─────────────────────────────────────────────────────────────────┘
```

### Async Concurrency Model

```
┌─────────────────────────────────────────────────────────────────┐
│                  Async Concurrency Model                         │
│                                                                 │
│   max_concurrent_rows = 10                                      │
│   max_concurrent_columns = 5                                    │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    Row Semaphore                         │   │
│   │                   (limits: 10 rows)                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│              │         │         │         │                    │
│              ▼         ▼         ▼         ▼                    │
│         ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐                │
│         │ Row 1 │ │ Row 2 │ │ Row 3 │ │ Row N │  ...           │
│         └───────┘ └───────┘ └───────┘ └───────┘                │
│              │                                                  │
│              ▼                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Per-Row Column Events                       │   │
│   │                                                         │   │
│   │    Col A ──▶ Col B ──▶ Col C    (dependency chain)      │   │
│   │      │         │         │                              │   │
│   │    event     event     event    (signal when done)      │   │
│   │                                                         │   │
│   │    Columns without dependencies run in parallel         │   │
│   │    Columns WITH dependencies wait for events            │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Define  │───▶│  Analyze │───▶│ Generate │───▶│  Output  │───▶│ Evaluate │
│  Schema  │    │   Deps   │    │   Data   │    │   Data   │    │  Quality │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
     │               │               │               │               │
     │               │               │               │               │
     ▼               ▼               ▼               ▼               ▼
 generators      topological      sync or       DataFrame      metrics &
 samplers          sort           async          exports         scores
 values          ordering       execution        viewer
```

## Module Structure

```
src/chatan/
├── __init__.py          # Public API (dataset, generator, sample, eval)
├── dataset.py           # Async Dataset class with dependency-aware execution
├── generator.py         # Async LLM generators (OpenAI, Anthropic, Transformers)
├── sampler.py           # Data samplers (choice, uuid, datetime, etc.)
├── evaluate.py          # Evaluation metrics
└── viewer.py            # Live HTML viewer
```

## Quick Start Example

```python
import asyncio
import chatan

async def main():
    # 1. Create a generator
    gen = chatan.generator("openai", "your-api-key")

    # 2. Define dataset schema
    ds = chatan.dataset({
        "language": chatan.sample.choice(["Python", "JavaScript", "Rust"]),
        "difficulty": chatan.sample.weighted({"easy": 0.5, "medium": 0.3, "hard": 0.2}),
        "question": gen("Write a {difficulty} coding question about {language}"),
        "solution": gen("Provide a solution for: {question}")
    })

    # 3. Generate data (async for concurrent API calls)
    df = await ds.generate(n=100)

    # 4. Export
    df.to_csv("coding_questions.csv")
    ds.save("dataset.parquet", format="parquet")

asyncio.run(main())
```
