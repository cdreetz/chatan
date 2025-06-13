Quick Start
===================================

Installation
------------

Install chatan from PyPI:

.. code-block:: bash

   pip install chatan

Basic Usage
-----------

1. **Create a generator**

   .. code-block:: python

      import chatan
      
      gen = chatan.generator("openai", "YOUR_OPENAI_API_KEY")
      # or for Anthropic
      # gen = chatan.generator("anthropic", "YOUR_ANTHROPIC_API_KEY")

2. **Define your dataset schema**

   .. code-block:: python

      ds = chatan.dataset({
          "prompt": gen("write a coding question about {language}"),
          "language": chatan.sample.choice(["Python", "JavaScript", "Rust"]),
          "response": gen("answer this question: {prompt}")
      })

3. **Generate data**

   .. code-block:: python

      # Generate 100 samples with a progress bar
      df = ds.generate(100)
      
      # Save to file
      ds.save("my_dataset.parquet")

Core Concepts
-------------

**Generators**
   Use LLMs to create text content. Support OpenAI and Anthropic APIs.

**Samplers** 
   Create structured data like UUIDs, choices, ranges, dates.

**Schemas**
   Define relationships between columns using generators and samplers.

**Dependencies**
   Columns can reference other columns using ``{column_name}`` syntax.

Next Steps
----------

- Check out :doc:`examples` for more complex use cases
- Browse the :doc:`api` reference for all available functions
