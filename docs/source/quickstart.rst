Quick Start
===================================

Installation
------------

Install chatan from PyPI:

.. code-block:: bash

   pip install chatan

Basic Usage
-----------

Chatan uses async/await for concurrent API calls, which speeds up dataset generation significantly.

1. **Create a generator**

   .. code-block:: python

      import chatan

      gen = chatan.generator("openai", "YOUR_OPENAI_API_KEY")
      # or for Anthropic
      # gen = chatan.generator("anthropic", "YOUR_ANTHROPIC_API_KEY")

2. **Define your dataset schema**

   .. code-block:: python

      ds = chatan.dataset({
          "language": chatan.sample.choice(["Python", "JavaScript", "Rust"]),
          "prompt": gen("write a coding question about {language}"),
          "response": gen("answer this question: {prompt}")
      })

3. **Generate data (async)**

   .. code-block:: python

      import asyncio

      async def main():
          # Generate 100 samples with concurrent API calls
          df = await ds.generate(n=100)

          # Save to file
          ds.save("my_dataset.parquet")
          return df

      df = asyncio.run(main())

Basic Evaluation
----------------
You can measure quality while you generate data or after rows are produced.

Inline evaluation
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import asyncio
   from chatan import dataset, eval, sample

   async def main():
       ds = dataset({
           "col1": sample.choice(["a", "a", "b"]),
           "col2": "b",
           "exact_match": eval.exact_match("col1", "col2")
       })

       df = await ds.generate(n=100)
       return df

   df = asyncio.run(main())

Aggregate evaluation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # After generating data
   aggregate = ds.evaluate({
       "exact_match": ds.eval.exact_match("col1", "col2"),
   })
   print(aggregate)

Next Steps
----------

- Check out :doc:`datasets_and_generators` for more complex use cases
- Browse the :doc:`api` reference for all available functions
