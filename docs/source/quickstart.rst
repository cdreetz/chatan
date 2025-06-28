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

Basic Evaluation
----------------
You can measure quality while you generate data or after rows are produced.

Inline evaluation
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from chatan import dataset, eval, sample

   ds = dataset({
       "col1": sample.choice(["a", "a", "b"]),
       "col2": "b",
       "exact_match": eval.exact_match("col1", "col2")
   }, n=100)

   df = ds.generate()

Aggregate evaluation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   aggregate = ds.evaluate({
       "exact_match": ds.eval.exact_match("col1", "col2"),
   })
   print(aggregate)

Next Steps
----------

 - Check out :doc:`datasets_and_generators` for more complex use cases
- Browse the :doc:`api` reference for all available functions
