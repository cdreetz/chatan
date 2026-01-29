Evaluators
==========

Evaluation helpers let you score generated data. Metrics can run inline as
columns are produced or over the entire dataset after generation.

Available evaluation functions
------------------------------
Current helpers include exact match, semantic similarity, BLEU score, normalized edit distance and an LLM-as-a-judge metric.

Inline evaluation
-----------------
Add evaluation functions directly to the dataset schema. Each row will include a
score column.

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
       print(df.head())
       return df

   df = asyncio.run(main())

Aggregate evaluation
--------------------
Metrics can also be aggregated across the whole dataset using
``Dataset.evaluate``. Note: data must be generated first.

.. code-block:: python

   # After generating data
   aggregate = ds.evaluate({
       "exact_match": ds.eval.exact_match("col1", "col2"),
   })
   print(aggregate)

Comparing variations
--------------------
Evaluate multiple columns at once to compare prompts or models.

.. code-block:: python

   import asyncio
   from chatan import dataset, eval, sample

   async def main():
       ds = dataset({
           "sample_1": sample.choice(["a", "a", "b"]),
           "sample_2": sample.choice(["a", "b"]),
           "ground_truth": "b",
       })

       df = await ds.generate(n=100)

       results = ds.evaluate({
           "sample_1_match": ds.eval.exact_match("sample_1", "ground_truth"),
           "sample_2_match": ds.eval.exact_match("sample_2", "ground_truth"),
       })
       return df, results

   df, results = asyncio.run(main())

Supported metrics
-----------------
The ``evaluate`` module provides helpers such as exact match, semantic
similarity, BLEU score, edit distance and an LLM-as-a-judge metric. Access them
through ``ds.eval`` for aggregate evaluation or ``eval`` for inline use.
