Evaluation
==========

Chatan includes helpers for evaluating generated data. Metrics can be computed
while rows are created or aggregated afterwards.

Inline evaluation
-----------------
Add evaluation functions directly in the dataset schema. The resulting column
contains the score for each row.

.. code-block:: python

   from chatan import dataset, eval, sample

   ds = dataset({
       "col1": sample.choice(["a", "a", "b"]),
       "col2": "b",
       "exact_match": eval.exact_match("col1", "col2")
   }, n=100)

   df = ds.generate()
   print(df.head())

Aggregate evaluation
--------------------
Compute metrics across the dataset using ``Dataset.evaluate`` and the
``Dataset.eval`` helper.

.. code-block:: python

   aggregate = ds.evaluate({
       "exact_match": ds.eval.exact_match("col1", "col2"),
   })
   print(aggregate)

Comparing variations
--------------------
Evaluate multiple columns—useful for comparing different prompts or models.

.. code-block:: python

   ds = dataset({
       "sample_1": sample.choice(["a", "a", "b"]),
       "sample_2": sample.choice(["a", "b"]),
       "ground_truth": "b",
   }, n=100)

   df = ds.generate()
   results = ds.evaluate({
       "sample_1_match": ds.eval.exact_match("sample_1", "ground_truth"),
       "sample_2_match": ds.eval.exact_match("sample_2", "ground_truth"),
   })

Supported metrics
-----------------
The ``evaluate`` module provides metrics such as exact match, semantic similarity,
BLEU score, edit distance and an LLM-as-a-judge metric. Access them through
``ds.eval`` for aggregate evaluation or ``eval`` for inline use.
