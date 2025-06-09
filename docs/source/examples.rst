Examples
================================================



Basic QA Dataset
----------------------------

.. code-block:: python

   import chatan

   gen = chatan.generator("openai", "YOUR_API_KEY")
   ds = chatan.dataset({
       "question": gen("write a example question from a 5th grade math test"),
       "answer": gen("answer: {question}")
   })

   df = ds.generate(100)

Creating Data Mixes
----------------------------

.. code-block:: python

   import uuid
   from chatan import dataset, generator, sample

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

Dataset Augmentation
-------------------------------

.. code-block:: python

   from datasets import load_dataset
   import chatan

   gen = chatan.generator("openai", "YOUR_API_KEY")
   hf_data = load_dataset("some/dataset")

   ds = chatan.dataset({
       "original_prompt": chatan.sample.from_dataset(hf_data, "prompt"),
       "variation": gen("rewrite this prompt: {original_prompt}"),
       "response": gen("respond to: {variation}")
   })

Saving Datasets
---------------

.. code-block:: python

   # Generate and save
   df = ds.generate(1000)
   ds.save("my_dataset.parquet")
   ds.save("my_dataset.csv", format="csv")

   # Convert to HuggingFace format
   hf_dataset = ds.to_huggingface()
