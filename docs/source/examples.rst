Examples
================================================



Basic QA Dataset
----------------

.. code-block:: python

   import chatan

   gen = chatan.generator("openai", "YOUR_API_KEY")
   ds = chatan.dataset({
       "question": gen("write a question about {topic}"),
       "topic": chatan.sample.choice(["Python", "ML", "Data Science"]),
       "answer": gen("answer: {question}")
   })

   df = ds.generate(100)

Mixed Data Types
----------------

.. code-block:: python

   import uuid
   from chatan import dataset, generator, sample

   gen = generator("openai", "YOUR_API_KEY")

   mix = {
       "implementation": "implement a function",
       "conversion": "convert this code", 
       "explanation": "explain this concept"
   }

   ds = dataset({
       "id": sample.uuid(),
       "task_type": sample.choice(mix),
       "prompt": gen("write a prompt for {task_type}"),
       "response": gen("respond to: {prompt}"),
       "difficulty": sample.choice(["easy", "medium", "hard"])
   })

Dataset Augmentation
-------------------

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
