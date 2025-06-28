Datasets and Generators
=======================

Chatan builds datasets from two simple concepts. **Generators** call large
language models while **samplers** create structured values. Together they form
the schema for a ``Dataset``.

Supported generator providers
-----------------------------
Chatan includes built-in clients for a few common model sources:

* ``openai`` - access GPT models via the OpenAI API
* ``anthropic`` - use Claude models from Anthropic
* ``transformers``/``huggingface`` - run local HuggingFace models with ``transformers``


Basic QA Dataset
----------------
A minimal dataset uses a single generator to create questions and answers.

.. code-block:: python

   import chatan

   gen = chatan.generator("openai", "YOUR_API_KEY")
   ds = chatan.dataset({
       "question": gen("write a example question from a 5th grade math test"),
       "answer": gen("answer: {question}")
   })

   df = ds.generate(100)

Creating Data Mixes
-------------------
Mix generators with samplers to diversify prompts.

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
--------------------
Pull rows from existing corpora and ask the model to create new variations.

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
After generation, datasets can be saved or converted to other formats.

.. code-block:: python

   # Generate and save
   df = ds.generate(1000)
   ds.save("my_dataset.parquet")
   ds.save("my_dataset.csv", format="csv")

   # Convert to HuggingFace format
   hf_dataset = ds.to_huggingface()

Advanced Examples
-----------------
The snippets below show more complex recipes and local model usage.

Dataset Triton
^^^^^^^^^^^^^^

.. code-block:: python

   from datasets import load_dataset
   from chatan import generator, dataset, sample
   import chatan

   gen = generator("openai", "YOUR_API_KEY")
   kernelbook = load_dataset("GPUMODE/KernelBook")
   kernelbench = load_dataset("ScalingIntelligence/KernelBench")

   ds_1 = dataset({
       "operation": sample.from_dataset(kernelbench, "id"),
       "prompt": gen("write a prompt asking for a Triton kernel for: {operation}"),
       "response": gen("{prompt}")
   })

   ds_2 = dataset({
       "original_prompt": sample.from_dataset(kernelbook, "python_code"),
       "prompt": gen("write a question asking for this code to be written as a Triton kernel"),
       "response": gen("{prompt}")
   })

   df_1 = ds_1(n=500)
   df_2 = ds_2(n=500)
   combined_df = pd.concat([df_1, df_2], ignore_index=True)

Complex Mixes
^^^^^^^^^^^^^

.. code-block:: python

   mixed_ds = dataset({
       "dataset_type": sample.choice(["kernelbench", "kernelbook"]),
       "operation": sample.from_dataset(kernelbench, "id"),
       "original_code": sample.from_dataset(kernelbook, "python_code"),
       "prompt": gen("""
       {%- if dataset_type == "kernelbench" -%}
       write a prompt asking for a Triton kernel for: {operation}
       {%- else -%}
       write a question asking for this code to be written as a Triton kernel: {original_code}
       {%- endif -%}
       """),
       "response": gen("{prompt}")
   })

   schema_choice = sample.choice([
       {"source": "kernelbench", "operation": sample.from_dataset(kernelbench, "id")},
       {"source": "kernelbook", "code": sample.from_dataset(kernelbook, "python_code")}
   ])

   final_ds = dataset({
       "source": schema_choice,
       "prompt": gen("create a Triton kernel prompt based on {source}"),
       "response": gen("{prompt}")
   })

   final_df = final_ds.generate(1000)
   final_ds.save("triton_kernel_dataset.parquet")

Transformers Local Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from chatan import generator, dataset, sample

   # Use a local HuggingFace model
   gen = generator("transformers", model="gpt2")

   ds = dataset({
       "topic": sample.choice(["space", "history", "science"]),
       "prompt": gen("Ask a short question about {topic}"),
       "response": gen("{prompt}")
   })

   df = ds.generate(5)
