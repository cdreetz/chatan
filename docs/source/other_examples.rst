Examples
================================================




Dataset Triton
----------------------------------------------

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


~~WIP~~ Ways to Create Complex Mixes
------------------------------------------------------------

.. code-block:: python

   # Method 2: Create a mixed dataset with sampling
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

    # Method 3: Use sample.choice to pick between schemas
    schema_choice = sample.choice([
        {"source": "kernelbench", "operation": sample.from_dataset(kernelbench, "id")},
        {"source": "kernelbook", "code": sample.from_dataset(kernelbook, "python_code")}
    ])

    final_ds = dataset({
        "source": schema_choice,
        "prompt": gen("create a Triton kernel prompt based on {source}"),
        "response": gen("{prompt}")
    })

    # Generate the final mixed dataset
    final_df = final_ds.generate(1000)
    final_ds.save("triton_kernel_dataset.parquet")


