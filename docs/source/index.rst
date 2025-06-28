.. chatan documentation master file, created by
   sphinx-quickstart on Mon Jun  9 13:53:32 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

chatan documentation
===========================================

Create synthetic datasets with LLM generators and samplers.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   quickstart
   datasets_and_generators
   evaluators
   api

Installation
------------

.. code-block:: bash

   pip install chatan

Quick Start
-----------

.. code-block:: python

   import chatan

   # Create a generator
   gen = chatan.generator("openai", "YOUR_API_KEY")

   # Define a dataset schema
   ds = chatan.dataset({
       "topic": chatan.sample.choice(["Python", "JavaScript", "Rust"]),
       "prompt": gen("write a programming question about {topic}"),
       "response": gen("answer this question: {prompt}")
   })

   # Generate the data with a progress bar
   df = ds.generate(n=10)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
