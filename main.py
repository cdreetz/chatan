import uuid
from minos import dataset, generator, mix

gen = generator.client("YOUR_OPENAI_API_KEY")
#generator.client("anthropic", "YOUR_ANTHROPIC_API_KEY")

mix = {
    "implementation": "Can you implement a matmul kernel in Triton", 
    "conversion": "Convert this pytorch model to Triton", 
    "explanation": "What memory access optimizations are being used here?"
}

ds = dataset({
    "id": uuid,
    "task": sample(mix),
    "prompt": gen("write a prompt for {task}"),
    "response": gen(f"write a response to {prompt}"),
)}



