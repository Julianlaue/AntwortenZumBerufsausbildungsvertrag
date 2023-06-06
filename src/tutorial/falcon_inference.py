from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import transformers
import torch

# Set the path to your model
model_path = '/home/julian.laue/models/falcon/falcon7b/'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Create a pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16,)

# Your input text
input_text = "A cow jumps across a fence and says "

# Generate output
output = generator(input_text, max_length=100, do_sample=True)

# Print the generated text
print(output[0]['generated_text'])

