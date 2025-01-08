import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # Update this with the correct model identifier
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# tokenizer = LlamaTokenizer.from_pretrained(model_name)
# model = LlamaForCausalLM.from_pretrained(model_name)

# Enable gradient calculation if you need to capture intermediate outputs
model.eval()  # Set the model to evaluation mode

# Define a hook to capture logits from a specific layer
layer_outputs = []

def get_layer_outputs(module, input, output):
    layer_outputs.append(output)

# Attach the hook to a specific layer (e.g., the 6th layer)
target_layer = model.transformer.h[6]  # Adjust index as needed
hook = target_layer.register_forward_hook(get_layer_outputs)

# Tokenize input text
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt")

# Forward pass through the model
with torch.no_grad():
    outputs = model(**inputs)

# Get logits from the final layer (output)
final_logits = outputs.logits

# Remove the hook after getting the desired outputs
hook.remove()

# Output the logits from the specific layer
print("Logits from the specific layer:")
print(layer_outputs[-1])  # Get the output from the registered hook
