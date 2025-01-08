import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaModel
from transformers import LogitsProcessorList
from dotenv import load_dotenv

load_dotenv()
access_token = os.environ.get("HUGGINGFACE_KEY")

model_name = "meta-llama/Llama-2-7b-hf"  # Update this with the correct model identifier
# model_name = "huggyllama/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
model:LlamaModel = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)

if torch.cuda.is_available():
    model = model.to('cuda')

input_text = "Ice sinks in water due to its higher density."
# input_text = "Ice floats on water due to its lower density."
inputs = tokenizer(input_text, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
# print(inputs)
print(model)

# Forward pass to get logits from all layers
with torch.no_grad():
    generate_ids = model.generate(
        **inputs, 
        # top_p=0.9,
        top_k=5,
        temperature=0.1,
        return_dict_in_generate=True,
        # do_sample=True,
        output_logits=True, 
        output_hidden_states=True,
        max_length=30
    )
    # generate_ids = model(inputs, )
    # print(generate_ids.sequences)

output_text = tokenizer.batch_decode(generate_ids.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("Output Seq: ", generate_ids.sequences)
print("Output: ", output_text)
print("Hidden states: ")
for layer_idx, states in enumerate(generate_ids.hidden_states):
    # print(states)
    print(f"#{layer_idx}: {len(states)}, {states[0].shape}")

print("Logits: ")
for logit_idx, logits in enumerate(generate_ids.logits):
    print(f"#{logit_idx}: {logits.shape}")
    
print("Last layer info:")
# print(model.lm_head.weight.shape.transpose(0, 1))
output_conv = ''
hidden_layer = -1

for states in generate_ids.hidden_states:
    next_token_logits = torch.matmul(states[hidden_layer][:,-1,:], model.lm_head.weight.transpose(0, 1))
    # probs = torch.softmax(logit_last, dim=-1)
    logits_processor = LogitsProcessorList()
    
    # pre-process distribution
    # next_tokens_scores = logits_processor(inputs['input_ids'], next_token_logits)
    next_tokens_scores = torch.softmax(next_token_logits, dim=-1)
    # print(next_tokens_scores.shape)
    # argmax
    # next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    
    L = torch.argsort(-next_tokens_scores, dim=-1).squeeze()
    # print(L[0], L[1], L[2])
    
    # next_token_idx = torch.argmax(probs, dim=-1)
    print("Next top-3 token idx: ", L[0].item(), L[1].item(), L[2].item())
    print("Next top-3 token score: ", next_tokens_scores[:,L[0]].item(), next_tokens_scores[:,L[1]].item(), next_tokens_scores[:,L[2]].item())
    next_token = tokenizer.batch_decode(torch.tensor([L[0], L[1], L[2]]))
    print("Next token:" , next_token)
    output_conv += next_token[0] + " "
    
print("Generated Text: ", output_text)
print("Decoding for all logits: ", [output_conv])

