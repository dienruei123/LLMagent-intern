import torch
from multiprocess import set_start_method
from transformers import AutoTokenizer, AutoModelForCausalLM 
from datasets import load_dataset

# Get an example dataset
dataset = load_dataset("fka/awesome-chatgpt-prompts", split="train")

# Get an example model and its tokenizer 
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")
def gpu_computation(batch, rank):
    # Move the model on the right GPU if it's not there already
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    model.to(device)
    
    # Your big GPU call goes here, for example:
    chats = [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ] for prompt in batch["prompt"]]
    texts = [tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True
    ) for chat in chats]
    model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**model_inputs, max_new_tokens=512)
    batch["output"] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return batch
if __name__ == "__main__":
    set_start_method("spawn")
    updated_dataset = dataset.map(
        gpu_computation,
        batched=True,
        batch_size=16,
        with_rank=True,
        num_proc=torch.cuda.device_count(),  # one process per GPU
    )