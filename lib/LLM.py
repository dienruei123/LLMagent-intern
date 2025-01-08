from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

class LLM():
    def __init__(self, device = 'cpu', temperature = 0.01):
        load_dotenv()
        # Load the model and tokenizer
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"  # Update this with the correct model identifier
        # self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.device = device
        self.access_token = os.environ.get('HUGGINGFACE_KEY')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.access_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device,
            token=self.access_token
        )
        self.scaling_factor = 1
        self.temperature = temperature
        
    def _get_hidden_state_and_output(self, target_layer, inputs):
        target_act = None
        def get_target_act_hook(mod, inputs, outputs):
            nonlocal target_act
            target_act = outputs[0]
            return outputs
        handle = self.model.model.layers[target_layer].register_forward_hook(get_target_act_hook)
        output = self.model.generate(inputs, max_new_tokens=30, temperature=self.temperature)
        handle.remove()
        return target_act, output

    def get_llm_state_and_output(self, prompt, target_layer=20):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
        # print(inputs)
        # print(inputs.shape)
        
        target_hidden_state, target_output = self._get_hidden_state_and_output(target_layer, inputs)
        
        # print(target_output)
        output = self.tokenizer.batch_decode(target_output[:, inputs.shape[1]:], skip_special_tokens=True)
        # print(output[0].strip())
        target_hidden_state *= self.scaling_factor
        
        # get the last token's hidden state (affected by all previous tokens)
        # print(target_hidden_state.shape)
        target_hidden_state = target_hidden_state[:,-1,:].squeeze().clone().detach().requires_grad_(True)
        # print(target_hidden_state.shape)
        return target_hidden_state, output[0].strip()
    
    def generate(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
        
        output_ids = self.model.generate(input_ids, max_new_tokens = 200, temperature=self.temperature)
        response = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
        return response[0].strip()
    