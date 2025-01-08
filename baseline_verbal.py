import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from lib.prompt_template import *
from lib.LLM import LLM


def determine_new_label(output, label, threshold: float = 5):
    score_min, score_max = 0, 10
    
    predict = output.split('\n')[0][8:]
    confidence_score = float(output.split('\n')[-1][7:])
    if confidence_score < score_min or confidence_score > score_max:
        return -1
    
    pred_label = 1 if 'true' in predict.lower() else 0
    if pred_label == label and confidence_score >= threshold:
        return 1
    
    if pred_label != label and confidence_score < threshold:
        return 1
    
    return 0

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    llm = LLM(device=device)
    # llm = LLM(device=device, temperature=0.9)
    
    # dataset_name = 'generated'
    dataset_name = 'boolq'
    
    if dataset_name == 'generated':
        ds = load_dataset("csv", data_files={'train': f'publicDataset/{dataset_name}_true_false.csv'})
        ds = ds['train']
    elif dataset_name == 'boolq':
        ds = load_dataset("google/boolq")
        ds = ds['validation']
    
    preprocessed = list()
    for data in tqdm(ds, desc="Verbal"):
        # TODO: optimize prompt
        system_message = SYSTEM_PROMPT
        # print(data)
        if dataset_name == 'generated':
            question = data['statement']
        elif dataset_name == 'boolq':
            question = data['question']
            
        user_message = USER_PROMPT_SCORE.format(question)
        refined_input = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message}[/INST]"

        output = ''
        for _ in range(5):
            output = llm.generate(refined_input)
            try:
                if output.split('\n')[0][8:].lower() in ['true', 'false'] and type(float(output.split('\n')[-1][7:])) == float:
                    break
            except:
                pass
        else:
            print(output)
            output = 'Result: False\nScore: -1'
        print(output)
        
        # TODO: determine the label (GPT possible)
        
        if dataset_name == 'generated':
            label = data['label']
        elif dataset_name == 'boolq':
            
            label = data['answer']
        refined_label = determine_new_label(output, label)
        
        # print(state.shape)
        preprocessed.append({
            'label': refined_label
        })
        
    correct = np.sum([(data['label'] == 1) for data in preprocessed])
    invalid_response = np.sum([(data['label'] == -1) for data in preprocessed])
    print(f"Accuracy: {correct}/{len(preprocessed)} ({correct/len(preprocessed)*100:.2f}%)")
    print(f"Invalid response: {invalid_response}/{len(preprocessed)} ({invalid_response/len(preprocessed)*100:.2f}%)")