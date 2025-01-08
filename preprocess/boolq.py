from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import torch
from lib.LLM import LLM
from lib.args import parse_args
from lib.prompt_template import *

import argparse
import os

def split_dataset(ds: DatasetDict):
    train_testvalid = ds['train'].train_test_split(test_size=0.2)
    # Split the 10% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    ds = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']}
    )

    # print(ds)
    return ds

def preprocess_dataset(raw_data, ref: bool = False, llm: LLM = None, target_layer: int = 20):
    """New dataset
    input: Ask LLM truthfulness of the statement
    data format: (hidden state, label)
      hidden state: output of the LLM layer
      label: 
        If ground truth is TRUE, and LLM responds TRUE, then LLM "is not lying", label 1; else LLM "is lying", label 0.
        If ground truth is FALSE, and LLM responds FALSE, then LLM "is not lying", label 1; else LLM "is lying", label 0.
    
    Args:
        raw_data: dataset to be preprocessed (List of dict)
        llm (LLM, optional): LLM to be tested. Defaults to None.

    Returns:
        List of dict 
        [{'statement': [...], 'label': Literal[0, 1]}, {...}, ... ]
        # DatasetDict: preprocessed dataset with hidden states and refined labels
    """
    
    def get_state_and_output(prompt, target_layer):
        return llm.get_llm_state_and_output(prompt, target_layer)
    
    def determine_new_label(predict:str, label:bool):
        pred_label = True if 'true' in predict.lower() else False
        return 1 if pred_label == label else 0
    
    preprocessed = list()
    for data in tqdm(raw_data, desc=f"Preprocess"):
        # TODO: optimize prompt
        system_message = SYSTEM_PROMPT
        user_message = USER_PROMPT_Q.format(data['question']) if not ref else USER_PROMPT_Q_PASSAGE.format(data['question'], data['passage'])
        refined_input = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message}[/INST]"

        state, output = get_state_and_output(refined_input, target_layer)
        # print(output)
        
        # TODO: determine the label (GPT possible)
        refined_label = determine_new_label(output, data['answer'])
        
        # print(state.shape)
        preprocessed.append({
            'hidden_state': state,
            'label': refined_label
        })
    # print(preprocessed)
    return preprocessed

if __name__ == '__main__':
    args = parse_args()
    target_layer = args.layer
    dataset_name = 'boolq'
    ref = args.ref
    
    if not os.path.exists(f'Dataset_train'):
        os.makedirs(f'Dataset_train')
    if not os.path.exists(f'Dataset_test'):
        os.makedirs(f'Dataset_test')
    
    llm = LLM(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    ds = load_dataset("google/boolq")
    # {question, answer, passage}

    print(ds)

    # for data in ds['train']:
    #     print(data)

    ds_train = preprocess_dataset(ds['train'], ref, llm, target_layer)
    ds_val = preprocess_dataset(ds['validation'], ref, llm, target_layer)
    # ds = preprocess_dataset([ds['train'][2]], ref, llm, target_layer)

    # print(ds)
    
    # torch.save(ds, f'preprocessed.pt')
    if not ref:
        torch.save(ds_train, f'Dataset_train/{dataset_name}_layer{target_layer}.pt')
        torch.save(ds_val, f'Dataset_test/{dataset_name}_layer{target_layer}.pt')
        print(f"Preprocessed data successfully dumped into Dataset_train/{dataset_name}_layer{target_layer}.pt")
        print(f"Preprocessed data successfully dumped into Dataset_test/{dataset_name}_layer{target_layer}.pt")
    else:
        torch.save(ds_train, f'Dataset_train/{dataset_name}_passage_layer{target_layer}.pt')
        torch.save(ds_val, f'Dataset_test/{dataset_name}_passage_layer{target_layer}.pt')
        print(f"Preprocessed data successfully dumped into Dataset_train/{dataset_name}_passage_layer{target_layer}.pt")
        print(f"Preprocessed data successfully dumped into Dataset_test/{dataset_name}_passage_layer{target_layer}.pt")
    # print("Preprocessed data successfully dumped into preprocessed.pt")


