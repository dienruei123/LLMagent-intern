from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import torch
from lib.LLM import LLM
from lib.args import parse_args

import argparse
import os
import re

if __name__ == '__main__':
    args = parse_args()
    target_layer = args.layer
    ds = []
    for dirpath, dirnames, filenames in os.walk('Dataset_train'):
        # print(f'Current Directory: {dirpath}')
        for file in filenames:
            # print(f'File: {file}')
            if re.search(rf'(?<!\d){str(target_layer)}(?!\d)', file):
            # if str(target_layer) in file:
                print(f'Concat file: {file}')
                tmp_ds = torch.load(f'Dataset_train/{file}', weights_only=True)
                ds += tmp_ds
                # print(ds)
            
    # print(ds)
    torch.save(ds, f'Dataset_train/all_layer{target_layer}.pt')
    # torch.save(ds, f'preprocessed.pt')

    print(f"Preprocessed data successfully dumped into Dataset_train/all_layer{target_layer}.pt")
    # print("Preprocessed data successfully dumped into preprocessed.pt")


