import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import math
import os
import random

from lib.classifier import TrueFalseClassifier
from lib.args import parse_args

from datasets import load_dataset, DatasetDict
import gc

def split_train_dataset(ds_train: list):
    # TODO: train set true-false label ratio 1:1
    
    # train_testvalid = ds['train'].train_test_split(test_size=0.2)
    # # Split the 10% test + valid in half test, half valid
    # test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    # # gather everyone if you want to have a single DatasetDict
    # ds = DatasetDict({
    #     'train': train_testvalid['train'],
    #     'test': test_valid['test'],
    #     'valid': test_valid['train']}
    # )
    # ds = DatasetDict({
    #     'train': [train_testvalid['train'][0]],
    #     'test': [test_valid['test'][0]],
    #     'valid': [test_valid['train'][0]]}
    # )
    
    # print(ds)    
    ds_train_one = list(filter(lambda data : data['label'] == 1, ds_train))
    ds_train_zero = list(filter(lambda data : data['label'] == 0, ds_train))
    min_len = min(len(ds_train_one), len(ds_train_zero))
    ds_train_one = random.sample(ds_train_one, min_len)
    ds_train_zero = random.sample(ds_train_zero, min_len)
    ds = ds_train_one + ds_train_zero
    
    random.shuffle(ds)
    
    return ds

def test(testloader, model, device):
    criterion = nn.BCELoss()
    mapping = {0: [1, 0], 1: [0, 1]}
    model.eval() # Set your model to evaluation mode.
    loss_record = []
    results = []

    # F1 score
    tp, tn, fp, fn, total = 0, 0, 0, 0, 0
    
    for i, data in enumerate(testloader):
        inputs, labels = data
        binary_labels = torch.tensor([mapping[i.detach().item()] for i in labels], dtype=torch.float).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, binary_labels)

        loss_record.append(loss.detach().item())
        pred = torch.max(outputs, dim=1)
        # print(pred)
        result = torch.abs(pred.indices - labels)
        tp += len(result[(result == 0) & (pred.indices == 1)])
        tn += len(result[(result == 0) & (pred.indices == 0)])
        fp += len(result[(result == 1) & (pred.indices == 1)])
        fn += len(result[(result == 1) & (pred.indices == 0)])
        results.append(pred)
        total += len(pred.indices)
        print(f'#{i+1}: Result: {pred.indices}, Label: {labels}')
    
    print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
    test_acc = (tp + tn) / total
    
    precision_true = tp / (tp + fp) if tp + fp > 0 else 0
    recall_true = tp / (tp + fn) if tp + fn > 0 else 0
    f1_true = 2 * precision_true * recall_true / (precision_true + recall_true) if precision_true + recall_true > 0 else 0
    
    precision_false = tn / (tn + fn) if tn + fn > 0 else 0
    recall_false = tn / (tn + fp) if tn + fp > 0 else 0
    f1_false = 2 * precision_false * recall_false / (precision_false + recall_false) if precision_false + recall_false > 0 else 0
    
    mean_test_loss = sum(loss_record)/len(loss_record)
    # print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
    print(f'\n[Testing: ({dataset_name}, Layer {target_layer})]\nTest loss: {mean_test_loss:.4f}\nTest acc: {test_acc*100:.2f} %\n\
[True stats]\nF1 Score: {f1_true*100:.2f}\nRecall: {recall_true:.4f}\nPrecision: {precision_true:.4f}\n\
[False stats]\nF1 Score: {f1_false*100:.2f}\nRecall: {recall_false:.4f}\nPrecision: {precision_false:.4f}')
    
#     with open('results.txt', 'a') as file:
#         file.write(f'\n[Testing: ({dataset_name}, Layer {target_layer})]\nTest loss: {mean_test_loss:.4f}\nTest acc: {test_acc:.4f}\n\
# [True stats]\nF1 Score: {f1_true:.4f}\nRecall: {recall_true:.4f}\nPrecision: {precision_true:.4f}\n\
# [False stats]\nF1 Score: {f1_false:.4f}\nRecall: {recall_false:.4f}\nPrecision: {precision_false:.4f}\n')  
    
if __name__ == '__main__':
    args = parse_args()
    target_layer, dataset_name, model_name, ref = args.layer, args.dataset, args.model_name, args.ref
    print(ref, type(ref))
    
    ds_train_path, ds_test_path = '', ''
    if 'boolq' in dataset_name:
        if ref == True:
            dataset_name = 'boolq_passage'
        ds_train_path = f'Dataset_train/{dataset_name}_layer{target_layer}.pt'
        ds_test_path = f'Dataset_test/{dataset_name}_layer{target_layer}.pt'
    # elif dataset_name == 'generated':
    #     raise Exception("'Generated' dataset is not accepted for train set.")
    elif 'generated' in dataset_name:
        ds_test_path = f'Dataset_test/{dataset_name}_layer{target_layer}.pt'
    else:
        ds_test_path = f'Dataset_train/{dataset_name}_layer{target_layer}.pt'
        # ds_test_path = f'Dataset_test/generated_layer{target_layer}.pt'
    
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    model = TrueFalseClassifier()
    model = model.to(device)
    
    # train_ds = torch.load(ds_train_path, weights_only=True)
    test_ds = torch.load(ds_test_path, weights_only=True)
    
    # print(test_ds)
    
    # if equal label distribution
    # train_ds = split_train_dataset(train_ds)
    
    # Extract hidden_states and labels
    # train_hidden_states = torch.stack([item['hidden_state'] for item in train_ds])
    # train_labels = torch.tensor([item['label'] for item in train_ds], device=device)

    test_hidden_states = torch.stack([item['hidden_state'] for item in test_ds])
    test_labels = torch.tensor([item['label'] for item in test_ds], device=device)

    # Create TensorDataset
    # train_dataset = TensorDataset(train_hidden_states, train_labels)
    # train_size = int(0.8 * len(train_dataset))
    # val_size = len(train_dataset) - train_size

    # Split the dataset
    # train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    test_dataset = TensorDataset(test_hidden_states, test_labels)
    # print("Train Labels: ", str(torch.sum(train_labels).item()) + '/' + \
    #     str(train_labels.shape[0]), f"({(torch.sum(train_labels).item() / train_labels.shape[0]):.4f})")

    print("Test Labels: ", str(torch.sum(test_labels).item()) + '/' + \
        str(test_labels.shape[0]), f"({(torch.sum(test_labels).item() / test_labels.shape[0]):.4f})")

    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Testing
    model.load_state_dict(torch.load(f'./models/{model_name}.ckpt', weights_only=True))
    with open('results.txt', 'a') as file:
        file.write(f'\n[MODEL NAME: {model_name}.ckpt]')
    
    test(testloader, model, device)
    del model
