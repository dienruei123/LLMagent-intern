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
import numpy as np

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

def test_vote(candidates: list, testloaders: list, models: list, device):
    criterion = nn.BCELoss()
    mapping = {0: [1, 0], 1: [0, 1]}
    loss_record = []

    # F1 score
    tp, tn, fp, fn, total = 0, 0, 0, 0, 0
    predictions = list()
    ground_truth = list()
    for idx in range(len(candidates)):
        model = models[idx]
        testloader = testloaders[idx]
        results = []
        ground_truth = []
        
        for i, data in enumerate(testloader):
            inputs, labels = data
            binary_labels = torch.tensor([mapping[i.detach().item()] for i in labels], dtype=torch.float).to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, binary_labels)

            # loss_record.append(loss.detach().item())
            pred = torch.max(outputs, dim=1)
            # print(pred)
            # result = torch.abs(pred.indices - labels)
            # tp += len(result[(result == 0) & (pred.indices == 1)])
            # tn += len(result[(result == 0) & (pred.indices == 0)])
            # fp += len(result[(result == 1) & (pred.indices == 1)])
            # fn += len(result[(result == 1) & (pred.indices == 0)])
            # print(result.cpu())
            results += pred.indices.cpu()
            ground_truth += [i.detach().item() for i in labels]
            # print(f'#{i+1}: Result: {pred.indices}, Label: {labels}')
        predictions.append(results)
    
    predictions = np.transpose(predictions)
    print(predictions.shape)
    # print(predictions, ground_truth)
    total = predictions.shape[0]
    
    final_results = []
    for answers in predictions:
        final_result = np.bincount(answers).argmax()
        final_results.append(final_result)
        
    final_results = np.array(final_results)
    ground_truth = np.array(ground_truth)
        
    result = np.array(np.abs(final_results - ground_truth))
    tp += len(result[(result == 0) & (final_results == 1)])
    tn += len(result[(result == 0) & (final_results == 0)])
    fp += len(result[(result == 1) & (final_results == 1)])
    fn += len(result[(result == 1) & (final_results == 0)])
        
    test_acc = (tp + tn) / total
    
    precision_true = tp / (tp + fp)
    recall_true = tp / (tp + fn)
    f1_true = 2 * precision_true * recall_true / (precision_true + recall_true)
    
    precision_false = tn / (tn + fn)
    recall_false = tn / (tn + fp)
    f1_false = 2 * precision_false * recall_false / (precision_false + recall_false)
    
    print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
    print(f'\n[Testing: (Voting, Layer {candidates})]\nTest acc: {test_acc*100:.2f} %\n\
[True stats]\nF1 Score: {f1_true*100:.2f}\nRecall: {recall_true:.4f}\nPrecision: {precision_true:.4f}\n\
[False stats]\nF1 Score: {f1_false*100:.2f}\nRecall: {recall_false:.4f}\nPrecision: {precision_false:.4f}')
    
if __name__ == '__main__':
    # args = parse_args()
    # target_layer, dataset_name, model_name = args.layer, args.dataset, args.model_name
    # ds_train_path = f'Dataset_train/{dataset_name}_layer{target_layer}.pt'
    # 15, 19, 23, 27
    candidate_layers = [15, 19, 23]
    candidate_layers = [15, 19, 27]
    candidate_layers = [15, 23, 27]
    candidate_layers = [19, 23, 27]
    # ds_test_paths = [f'Dataset_test/generated_layer{layer}.pt' for layer in candidate_layers]
    # ds_test_paths = [f'Dataset_test/boolq_layer{layer}.pt' for layer in candidate_layers]
    ds_test_paths = [f'Dataset_test/boolq_passage_layer{layer}.pt' for layer in candidate_layers]
    
    # if dataset_name == 'generated':
    #     raise Exception("'Generated' dataset is not accepted for train set.")
    
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    
    models = [TrueFalseClassifier() for _ in range(len(candidate_layers))]
    for idx, model in enumerate(models):
        model.to(device)
        model.load_state_dict(torch.load(f'./models/single_boolq_ref_equal_{candidate_layers[idx]}.ckpt', weights_only=True))
        model.eval()
    
    # train_ds = torch.load(ds_train_path, weights_only=True)
    test_ds = [torch.load(test_path, weights_only=True) for test_path in ds_test_paths]
    
    # print(test_ds)
    # train_ds = split_train_dataset(train_ds)
    
    # Extract hidden_states and labels
    # shape: [len(candidate_layers), ...]
    test_hidden_states = [torch.stack([item['hidden_state'] for item in ds]) for ds in test_ds]
    test_labels = [torch.tensor([item['label'] for item in ds], device=device) for ds in test_ds]

    # Create TensorDataset
    # train_dataset = TensorDataset(train_hidden_states, train_labels)
    # train_size = int(0.8 * len(train_dataset))
    # val_size = len(train_dataset) - train_size

    # Split the dataset
    # train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    test_dataset = list()
    for idx in range(len(candidate_layers)):
        test_dataset.append(TensorDataset(test_hidden_states[idx], test_labels[idx]))
    # print("Train Labels: ", str(torch.sum(train_labels).item()) + '/' + \
    #     str(train_labels.shape[0]), f"({(torch.sum(train_labels).item() / train_labels.shape[0]):.4f})")

    print("Test Labels: ", str(torch.sum(test_labels[0]).item()) + '/' + \
        str(test_labels[0].shape[0]), f"({(torch.sum(test_labels[0]).item() / test_labels[0].shape[0]):.4f})")

    # trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # validloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    testloaders = [DataLoader(dataset, batch_size=16, shuffle=False) for dataset in test_dataset]
    
    # trainer(trainloader, validloader, model, device, model_name)

    # Testing
    test_vote(candidate_layers, testloaders, models, device)
    del models
