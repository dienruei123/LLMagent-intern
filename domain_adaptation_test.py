import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

import torch.optim as optim
import random

from lib.DaNN import *
from lib.args import parse_args

feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()

class_criterion = nn.BCELoss()
domain_criterion = nn.CrossEntropyLoss()

optimizer_F = optim.Adam(feature_extractor.parameters(), weight_decay=1e-6)
optimizer_C = optim.Adam(label_predictor.parameters(), weight_decay=1e-6)
optimizer_D = optim.Adam(domain_classifier.parameters(), weight_decay=1e-6)

categories = ['animals', 'cities', 'companies', 'elements', 'facts', 'inventions']
keys = dict()
for i, category in enumerate(categories):
    keys.update({category: i})

def add_domain_label(rawdata, domain, equal=False):
    # Labels:
    # Animals - 0
    # Cities - 1
    # Companies - 2
    # Elements - 3
    # Facts - 4
    # Inventions - 5
    
    for data in rawdata:
        data.update({'domain': keys[domain]})
    
    ds = rawdata
    if equal:
        ds_train_one = list(filter(lambda data : data['label'] == 1, rawdata))
        ds_train_zero = list(filter(lambda data : data['label'] == 0, rawdata))
        min_len = min(len(ds_train_one), len(ds_train_zero))
        ds_train_one = random.sample(ds_train_one, min_len)
        ds_train_zero = random.sample(ds_train_zero, min_len)
        ds = ds_train_one + ds_train_zero
    
    return ds

def test(testloader, device):
    '''
      Args:
        test_dataloader: test data的dataloader
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    running_D_loss, running_F_loss = [], []
    # total_hit, total_num = 0.0, 0.0
    
    mapping = {0: [1, 0], 1: [0, 1]}
    
    results = []
    tp, tn, fp, fn, total = 0, 0, 0, 0, 0

    for i, data in enumerate(testloader):
        # print(data)
        inputs, labels = data
        binary_labels = torch.tensor([mapping[i.detach().item()] for i in labels], dtype=torch.float).to(device)
        with torch.no_grad():
            # no need to use domain classifier
            outputs = label_predictor(feature_extractor(inputs))
            loss = class_criterion(outputs, binary_labels)

        running_F_loss.append(loss.detach().item())
        pred = torch.max(outputs, dim=1)
        # print(pred)
        result = torch.abs(pred.indices - labels)
        tp += len(result[(result == 0) & (pred.indices == 1)])
        tn += len(result[(result == 0) & (pred.indices == 0)])
        fp += len(result[(result == 1) & (pred.indices == 1)])
        fn += len(result[(result == 1) & (pred.indices == 0)])
        results.append(pred)
        total += len(pred.indices)
        # print(f'#{i+1}: Result: {pred.indices}, Label: {labels}')


    print(tp, tn, fp, fn)
    test_acc = (tp + tn) / total
    
    precision_true = tp / (tp + fp)
    recall_true = tp / (tp + fn)
    f1_true = 2 * precision_true * recall_true / (precision_true + recall_true)
    
    precision_false = tn / (tn + fn)
    recall_false = tn / (tn + fp)
    f1_false = 2 * precision_false * recall_false / (precision_false + recall_false)
    
    mean_test_loss = sum(running_F_loss)/len(running_F_loss)
    print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
    print(f'\n[Testing: ({dataset_name}, Layer {target_layer})]\nTest loss: {mean_test_loss:.4f}\nTest acc: {test_acc*100:.2f} %\n\
[True stats]\nF1 Score: {f1_true*100:.2f}\nRecall: {recall_true:.4f}\nPrecision: {precision_true:.4f}\n\
[False stats]\nF1 Score: {f1_false*100:.2f}\nRecall: {recall_false:.4f}\nPrecision: {precision_false:.4f}')


if __name__ == '__main__':
    args = parse_args()
    target_layer = args.layer
    ref = args.ref
    lamb = args.lamb
    
    ds_test_path = f'Dataset_test/boolq_layer{target_layer}.pt'
    dataset_name = 'boolq'
    
    if ref == True:
        ds_test_path = f'Dataset_test/boolq_passage_layer{target_layer}.pt'
        dataset_name = 'boolq_passage'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_ds = torch.load(ds_test_path, weights_only=True)
    # print(test_ds)

    test_hidden_states = torch.stack([item['hidden_state'] for item in test_ds])
    test_labels = torch.tensor([item['label'] for item in test_ds], device=device)

    # Split the dataset
    test_dataset = TensorDataset(test_hidden_states, test_labels)

    print("Test Labels: ", str(torch.sum(test_labels).item()) + '/' + \
        str(test_labels.shape[0]), f"({(torch.sum(test_labels).item() / test_labels.shape[0]):.4f})")

    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    result = []
    
    # lamb = 0.3
    
    # domain adaptation
    # label_predictor.load_state_dict(torch.load(f'./models/predictor_model_lamb_{lamb}_layer_{target_layer}.bin', weights_only=True))
    # feature_extractor.load_state_dict(torch.load(f'./models/extractor_model_lamb_{lamb}_layer_{target_layer}.bin', weights_only=True))
    
    # domain adaptation equal
    # label_predictor.load_state_dict(torch.load(f'./models/predictor_model_equal_lamb_{lamb}_layer_{target_layer}.bin', weights_only=True))
    # feature_extractor.load_state_dict(torch.load(f'./models/extractor_model_equal_lamb_{lamb}_layer_{target_layer}.bin', weights_only=True))
    
    # domain adaptation oneepoch
    label_predictor.load_state_dict(torch.load(f'./models/predictor_model_one_lamb_{lamb}_layer_{target_layer}.bin', weights_only=True))
    feature_extractor.load_state_dict(torch.load(f'./models/extractor_model_one_lamb_{lamb}_layer_{target_layer}.bin', weights_only=True))
    
    label_predictor.eval()
    feature_extractor.eval()
    
    test(testloader, device)
    

    # import pandas as pd
    # result = np.concatenate(result)

    # # Generate your submission
    # df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    # df.to_csv('DaNN_submission.csv',index=False)