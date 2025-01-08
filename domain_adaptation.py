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

def train_epoch(trainloader, device, lamb):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: control the balance of domain adaptatoin and classification.
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0
    
    mapping = {0: [1, 0], 1: [0, 1]}

    for i, data in enumerate(trainloader):
        # print(data)
        inputs, labels, domains = data
        binary_labels = torch.tensor([mapping[i.detach().item()] for i in labels], dtype=torch.float).to(device)

        # Step 1 : train domain classifier
        feature = feature_extractor(inputs)
        # We don't need to train feature extractor in step 1.
        # Thus we detach the feature neuron to avoid backpropgation.
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domains)
        running_D_loss += loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : train feature extractor and label classifier
        class_logits = label_predictor(feature)
        domain_logits = domain_classifier(feature)
        # loss = cross entropy of classification - lamb * domain binary cross entropy.
        #  The reason why using subtraction is similar to generator loss in disciminator of GAN
        loss = class_criterion(class_logits, binary_labels) - lamb * domain_criterion(domain_logits, domains)
        running_F_loss+= loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()
        
        pred = torch.max(class_logits, dim=1)
        result = torch.abs(pred.indices - labels)
        result = result[result == 0]
        total_hit += len(result)
        total_num += len(pred.indices)
        print(i, end='\r')

    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num


if __name__ == '__main__':
    args = parse_args()
    target_layer = args.layer
    lamb = args.lamb
    
    # Equal label tag
    is_equal = args.equal
    
    ds_train_path, ds_test_path = '', f'Dataset_test/generated_layer{target_layer}.pt'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_ds: list = None
    test_ds = torch.load(ds_test_path, weights_only=True)
    # print(test_ds)
    
    for category in categories:
        ds_train_path = f'Dataset_train/{category}_layer{target_layer}.pt'
        train_tmpds = torch.load(ds_train_path, weights_only=True)
    
        # add domain label
        train_tmpds = add_domain_label(train_tmpds, category, equal=is_equal)
        
        if train_ds is None:
            train_ds = list()
            
        train_ds.extend(train_tmpds)
    
    # Extract hidden_states and labels
    train_hidden_states = torch.stack([item['hidden_state'] for item in train_ds])
    train_labels = torch.tensor([item['label'] for item in train_ds], device=device)
    train_domains = torch.tensor([item['domain'] for item in train_ds], device=device)

    test_hidden_states = torch.stack([item['hidden_state'] for item in test_ds])
    test_labels = torch.tensor([item['label'] for item in test_ds], device=device)

    # Create TensorDataset
    train_dataset = TensorDataset(train_hidden_states, train_labels, train_domains)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # Split the dataset
    # train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    test_dataset = TensorDataset(test_hidden_states, test_labels)
    print("Train Labels: ", str(torch.sum(train_labels).item()) + '/' + \
        str(train_labels.shape[0]), f"({(torch.sum(train_labels).item() / train_labels.shape[0]):.4f})")

    print("Test Labels: ", str(torch.sum(test_labels).item()) + '/' + \
        str(test_labels.shape[0]), f"({(torch.sum(test_labels).item() / test_labels.shape[0]):.4f})")

    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # validloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # train 200 epochs
    for epoch in range(80):
        # You should chooose lambda cleverly.
        # lamb = 0.3
        train_D_loss, train_F_loss, train_acc = train_epoch(trainloader, device, lamb=lamb)

        if is_equal:
            torch.save(feature_extractor.state_dict(), f'models/extractor_model_equal_lamb_{lamb}_layer_{target_layer}.bin')
            torch.save(label_predictor.state_dict(), f'models/predictor_model_equal_lamb_{lamb}_layer_{target_layer}.bin')
        else:
            torch.save(feature_extractor.state_dict(), f'models/extractor_model_lamb_{lamb}_layer_{target_layer}.bin')
            torch.save(label_predictor.state_dict(), f'models/predictor_model_lamb_{lamb}_layer_{target_layer}.bin')

        print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch+1, train_D_loss, train_F_loss, train_acc))

    print("Training finished")

    # result = []
    # label_predictor.eval()
    # feature_extractor.eval()
    # for i, (test_data, _) in enumerate(testloader):
    #     test_data = test_data.cuda()

    #     class_logits = label_predictor(feature_extractor(test_data))

    #     x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    #     result.append(x)

    # import pandas as pd
    # result = np.concatenate(result)

    # # Generate your submission
    # df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    # df.to_csv('DaNN_submission.csv',index=False)