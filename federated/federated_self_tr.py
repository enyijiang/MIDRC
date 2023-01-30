'''
Code is adapted from the following link:
https://github.com/uiuc-federated-learning/ml-fault-injector/blob/master/federated.py
'''

import argparse
import copy
import json
import os
import time
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
from imgaug import augmenters as iaa
import imgaug as ia
from PIL import Image

from models.resnet import ResNetClassifier, ResNetOrig
from models.cnn import CNN
from utils import constants
from utils.data_sampler import get_subset_indices, get_train_valid_indices
from utils.utils import deterministic
from dataset import MidrcDataset
from torch.multiprocessing import Pool

def set_device():
    global device
    device = torch.device(0 if torch.cuda.is_available() else "cpu")

def show_image(dataset, index):
    plt.gray()
    plt.imshow(dataset.__getitem__(index)[0].reshape((28,28)))
    plt.show()

def load_clients_data(args, states, transform):
    dls = {'train':[], 'test':[]}
    for mode in ['train', 'test']:
        for state in states:
            if mode == 'test':
                test_dataset = MidrcDataset(os.path.join(constants.STATE_DATA_DIR, f'MIDRC_table_{state}_{mode}.csv'), args.data_aug_times, transform[mode], 100)
                dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.source_batch_size, shuffle=False, drop_last=args.no_drop_last)
            else:
                train_dataset = MidrcDataset(os.path.join(constants.STATE_DATA_DIR, f'MIDRC_table_{state}_{mode}.csv'), args.data_aug_times, transform[mode], 400)
                dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.source_batch_size, shuffle=True, drop_last=args.no_drop_last)
            dls[mode].append(dl)
    return dls

def load_server_data(args, state, transform):
    dls = {'train':[], 'test':[]}
    for mode in ['train', 'test']:
        train_dataset = MidrcDataset(os.path.join(constants.STATE_DATA_DIR, f'MIDRC_table_{state}_{mode}.csv'), args.data_aug_times, transform[mode])
        if mode == 'test':
            dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.target_batch_size, shuffle=False, drop_last=args.no_drop_last)
        else:
            subset_idx = get_subset_indices(train_dataset, args.n_target_samples, args.data_sampler_seed)
            rest_idx = list(set(range(len(train_dataset.labels))) - set(subset_idx))
            subset = torch.utils.data.Subset(train_dataset, subset_idx)
            subset_rest = torch.utils.data.Subset(train_dataset, rest_idx)
            dl = torch.utils.data.DataLoader(subset, batch_size=args.target_batch_size, shuffle=True, drop_last=args.no_drop_last)
        dls[mode].append(dl)
    return dls, subset_rest

def train(args, da_phase, model, criterion: torch.nn.Module, train_dl):
    global device

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.source_lr if da_phase=='source' else args.target_lr)
    model.train()
    num_epochs = args.num_source_epochs if da_phase == 'source' else args.num_target_epochs

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        y_true_list = list()
        y_pred_list = list()


        with tqdm(train_dl, unit="batch") as tepoch:
            for sample in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs = sample['img'].to(device)
                labels = sample['label'].unsqueeze(1).to(device)
                #print(lbl)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                labels = labels.type_as(outputs)
                probs = torch.sigmoid(outputs)
                preds = probs > 0.5
                loss = criterion(probs, labels)
                tepoch.set_postfix(loss=loss.item())
                for i in range(len(outputs)):
                    y_true_list.append(labels[i].cpu().data.tolist())
                    y_pred_list.append(probs[i].cpu().data.tolist())

                # Backward pass
                loss.backward()
                optimizer.step()

                # Keep track of performance metrics (loss is mean-reduced)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(y_true_list)
            epoch_acc = float(running_corrects) / len(y_true_list)
            auc = roc_auc_score(y_true_list, y_pred_list)

    # Keep track of current training loss and accuracy
    final_train_loss = epoch_loss
    final_train_acc = epoch_acc
    final_train_auc = auc

    return model, (final_train_loss, final_train_acc, final_train_auc)

def self_train(args, da_phase, model, criterion: torch.nn.Module, train_dl, unlabel_dl):
    global device

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.target_lr)
    model.train()
    num_epochs = args.num_target_epochs

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        y_true_list = list()
        y_pred_list = list()

        # train on the original finetune dataset
        with tqdm(train_dl, unit="batch") as tepoch:
            for sample in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs = sample['img'].to(device)
                labels = sample['label'].unsqueeze(1).to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                labels = labels.type_as(outputs)
                probs = torch.sigmoid(outputs)
                preds = probs > 0.5
                loss = criterion(probs, labels)
                tepoch.set_postfix(loss=loss.item())
                for i in range(len(outputs)):
                    y_true_list.append(labels[i].cpu().data.tolist())
                    y_pred_list.append(probs[i].cpu().data.tolist())

                # Backward pass
                loss.backward()
                optimizer.step()

                # Keep track of performance metrics (loss is mean-reduced)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(y_true_list)
            epoch_acc = float(running_corrects) / len(y_true_list)
            auc = roc_auc_score(y_true_list, y_pred_list)

        # train on pseudo labels
        with tqdm(unlabel_dl, unit='batch') as tepoch:
            for sample in tepoch:
                tepoch.set_description(f"Self-training Epoch {epoch}")
                x_unlabeled = sample['img'].to(device)
                # get pseudo labels
                model.eval()
                output_unlabeled = model(x_unlabeled)                
                probs = torch.sigmoid(output_unlabeled)
                pseudo_labeled = (probs > 0.5).long()
                # Now calculate the unlabeled loss using the pseudo label
                model.train()
                optimizer.zero_grad()
                outputs = model(x_unlabeled)
                probs = torch.sigmoid(outputs)
                pseudo_labeled = pseudo_labeled.type_as(outputs)
                # preds = probs > 0.5
                unlabeled_loss = af * criterion(probs, pseudo_labeled)           
                tepoch.set_postfix(loss=unlabeled_loss.item())       
                # Backpropogate
                
                unlabeled_loss.backward()
                optimizer.step()

    # Keep track of current training loss and accuracy
    final_train_loss = epoch_loss
    final_train_acc = epoch_acc
    final_train_auc = auc

    return model, (final_train_loss, final_train_acc, final_train_auc)

def test(args: argparse.Namespace, model: torch.nn.Module,
         criterion: torch.nn.Module, test_loader: torch.utils.data.DataLoader):
    global device

    model.to(device)
    model.eval()
    trial_results = dict()

    running_loss = 0.0
    running_corrects = 0
    y_true_list = list()
    y_pred_list = list()

    # Iterate over dataloader
    for sample in test_loader:
        inputs = sample['img'].to(device)
        labels = sample['label'].unsqueeze(1).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            labels = labels.type_as(outputs)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5
            loss = criterion(probs, labels)

            for i in range(len(outputs)):
                y_true_list.append(labels[i].cpu().data.tolist())
                y_pred_list.append(probs[i].cpu().data.tolist())

            # Keep track of performance metrics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    test_loss = running_loss / len(y_true_list)
    test_acc = float(running_corrects) / len(y_true_list)
    auc = roc_auc_score(y_true_list, y_pred_list)

    print('Test Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(
          test_loss, test_acc, auc), flush=True)
    print(flush=True)

    return (test_loss, test_acc, auc) 

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

if __name__ == '__main__':
    set_device()
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_dir', type=str, default='fl/self_train_base')
    parser.add_argument('--iter_idx', type=int, default=0)
    parser.add_argument('--resnet', type=str, default='resnet18')
    parser.add_argument('--load_trained_model', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--num_source_epochs', type=int, default=1)
    parser.add_argument('--num_target_epochs', type=int, default=1)
    parser.add_argument('--num_global_epochs', type=int, default=100)
    parser.add_argument('--source_lr', type=float, default=0.0001)
    parser.add_argument('--target_lr', type=float, default=0.00005)
    parser.add_argument('--source_batch_size', type=int, default=8)
    parser.add_argument('--target_batch_size', type=int, default=8)
    parser.add_argument('--no_drop_last', action='store_false')
    parser.add_argument('--train_seed', type=int, default=8)
    parser.add_argument('--data_sampler_seed', type=int, default=8)
    parser.add_argument('--n_source_samples', type=int, default=500)
    parser.add_argument('--n_target_samples', type=int, default=20)
    parser.add_argument('--n_valid_samples', type=int, default=500)
    parser.add_argument('--valid_fraction', type=float, default=None)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--data_aug_times', type=int, default=1)

    args = parser.parse_args()
    timestamp = time.strftime("%Y-%m-%d-%H%M")
    if args.iter_idx != 0:  # If running multiple iters, store in same dir
        exp_dir = os.path.join('experiments', args.exp_dir)
        if not os.path.isdir(exp_dir):
            raise OSError('Specified directory does not exist!')
    else:  # Otherwise, create a new dir
        exp_dir = os.path.join('experiments', args.exp_dir)
        os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, f'args_{args.iter_idx}.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    deterministic(args.train_seed)
    

    transform = {
        'train':
        transforms.Compose([
            # ImgAugTransform(),
            # lambda x: Image.fromarray(x),
            transforms.Resize((256, 256)),
            # transforms.RandomResizedCrop((224), scale=(0.9, 1)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True),
            ]),
        'test':
        transforms.Compose([
            # lambda x: Image.fromarray(x),
            # transforms.Resize((256, 256)),
            # transforms.ColorJitter(contrast=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True),
            ])
    }

    # initialize datalaoders, models, optimizer, criterions
    # ['IL', 'NC', 'CA', 'IN', 'TX']
    clients = ['CA', 'IN', 'TX']
    server = 'IL'
    num_clients = len(clients)
    dict_client = dict()
    for i in range(num_clients):
        dict_client.update({i: []})

    clients_dls = load_clients_data(args, clients, transform)
    server_dls, unlabelled_dataset = load_server_data(args, server, transform)
    unlabel_indices = list(range(len(unlabelled_dataset)))
    # np.random.shuffle(unlabel_indices)


    # print(clients_dls, server_dls)

    global_model = ResNetClassifier(resnet=args.resnet, hidden_size=args.hidden_size)
    local_models = [ResNetClassifier(resnet=args.resnet, hidden_size=args.hidden_size) for _ in range(num_clients)]
    global_model_dict = global_model.state_dict()

    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    clients_results = dict()
    clients_results['train'] = dict()
    clients_results['test'] = dict()
    clients_results['train']['loss'] = copy.deepcopy(dict_client)
    clients_results['train']['acc'] = copy.deepcopy(dict_client)
    clients_results['train']['auc'] = copy.deepcopy(dict_client)
    clients_results['test']['loss'] = copy.deepcopy(dict_client)
    clients_results['test']['acc'] = copy.deepcopy(dict_client)
    clients_results['test']['auc'] = copy.deepcopy(dict_client)

    server_results = dict()
    server_results['train'] = dict()
    server_results['test'] = dict()
    server_results['train']['loss'] = []
    server_results['train']['acc'] = []
    server_results['train']['auc'] = []
    server_results['test']['loss'] = []
    server_results['test']['acc'] = []
    server_results['test']['auc'] = []
 


    for i in range(args.num_global_epochs):
        af = 3*i / args.num_global_epochs 
        # training local models
        for idx in range(num_clients):
            local_models[idx].load_state_dict(global_model_dict)
            local_models[idx], (loss, acc, auc) = train(args, 'source', local_models[idx], criterion, clients_dls['train'][idx])
            clients_results['train']['loss'][idx].append(loss)
            clients_results['train']['acc'][idx].append(acc)
            clients_results['train']['auc'][idx].append(auc)
        
        # averaging the weights
        global_model_dict = average_weights([model.state_dict() for model in local_models])
        global_model.load_state_dict(global_model_dict)
        cur_indices = np.random.choice(unlabel_indices, args.n_target_samples, replace=False)
        print(len(unlabel_indices), len(cur_indices))
        cur_sampler = SubsetRandomSampler(cur_indices)
        unlabel_dl = torch.utils.data.DataLoader(unlabelled_dataset, shuffle=False, batch_size=args.target_batch_size, sampler=cur_sampler)
        global_model, (loss, acc, auc) = self_train(args, 'target', global_model, criterion, server_dls['train'][0], unlabel_dl)
        # print(loss, acc, auc)
        server_results['train']['loss'].append(loss)
        server_results['train']['acc'].append(acc)
        server_results['train']['auc'].append(auc)

        # print('testing each local model\'s accuracy on target domain')
        # for idx in range(num_clients): 
        #     (loss, acc, auc) = test(args, local_models[idx], criterion, server_dls['test'][0])

        print('testing each local model\'s accuracy on their own source domain')
        for idx in range(num_clients): 
            (loss, acc, auc) = test(args, local_models[idx], criterion, clients_dls['test'][idx])
            clients_results['test']['loss'][idx].append(loss)
            clients_results['test']['acc'][idx].append(acc)
            clients_results['test']['auc'][idx].append(auc)
        
        print('testing global model on its target domain')
        (loss, acc, auc) = test(args, global_model, criterion, server_dls['test'][0])
        server_results['test']['loss'].append(loss)
        server_results['test']['acc'].append(acc)
        server_results['test']['auc'].append(auc)

    # saving results
    with open(os.path.join(exp_dir,(f'clients_results_{args.iter_idx}.json')), 'w') as fp:
            json.dump(clients_results, fp, indent=4)
    fp.close()
    
    with open(os.path.join(exp_dir,(f'server_results_{args.iter_idx}.json')), 'w') as fp:
            json.dump(server_results, fp, indent=4)
    fp.close()

    torch.save(global_model.state_dict(),os.path.join(exp_dir,f'server_checkpoint_{args.iter_idx}.pt'))

    for idx, model in enumerate(local_models):
        torch.save(model.state_dict(),os.path.join(exp_dir,f'client_{idx}_checkpoint_{args.iter_idx}.pt'))
    