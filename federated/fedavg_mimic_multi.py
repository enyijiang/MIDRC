

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
from tqdm import tqdm
from imgaug import augmenters as iaa
import imgaug as ia
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler

from models.resnet_mlt import ResnetMultiTaskNet
from models.cnn import CNN
from utils import constants
from utils.data_sampler import get_subset_indices, get_train_valid_indices
from utils.utils import deterministic
from datasets.MIMIC import MimicMLTDataset
from torch.multiprocessing import Pool

def set_device():
    global device
    device = torch.device(0 if torch.cuda.is_available() else "cpu")

def show_image(dataset, index):
    plt.gray()
    plt.imshow(dataset.__getitem__(index)[0].reshape((28,28)))
    plt.show()

def load_data(args, domains, transform):
    dls = {'train':[], 'test':[]}
    target = domains[0]
    for mode in ['train', 'test']:
        for domain in domains:
            if mode == 'train':
                if domain != target:
                    train_dataset = MimicMLTDataset(os.path.join(constants.MIMIC_DATA_DIR, f'mimic_table_{domain}_{mode}.csv'), args.data_aug_times, transform[mode], n_samples=args.n_source_samples)
                else:
                    train_dataset = MimicMLTDataset(os.path.join(constants.MIMIC_DATA_DIR, f'mimic_table_{domain}_{mode}.csv'), args.data_aug_times, transform[mode])
            else:
                train_dataset = MimicMLTDataset(os.path.join(constants.MIMIC_DATA_DIR, f'mimic_table_{domain}_{mode}.csv'), args.data_aug_times, transform[mode])

            dls[mode].append(train_dataset)
    return dls

def train(args, da_phase, model, criterions, train_dl, finetuneing=False, cur_global_epoch=0):
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
            for batch_idx, sample in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                imgs, targets = sample
                imgs = imgs.to(device)
                targets = targets[:len(criterions)]

                for i in range(len(targets)):
                    targets[i] = targets[i].to(device)
                labels = targets[0].unsqueeze(1)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                preds_all = model(imgs)
                outputs = preds_all[0]
                labels = labels.type_as(outputs)
                probs = torch.sigmoid(outputs)
                preds = probs > 0.5

                if not finetuneing:
                    loss = criterions[0](probs, labels)
                    for idx, c in enumerate(criterions[1:]):
                        loss += args.alpha * c(preds_all[idx+1], targets[idx+1])
                else:
                    loss = criterions[0](probs, labels)

                tepoch.set_postfix(loss=loss.item())
                for i in range(len(outputs)):
                    y_true_list.append(labels[i].cpu().data.tolist())
                    y_pred_list.append(probs[i].cpu().data.tolist())

                # Backward pass
                loss.backward()
                optimizer.step()

                # Keep track of performance metrics (loss is mean-reduced)
                running_loss += loss.item() * imgs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(y_true_list)
            epoch_acc = float(running_corrects) / len(y_true_list)
            auc = roc_auc_score(y_true_list, y_pred_list)

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
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels[0].unsqueeze(1).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs[0]
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

def average_weights(w, alpha):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key]).float()
        for i in range(len(w)):
            w_avg[key] += w[i][key] * alpha[i]
    return w_avg

def update_dict(old_model_dict, new_model_dict, alpha):
    new_w = copy.deepcopy(old_model_dict)
    for key in new_w.keys():
        new_w[key] = torch.zeros_like(new_w[key]).float()
        new_w[key] = old_model_dict[key] * alpha + new_model_dict[key] * (1-alpha)
    return new_w

def update_global(args, local_models_dict, old_global_model_dict, finetune_global_model_dict, clients_size, clients_size_frac):
    ret_dict = copy.deepcopy(old_global_model_dict)
    b = 0.5
    cos = torch.nn.CosineSimilarity()
    for key in ret_dict.keys():
        if ret_dict[key].shape != torch.Size([]):
            global_grad = finetune_global_model_dict[key] - old_global_model_dict[key]
            for idx, local_dict in enumerate(local_models_dict):
                local_grad = local_dict[key] - old_global_model_dict[key]
                cur_sim = cos(global_grad.reshape(1,-1), local_grad.reshape(1,-1))

                if cur_sim > 0:
                    ret_dict[key] = ret_dict[key] + b * (args.target_lr / args.source_lr) * ((args.n_target_samples/args.target_batch_size)/(clients_size[idx]/args.source_batch_size)) * clients_size_frac[idx] * cur_sim * local_grad
            ret_dict[key] = ret_dict[key] + (1-b) * (global_grad)
        else:
            ret_dict[key] = old_global_model_dict[key]
    return ret_dict

# get the grad updates
def get_model_updates(init_model, new_model):
    init = get_param_list(init_model)
    new = get_param_list(new_model)
    return (new - init).reshape(1, -1)

def get_param_list(model):
	m_dict = model.state_dict()
	param = []
	for key in m_dict.keys():
		param.append(np.linalg.norm(m_dict[key]))
	return np.array(param)

if __name__ == '__main__':
    set_device()
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_dir', type=str, default='fl_mimic')
    parser.add_argument('--iter_idx', type=int, default=0)
    parser.add_argument('--resnet', type=str, default='resnet18')
    parser.add_argument('--load_trained_model', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--num_source_epochs', type=int, default=1)
    parser.add_argument('--num_target_epochs', type=int, default=1)
    parser.add_argument('--num_global_epochs', type=int, default=60)
    parser.add_argument('--source_lr', type=float, default=0.0001)
    parser.add_argument('--target_lr', type=float, default=0.00002)
    parser.add_argument('--source_batch_size', type=int, default=32)
    parser.add_argument('--target_batch_size', type=int, default=32)
    parser.add_argument('--no_drop_last', action='store_true')
    parser.add_argument('--train_seed', type=int, default=8)
    parser.add_argument('--data_sampler_seed', type=int, default=8)
    parser.add_argument('--n_source_samples', type=int, default=500)
    parser.add_argument('--n_target_samples', type=int, default=100)
    parser.add_argument('--n_valid_samples', type=int, default=500)
    parser.add_argument('--valid_fraction', type=float, default=None)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--data_aug_times', type=int, default=1)
    parser.add_argument('--use_sim', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--partition_factor', type=int, default=1)

    args = parser.parse_args()
    timestamp = time.strftime("%Y-%m-%d-%H%M")

    transform = {
        'train':
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop((224), scale=(0.9, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True),
            ]),
        'test':
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True),
            ])
    }

    domains = ['Asian', 'Black', 'White', 'Hispanic']
    ds = load_data(args, domains, transform)
    
    for idx, server in enumerate(domains):
        clients = copy.deepcopy(domains)
        clients.remove(server)
        print(clients, server)
        clients_dls = {'train':[], 'test':[]}
        server_dls = {'train':[], 'test':[]}
        
        # for the client 
        
        # randperms = [torch.randperm(len(ds['train'][i])) for i in range(len(domains))]
        # factor = args.partition_factor
        # for f in range(factor):
        #     indices = [rp[f*int(len(rp)//factor):(f+1)*int(len(rp)//factor)] for rp in randperms]
        #     samplers = [SubsetRandomSampler(ind) for ind in indices]
        #     print([len(ind) for ind in indices])
        #     clients_dls['train'].extend([torch.utils.data.DataLoader(ds['train'][i], batch_size=args.source_batch_size, shuffle=False, drop_last=args.no_drop_last, sampler=samplers[i]) for i in range(len(domains)) if i != idx]) 
        #     clients_dls['test'].extend([torch.utils.data.DataLoader(ds['test'][i], batch_size=args.source_batch_size, shuffle=False, drop_last=args.no_drop_last) for i in range(len(domains)) if i != idx])

        # # for the server 
        # for mode in ['train', 'test']:
        #     if mode == 'test':
        #         server_dls[mode] = [torch.utils.data.DataLoader(ds[mode][i], batch_size=args.target_batch_size, shuffle=False, drop_last=args.no_drop_last) for i in range(len(domains)) if i == idx]    
        #     else:
        #         server_dls[mode] = [torch.utils.data.DataLoader(ds[mode][i], batch_size=args.target_batch_size, shuffle=True, drop_last=args.no_drop_last) for i in range(len(domains)) if i == idx]       
        
        for mode in ['train', 'test']:
            if mode == 'test':
                clients_dls[mode] = [torch.utils.data.DataLoader(ds[mode][i], batch_size=args.source_batch_size, shuffle=False, drop_last=args.no_drop_last) for i in range(len(domains)) if i != idx]
                server_dls[mode] = [torch.utils.data.DataLoader(ds[mode][i], batch_size=args.target_batch_size, shuffle=False, drop_last=args.no_drop_last) for i in range(len(domains)) if i == idx]    
            else:
                clients_dls[mode] = [torch.utils.data.DataLoader(ds[mode][i], batch_size=args.source_batch_size, shuffle=True, drop_last=args.no_drop_last) for i in range(len(domains)) if i != idx]
                server_dls[mode] = [torch.utils.data.DataLoader(ds[mode][i], batch_size=args.target_batch_size, shuffle=True, drop_last=args.no_drop_last) for i in range(len(domains)) if i == idx] 
        
        target_train = ds['train'][idx]
        randperm = torch.randperm(len(target_train))
        indices = randperm[:args.n_target_samples]
        rest_indices = randperm[args.n_target_samples:]
        cur_sampler = SubsetRandomSampler(indices)
        cur_sampler_rest = SubsetRandomSampler(rest_indices)
        perturb_dl = torch.utils.data.DataLoader(target_train, shuffle=False, batch_size=args.target_batch_size, sampler=cur_sampler)
        unlabeled_dl = torch.utils.data.DataLoader(target_train, shuffle=False, batch_size=args.target_batch_size, sampler=cur_sampler_rest)

        if args.iter_idx != 0:  # If running multiple iters, store in same dir
            exp_dir = os.path.join('experiments', args.exp_dir, server)
            if not os.path.isdir(exp_dir):
                raise OSError('Specified directory does not exist!')
        else:  # Otherwise, create a new dir
            exp_dir = os.path.join('experiments', args.exp_dir, server)
            os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, f'args_{args.iter_idx}.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
        deterministic(args.train_seed)
        

        # initialize datalaoders, models, optimizer, criterions
        
        num_clients = len(clients_dls['train'])
        dict_client = dict()
        for i in range(num_clients):
            dict_client.update({i: []})

        clients_size = [len(clients_dls['train'][i])*args.source_batch_size for i in range(num_clients)]
        clients_size_frac = np.array(clients_size) / sum(clients_size)
        print(clients_size_frac)

        global_model = ResnetMultiTaskNet(resnet=args.resnet, hidden_size=args.hidden_size, num_classes=[1,2,9])
        global_model.to(device)
        local_models = [ResnetMultiTaskNet(resnet=args.resnet, hidden_size=args.hidden_size, num_classes=[1,2,9]) for _ in range(num_clients)]

        criterion_dignosed = torch.nn.BCEWithLogitsLoss(reduction='mean')
        criterion_sex = torch.nn.CrossEntropyLoss()
        criterion_age = torch.nn.CrossEntropyLoss()
        criterions = [criterion_dignosed, criterion_age, criterion_age]
        clients_grads = [None] * num_clients
        cos_sim = [None] * num_clients
        global_model_dict = global_model.state_dict()

        clients_results = dict()
        clients_results['train'] = dict()
        clients_results['test_s'] = dict()
        clients_results['test_t'] = dict()
        clients_results['train']['loss'] = copy.deepcopy(dict_client)
        clients_results['train']['acc'] = copy.deepcopy(dict_client)
        clients_results['train']['auc'] = copy.deepcopy(dict_client)
        clients_results['test_s']['loss'] = copy.deepcopy(dict_client)
        clients_results['test_s']['acc'] = copy.deepcopy(dict_client)
        clients_results['test_s']['auc'] = copy.deepcopy(dict_client)
        clients_results['test_t']['loss'] = copy.deepcopy(dict_client)
        clients_results['test_t']['acc'] = copy.deepcopy(dict_client)
        clients_results['test_t']['auc'] = copy.deepcopy(dict_client)

        server_results = dict()
        server_results['train'] = dict()
        server_results['test'] = dict()
        server_results['train']['loss'] = []
        server_results['train']['acc'] = []
        server_results['train']['auc'] = []
        server_results['test']['loss'] = []
        server_results['test']['acc'] = []
        server_results['test']['auc'] = []
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.3, patience=10, threshold=1e-4, min_lr=1e-10,
        #     verbose=True)   


        for i in range(args.num_global_epochs):
            # training local models
            for idx in range(num_clients):
                local_models[idx].load_state_dict(global_model_dict)
                local_models[idx], (loss, acc, auc) = train(args, 'source', copy.deepcopy(local_models[idx]), criterions, clients_dls['train'][idx], False, i)
                clients_results['train']['loss'][idx].append(loss)
                clients_results['train']['acc'][idx].append(acc)
                clients_results['train']['auc'][idx].append(auc)
            
            # averaging the weights
            if args.use_sim:
                if i < 5:
                    global_model_dict = average_weights([model.state_dict() for model in local_models], clients_size_frac)
                    global_model.load_state_dict(global_model_dict)
                new_model, _ = train(args, 'target', copy.deepcopy(global_model), criterions, perturb_dl, True, i)
                global_model_dict = update_global(args, [model.state_dict() for model in local_models], global_model.state_dict(), new_model.state_dict(), clients_size, clients_size_frac)
                global_model.load_state_dict(global_model_dict)
            else:
                global_model_dict = average_weights([model.state_dict() for model in local_models], clients_size_frac)
                global_model.load_state_dict(global_model_dict)
                if args.finetune:
                    # for name, param in global_model.named_parameters():
                        # if not 'fc_out' in name:
                        #     param.requires_grad = False
                    global_model, _ = train(args, 'target', global_model, criterions, perturb_dl, True, i)
                    # for name, param in global_model.named_parameters():
                    #     param.requires_grad = True
                    global_model_dict = global_model.state_dict()

            print('testing global model on its target domain')
            (loss, acc, auc) = test(args, global_model, criterion_dignosed, server_dls['test'][0])
            server_results['test']['loss'].append(loss)
            server_results['test']['acc'].append(acc)
            server_results['test']['auc'].append(auc)

        with open(os.path.join(exp_dir,(f'clients_results_{args.iter_idx}.json')), 'w') as fp:
                json.dump(clients_results, fp, indent=4)
        fp.close()
        
        with open(os.path.join(exp_dir,(f'server_results_{args.iter_idx}.json')), 'w') as fp:
                json.dump(server_results, fp, indent=4)
        fp.close()

        # torch.save(global_model.state_dict(),os.path.join(exp_dir,f'server_checkpoint_{args.iter_idx}.pt'))

        # for idx, model in enumerate(local_models):
        #     torch.save(model.state_dict(),os.path.join(exp_dir,f'client_{idx}_checkpoint_{args.iter_idx}.pt'))

        break
        