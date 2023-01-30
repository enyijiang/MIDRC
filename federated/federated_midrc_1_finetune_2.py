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
from tqdm import tqdm
from imgaug import augmenters as iaa
import imgaug as ia
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler

from models.resnet import ResNetClassifier, ResNetOrig
from models.cnn import CNN
from utils import constants
from utils.data_sampler import get_subset_indices, get_train_valid_indices
from utils.utils import deterministic
from datasets.MIDRC import MidrcDataset
from torch.multiprocessing import Pool

def set_device():
    global device
    device = torch.device(0 if torch.cuda.is_available() else "cpu")

def show_image(dataset, index):
    plt.gray()
    plt.imshow(dataset.__getitem__(index)[0].reshape((28,28)))
    plt.show()

def load_data(args, states, transform):
    dls = {'train':[], 'test':[]}
    for mode in ['train', 'test']:
        for state in states:
            # if mode == 'test':
            #     train_dataset = MidrcDataset(os.path.join(constants.STATE_DATA_DIR, f'MIDRC_table_{state}_{mode}.csv'), args.data_aug_times, transform[mode])
            #     # dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.source_batch_size, shuffle=False, drop_last=args.no_drop_last)
            # else:
            train_dataset = MidrcDataset(os.path.join(constants.STATE_DATA_DIR, f'MIDRC_table_{state}_{mode}.csv'), args.data_aug_times, transform[mode])
                # dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.source_batch_size, shuffle=True, drop_last=args.no_drop_last)
            dls[mode].append(train_dataset)
    return dls

def train(args, da_phase, model, criterion: torch.nn.Module, train_dl):
    global device

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.source_lr if da_phase=='source' else args.target_lr)
    # trial_results = dict()
    # trial_results['train_loss'] = list()
    # trial_results['train_acc'] = list()
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
            # trial_results['train_loss'] = epoch_loss
            # trial_results['train_acc'] = epoch_acc
            # trial_results['train_auc'] = auc

            # Update LR scheduler with current validation loss
            # if phase == 'valid':
            #     scheduler.step(epoch_loss)

    # Keep track of current training loss and accuracy
    final_train_loss = epoch_loss
    final_train_acc = epoch_acc
    final_train_auc = auc

            # move inputs to device
            # im, lbl = im.to(device), lbl.to(device)

            # # zero the parameter gradients
            # optimizer.zero_grad()

            # # forward -> backward -> optimize
            # outputs = model(im)
            # loss = criterion(outputs, lbl)
            # loss.backward()
            # optimizer.step()

            # # compute accuracy
            # _, predicted = torch.max(outputs.data, 1)
            # correct = (predicted == lbl).sum().item()
            # accuracy = correct / lbl.shape[0]

            # tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
            # time.sleep(0.1)
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
    # trial_results['test_loss'] = test_loss
    # trial_results['test_acc'] = test_acc
    # trial_results['test_auc'] = auc

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
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def update_dict(old_model_dict, new_model_dict, alpha):
    new_w = copy.deepcopy(old_model_dict)
    for key in new_w.keys():
        new_w[key] = torch.zeros_like(new_w[key]).float()
        new_w[key] = old_model_dict[key] * alpha + new_model_dict[key] * (1-alpha)
    return new_w

# update global model using the consine projection on the target converge direction
# def update_global(args, local_models_dict, old_global_model_dict, finetune_global_model_dict, cos_sim, clients_size, clients_size_frac):
#     ret_dict = copy.deepcopy(old_global_model_dict)
#     b = 0.5
#     cos = torch.nn.CosineSimilarity()
#     for key in ret_dict.keys():
#         if ret_dict[key].shape != torch.Size([]):
#             global_grad = finetune_global_model_dict[key] - old_global_model_dict[key]
#             for idx, local_dict in enumerate(local_models_dict):
#                 local_grad = local_dict[key] - old_global_model_dict[key]
#                 cur_sim = cos(global_grad.reshape(1,-1), local_grad.reshape(1,-1))
#                 # print(global_grad.shape, local_grad.shape)
#                 # print(cos(global_grad.reshape(1,-1), local_grad.reshape(1,-1)))
#                 # ret_dict[key] = ret_dict[key] + b * clients_size_frac[idx] * cos_sim[idx] * local_grad
#                 if cur_sim > 0:
#                     ret_dict[key] = ret_dict[key] + b * (clients_size[idx] / args.n_target_samples) * clients_size_frac[idx] * cur_sim * local_grad
#             ret_dict[key] = ret_dict[key] + (1-b) * (global_grad)
#         else:
#             # ret_dict[key] = torch.zeros_like(old_global_model_dict[key]).float()
#             # for idx, local_dict in enumerate(local_models_dict):
#             #     ret_dict[key] += clients_size_frac[idx] * local_dict[key]
#             ret_dict[key] = old_global_model_dict[key]
#     return ret_dict

# update global model using the consine projection on the target converge direction
def update_global(args, local_models_dict, old_global_model_dict, finetune_global_model_dict, cos_sim, clients_size, clients_size_frac):
    ret_dict = copy.deepcopy(old_global_model_dict)
    b = 0.5
    cos = torch.nn.CosineSimilarity()
    for key in ret_dict.keys():
        if ret_dict[key].shape != torch.Size([]):
            global_grad = finetune_global_model_dict[key] - old_global_model_dict[key]
            for idx, local_dict in enumerate(local_models_dict):
                local_grad = local_dict[key] - old_global_model_dict[key]
                cur_sim = cos(global_grad.reshape(1,-1), local_grad.reshape(1,-1))
                # print(global_grad.shape, local_grad.shape)
                # print(cos(global_grad.reshape(1,-1), local_grad.reshape(1,-1)))
                # ret_dict[key] = ret_dict[key] + b * clients_size_frac[idx] * cos_sim[idx] * local_grad
                if cur_sim > 0:
                    ret_dict[key] = ret_dict[key] + b * (args.target_lr / args.source_lr) * ((args.n_target_samples/args.target_batch_size)/(clients_size[idx]/args.source_batch_size)) * clients_size_frac[idx] * cur_sim * local_grad
                    # ret_dict[key] = ret_dict[key] + b * (clients_size[idx] / args.n_target_samples) * clients_size_frac[idx] * cur_sim * local_grad
            ret_dict[key] = ret_dict[key] + (1-b) * (global_grad)
        else:
            # ret_dict[key] = torch.zeros_like(old_global_model_dict[key]).float()
            # for idx, local_dict in enumerate(local_models_dict):
            #     ret_dict[key] += clients_size_frac[idx] * local_dict[key]
            ret_dict[key] = old_global_model_dict[key]
    return ret_dict

# get the grad updates
def get_model_updates(init_model, new_model):
    ret_updates = []
    init = get_param_list(init_model)
    new = get_param_list(new_model)
    # init_dict = init_model.state_dict()
    # new_dict = new_model.state_dict()
    # for key in init_dict.keys():
    #     # print(torch.subtract(new_dict[key], init_dict[key]).cpu().tolist())
    #     ret_updates.extend(torch.flatten(torch.subtract(new_dict[key], init_dict[key]).cpu()).tolist())
    # print(init.shape, new.shape)
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

    parser.add_argument('--exp_dir', type=str, default='fl_cosim_comp_2')
    parser.add_argument('--iter_idx', type=int, default=0)
    parser.add_argument('--resnet', type=str, default='resnet18')
    parser.add_argument('--load_trained_model', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--num_source_epochs', type=int, default=1)
    parser.add_argument('--num_target_epochs', type=int, default=1)
    parser.add_argument('--num_global_epochs', type=int, default=50)
    parser.add_argument('--source_lr', type=float, default=0.0001)
    parser.add_argument('--target_lr', type=float, default=0.00005)
    parser.add_argument('--source_batch_size', type=int, default=8)
    parser.add_argument('--target_batch_size', type=int, default=16)
    parser.add_argument('--no_drop_last', action='store_false')
    parser.add_argument('--train_seed', type=int, default=8)
    parser.add_argument('--data_sampler_seed', type=int, default=8)
    parser.add_argument('--n_source_samples', type=int, default=500)
    parser.add_argument('--n_target_samples', type=int, default=200)
    parser.add_argument('--n_valid_samples', type=int, default=500)
    parser.add_argument('--valid_fraction', type=float, default=None)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--data_aug_times', type=int, default=1)
    parser.add_argument('--use_sim', action='store_true')
    parser.add_argument('--finetune', action='store_true')

    args = parser.parse_args()
    timestamp = time.strftime("%Y-%m-%d-%H%M")

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

    # ['IL', 'NC', 'CA', 'IN', 'TX']
    states = ['IL', 'NC', 'CA', 'IN', 'TX']
    # states = ['NC', 'IL']
    # states = ['IL', 'CA', 'IN', 'TX']
    ds = load_data(args, states, transform)
    
    for idx, server in enumerate(states):
        clients = copy.deepcopy(states)
        clients.remove(server)
        print(clients, server)
        clients_dls = {'train':[], 'test':[]}
        server_dls = {'train':[], 'test':[]}
        for mode in ['train', 'test']:
            if mode == 'test':
                clients_dls[mode] = [torch.utils.data.DataLoader(ds[mode][i], batch_size=args.source_batch_size, shuffle=False, drop_last=args.no_drop_last) for i in range(len(states)) if i != idx]
                server_dls[mode] = [torch.utils.data.DataLoader(ds[mode][i], batch_size=args.target_batch_size, shuffle=False, drop_last=args.no_drop_last) for i in range(len(states)) if i == idx]    
            else:
                clients_dls[mode] = [torch.utils.data.DataLoader(ds[mode][i], batch_size=args.source_batch_size, shuffle=True, drop_last=args.no_drop_last) for i in range(len(states)) if i != idx]
                server_dls[mode] = [torch.utils.data.DataLoader(ds[mode][i], batch_size=args.target_batch_size, shuffle=True, drop_last=args.no_drop_last) for i in range(len(states)) if i == idx]    
        
        target_train = ds['train'][idx]
        indices = torch.randperm(len(target_train))[:args.n_target_samples]
        cur_sampler = SubsetRandomSampler(indices)
        perturb_dl = torch.utils.data.DataLoader(target_train, shuffle=False, batch_size=args.target_batch_size, sampler=cur_sampler)
        # perturb_dl = torch.utils.data.DataLoader(target_train, shuffle=False, batch_size=args.target_batch_size)

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
        
        global_epochs = [i*5 for i in range(1, 11)]
        # initialize datalaoders, models, optimizer, criterions

        for global_epoch in global_epochs:
            print(global_epoch)
            num_clients = len(clients)
            dict_client = dict()
            for i in range(num_clients):
                dict_client.update({clients[i]: []})

            clients_size = [len(clients_dls[mode][i])*args.source_batch_size for i in range(num_clients)]
            clients_size_frac = np.array(clients_size) / sum(clients_size)
            print(clients_size_frac)

            # print(clients_dls, server_dls)

            global_model = ResNetClassifier(resnet=args.resnet, hidden_size=args.hidden_size)
            global_model.to(device)
            local_models = [ResNetClassifier(resnet=args.resnet, hidden_size=args.hidden_size) for _ in range(num_clients)]
            clients_grads = [None] * num_clients
            # server_grads = [None] * num_clients
            cos_sim = [None] * num_clients
            global_model_dict = global_model.state_dict()

            criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
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
            # server_results['train'] = dict()
            server_results['test'] = dict()
            # server_results['train']['loss'] = []
            # server_results['train']['acc'] = []
            # server_results['train']['auc'] = []
            server_results['test']['loss'] = []
            server_results['test']['acc'] = []
            server_results['test']['auc'] = []
            # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     optimizer, factor=0.3, patience=10, threshold=1e-4, min_lr=1e-10,
            #     verbose=True)   


            for i in range(global_epoch):
                # training local models
                for idx in range(num_clients):
                    # if i == 0:
                    local_models[idx].load_state_dict(global_model_dict)
                    # else:
                        # new_local_model_dict = update_dict(global_model.state_dict(), local_models[idx].state_dict(), weights[idx])
                        # local_models[idx].load_state_dict(new_local_model_dict)
                    # if i < args.num_global_epochs // 2:
                    local_models[idx], (loss, acc, auc) = train(args, 'source', copy.deepcopy(local_models[idx]), criterion, clients_dls['train'][idx])
                    # clients_grads[idx] = get_model_updates(local_models[idx].to('cpu'), new_model.to('cpu'))
                    # local_models[idx].load_state_dict(new_model.state_dict())
                    clients_results['train']['loss'][clients[idx]].append(loss)
                    clients_results['train']['acc'][clients[idx]].append(acc)
                    clients_results['train']['auc'][clients[idx]].append(auc)
                # node_1.load_state_dict(global_model_dict)
                # node_2.load_state_dict(global_model_dict)
                # node_3.load_state_dict(global_model_dict)
                
                # small purterbation on the target set
                # new_model, _ = train(args, 'target', copy.deepcopy(global_model), criterion, perturb_dl)
                # server_grad = get_model_updates(global_model.to('cpu'), new_model.to('cpu'))
                # # set up the purtabation set
                # for idx in range(num_clients):
                # # for idx in range(num_clients):
                #     cos_sim[idx] = cosine_similarity(server_grad, clients_grads[idx])[0][0]
                
                # print(cos_sim)
                # weights = np.exp(cos_sim)/sum(np.exp(cos_sim))
                # print(weights)



                
                # node_1 = train(node_1, node_1_train, num_local_epochs)
                # node_2 = train(node_2, node_2_train, num_local_epochs)
                # node_3 = train(node_3, node_3_train, num_local_epochs)
                global_model_dict = average_weights([model.state_dict() for model in local_models], clients_size_frac)
                global_model.load_state_dict(global_model_dict)

                print('testing each local model\'s accuracy on their own source domain')
                for idx in range(num_clients): 
                    (loss, acc, auc) = test(args, local_models[idx], criterion, clients_dls['test'][idx])
                    clients_results['test_s']['loss'][clients[idx]].append(loss)
                    clients_results['test_s']['acc'][clients[idx]].append(acc)
                    clients_results['test_s']['auc'][clients[idx]].append(auc)
            
            for i in range(global_epoch):
                # averaging the weights
                if args.use_sim:
                    new_model, _ = train(args, 'target', copy.deepcopy(global_model), criterion, perturb_dl)
                    # if i == 0:
                    #     global_model_dict = average_weights([model.state_dict() for model in local_models], weights)
                    # else:
                    #     global_model_dict = update_dict(global_model_dict, average_weights([model.state_dict() for model in local_models], weights), 0.5)
                    global_model_dict = update_global(args, [model.state_dict() for model in local_models], global_model.state_dict(), new_model.state_dict(), cos_sim, clients_size, clients_size_frac)
                    global_model.load_state_dict(global_model_dict)
                else:
                    # print('eorigut')
                    
                    if args.finetune:
                        # Freeze all but last layer
                        # for name, param in global_model.named_parameters():
                            # if not 'linear' in name:
                            #     param.requires_grad = False
                        global_model, _ = train(args, 'target', global_model, criterion, perturb_dl)
                        # unfreeze all
                        # for name, param in global_model.named_parameters():
                        #     param.requires_grad = True
                        global_model_dict = global_model.state_dict()
                # global_model, (loss, acc, auc) = train(args, 'target', global_model, criterion, server_dls['train'][0])
                # print(loss, acc, auc)
                # server_results['train']['loss'].append(loss)
                # server_results['train']['acc'].append(acc)
                # server_results['train']['auc'].append(auc)
                # finetuning the global model

                # print('testing each local model\'s accuracy on target domain')
                # for idx in range(num_clients): 
                #     (loss, acc, auc) = test(args, local_models[idx], criterion, server_dls['test'][0])
                #     clients_results['test_t']['loss'][clients[idx]].append(loss)
                #     clients_results['test_t']['acc'][clients[idx]].append(acc)
                #     clients_results['test_t']['auc'][clients[idx]].append(auc)

                

                # global_model.load_state_dict(global_model_dict)
                # global_model.to(device)
                # global_model.eval()

                print('testing global model on its target domain')
                (loss, acc, auc) = test(args, global_model, criterion, server_dls['test'][0])
                server_results['test']['loss'].append(loss)
                server_results['test']['acc'].append(acc)
                server_results['test']['auc'].append(auc)

            with open(os.path.join(exp_dir,(f'clients_results_{args.iter_idx}_{global_epoch}.json')), 'w') as fp:
                    json.dump(clients_results, fp, indent=4)
            fp.close()
            
            with open(os.path.join(exp_dir,(f'server_results_{args.iter_idx}_{global_epoch}.json')), 'w') as fp:
                    json.dump(server_results, fp, indent=4)
            fp.close()

            # torch.save(global_model.state_dict(),os.path.join(exp_dir,f'server_checkpoint_{args.iter_idx}.pt'))

            # for idx, model in enumerate(local_models):
            #     torch.save(model.state_dict(),os.path.join(exp_dir,f'client_{idx}_checkpoint_{args.iter_idx}.pt'))

        break
        