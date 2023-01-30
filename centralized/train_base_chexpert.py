import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms
from torch import nn
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler
from PIL import Image
from imgaug import augmenters as iaa
import imgaug as ia

from models.resnet import ResNetClassifier, ResNetOrig
from models.cnn import MLTCNN
from utils import constants
from utils.data_sampler import get_subset_indices, get_train_valid_indices
from utils.utils import deterministic
from datasets.CheXpert import ChexpertDataset


def train(args: argparse.Namespace, da_phase: str, model: torch.nn.Module,
          criterion: torch.nn.Module, optimizer: torch.optim,
          scheduler: torch.optim.lr_scheduler, dataloaders: dict, device: str, finetuning):
    """
    Trains classifier module on the provided data set.

    :param args: training arguments
    :type args: argparse.Namespace
    :param model: model to train
    :type model: torch.nn.Module
    :param criterion: criterion used to train model
    :type criterion: torch.nn.Module
    :param optimizer: optimizer used to train model
    :type optimizer: torch.optim
    :param scheduler: learning rate scheduler
    :type scheduler: torch.optim.lr_scheduler
    :param dataloaders: train and valid loaders
    :type dataloaders: dict
    :param device: device to train model on
    :type device: str
    :return: trained nn.Module and training statistics/metrics
    :rtype: Tuple[nn.Module, dict]
    """

    trial_results = dict()
    trial_results['train_loss'] = list()
    trial_results['train_acc'] = list()
    trial_results['valid_loss'] = list()
    trial_results['valid_acc'] = list()
    start_time = time.time()

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.inf
    final_train_loss = 0.0
    final_train_acc = 0.0
    num_epochs = args.num_source_epochs if da_phase == 'source' \
        else args.num_target_epochs
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)
        print('-' * 10, flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0
            y_true_list = list()
            y_pred_list = list()

            # Iterate over dataloader
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    # inputs = sample['img'].to(device)
                    # labels = sample['label'].unsqueeze(1).to(device)
                    inputs = inputs.to(device)
                    labels = labels.unsqueeze(1).to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
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
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Keep track of performance metrics (loss is mean-reduced)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()

                epoch_loss = running_loss / len(y_true_list)
                epoch_acc = float(running_corrects) / len(y_true_list)
                auc = roc_auc_score(y_true_list, y_pred_list)
                trial_results[f'{phase}_loss'].append(epoch_loss)
                trial_results[f'{phase}_acc'].append(epoch_acc)
                trial_results[f'{phase}_auc'] = auc

                # Update LR scheduler with current validation loss
                if phase == 'valid':
                    scheduler.step(epoch_loss)

                # Keep track of current training loss and accuracy
                if phase == 'train':
                    final_train_loss = epoch_loss
                    final_train_acc = epoch_acc

                # Print metrics to stdout
                print('{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, auc), flush=True)

                # Save the best model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                if phase == 'valid' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                elif phase == 'valid' and epoch > 39:
                    epochs_no_improve += 1
                    if epochs_no_improve >= args.patience and args.early_stop:
                        print('\nEarly stopping...\n', flush=True)
                        trial_results['epoch_early_stop'] = epoch + 1
                        break
                else:
                    print(flush=True)
                    continue
                break
        # if epoch % 5 == 0:
        #     if not finetuning:
        #         test(args, model, criterion, source_test_loader, device)
        #     else:
        #         test(args, model, criterion, target_test_loader, device)

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), flush=True)
    print('Best valid acc: {:4f}'.format(best_acc), flush=True)
    trial_results['best_valid_acc'] = best_acc
    print('Best valid loss: {:4f}'.format(best_loss), flush=True)
    trial_results['best_valid_loss'] = best_loss
    print(flush=True)

    # Save final model results
    trial_results['final_train_loss'] = final_train_loss
    trial_results['final_valid_loss'] = epoch_loss
    trial_results['final_train_acc'] = final_train_acc
    trial_results['final_valid_acc'] = epoch_acc

    # Save final model weights
    final_model = copy.deepcopy(model)

    # Load best model weights
    model.load_state_dict(best_model)

    return model, final_model, trial_results

def test(args: argparse.Namespace, model: torch.nn.Module,
         criterion_covid: torch.nn.Module, 
         test_loader: torch.utils.data.DataLoader, device: str):
    """
    Evaluates classifier module on the provided data set.

    :param args: training arguments
    :type args: argparse.Namespace
    :param model: model to train
    :type model: torch.nn.Module
    :param criterion: criterion used to train model
    :type criterion: torch.nn.Module
    :param test_loader: test loader
    :type test_loader: torch.utils.data.DataLoader
    :param device: device to train model on
    :type device: str
    :return: evaluation statistics/metrics
    :rtype: dict
    """

    trial_results = dict()
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    y_true_list = list()
    y_pred_list = list()

    # Iterate over dataloader
    for inputs, labels in test_loader:
        # inputs = sample['img'].to(device)
        # labels = sample['label'].unsqueeze(1).to(device)
        inputs = inputs.to(device)
        labels = labels.unsqueeze(1).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            # outputs = outputs[0]
            labels = labels.type_as(outputs)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5
            loss = criterion_covid(probs, labels)

            for i in range(len(outputs)):
                y_true_list.append(labels[i].cpu().data.tolist())
                y_pred_list.append(probs[i].cpu().data.tolist())

            # Keep track of performance metrics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    test_loss = running_loss / len(y_true_list)
    test_acc = float(running_corrects) / len(y_true_list)
    auc = roc_auc_score(y_true_list, y_pred_list)
    trial_results['test_loss'] = test_loss
    trial_results['test_acc'] = test_acc
    trial_results['test_auc'] = auc

    print('Test Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(
          test_loss, test_acc, auc), flush=True)
    print(flush=True)

    return trial_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_dir', type=str, default='chex2mimic_base')
    parser.add_argument('--iter_idx', type=int, default=0)
    parser.add_argument('--resnet', type=str, default='resnet18')
    parser.add_argument('--load_trained_model', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--num_source_epochs', type=int, default=50)
    parser.add_argument('--num_target_epochs', type=int, default=25)
    parser.add_argument('--source_lr', type=float, default=0.0001)
    parser.add_argument('--target_lr', type=float, default=0.00005)
    parser.add_argument('--source_batch_size', type=int, default=32)
    parser.add_argument('--target_batch_size', type=int, default=32)
    parser.add_argument('--no_drop_last', action='store_false')
    parser.add_argument('--train_seed', type=int, default=8)
    parser.add_argument('--data_sampler_seed', type=int, default=8)
    parser.add_argument('--n_source_samples', type=int, default=8000)
    parser.add_argument('--n_target_samples', type=int, default=1000)
    parser.add_argument('--n_valid_samples', type=int, default=500)
    parser.add_argument('--valid_fraction', type=float, default=0.2)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--data_aug_times', type=int, default=1) # handle overfitting problem
    parser.add_argument('--augment_data', action='store_true') # handle data imbalance problem

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

    if torch.cuda.is_available():
        device_num = torch.cuda.current_device()
        device = f'cuda:{device_num}'
    else:
        device = 'cpu'

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

    # Load PyTorch datasets from directories
    source_train = ChexpertDataset(constants.PNEU_TRAIN_CSV_PATH, args.data_aug_times, 
                                          transform['train'], n_samples=args.n_source_samples)
    source_test = ChexpertDataset(constants.PNEU_TEST_CSV_PATH, 1, 
                                         transform['test'], n_samples=3000)

    # target_train = datasets.ImageFolder(constants.REAL_MIMIC_TRAIN_PATH,
    #                                    transform['train'])
    # target_test = datasets.ImageFolder(constants.REAL_MIMIC_TEST_PATH,
    #                                   transform['test'])
    target_train = ChexpertDataset(constants.EDEMA_TRAIN_CSV_PATH, args.data_aug_times, 
                                          transform['train'], n_samples=args.n_target_samples)
    target_test = ChexpertDataset(constants.EDEMA_TEST_CSV_PATH, args.data_aug_times, 
                                          transform['test'], n_samples=5000)

    # Define classifier, criterion, and optimizer
    model = ResNetClassifier(resnet=args.resnet, hidden_size=args.hidden_size)
    #                            bias=False)
    print(model, flush=True)
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    # criterion_sex = torch.nn.CrossEntropyLoss()
    # criterion_age = torch.nn.CrossEntropyLoss()
    # criterion_angle = torch.nn.CrossEntropyLoss()
    # criterions = [torch.nn.BCEWithLogitsLoss(reduction='mean'), torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()]
    # criterions = [criterion_dignosed, criterion_sex, criterion_age, criterion_angle]
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.source_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=10, threshold=1e-4, min_lr=1e-10,
        verbose=True)

    # ========== SOURCE PHASE ========== #
    # Control deterministic behavior
    deterministic(args.train_seed)

    # Select a stratified subset of the training dataset to use
    # subset_idx = get_subset_indices(source_train, args.n_source_samples,
    #                                 args.data_sampler_seed)
    # subset = torch.utils.data.Subset(source_train, subset_idx)

    subset = source_train

    # Split into train and validation sets and create PyTorch Dataloaders
    train_dataset, valid_dataset = torch.utils.data.random_split(
        subset,
        [int(np.floor((1 - args.valid_fraction) * len(subset))), int(np.ceil(args.valid_fraction * len(subset)))],
        generator=torch.Generator().manual_seed(args.data_sampler_seed))

    # print(type(train_dataset), type(valid_dataset))
    
    if args.augment_data:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.source_batch_size, shuffle=True,
            sampler=ImbalancedDatasetSampler(train_dataset),
            drop_last=args.no_drop_last)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.source_batch_size, shuffle=True, 
            sampler=ImbalancedDatasetSampler(train_dataset))
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.source_batch_size, shuffle=True,
            drop_last=args.no_drop_last)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.source_batch_size, shuffle=True)
    target_test_loader = torch.utils.data.DataLoader(
        target_test, batch_size=args.target_batch_size, shuffle=False)
    source_test_loader = torch.utils.data.DataLoader(
        source_test, batch_size=args.source_batch_size, shuffle=False)
    dataloaders = {'train': train_loader, 'valid': valid_loader}

    if args.load_trained_model:  # Load existing trained model
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:  # Train on source data
        model, _, source_results = train(
            args, 'source', model, criterion, optimizer, lr_scheduler,
            dataloaders, device, False)
        target_test_results = test(
            args, model, criterion, target_test_loader, device)
        # source_test_results = test(
        #     args, model, criterion, source_test_loader, device)
        source_results.update(
            {f'target_{k}': v for k, v in target_test_results.items()})
        # source_results.update(
        #     {f'source_{k}': v
        #         for k, v in source_test_results.items()})
        source_results_df = pd.DataFrame(columns=list(source_results.keys()))
        source_results_df = source_results_df.append(source_results,
                                                     ignore_index=True)
        source_results_df.to_parquet(
            os.path.join(exp_dir, f'source_results_{args.iter_idx}.parquet'),
            index=False)
        with open(os.path.join(exp_dir,f'source_results_{args.iter_idx}.json'), 'w') as fp:
            json.dump(source_results, fp, indent=4)
        fp.close()
        torch.save(model.state_dict(),
                   os.path.join(exp_dir,
                                f'source_checkpoint_{args.iter_idx}.pt'))

    # ========== TARGET PHASE ========== #
    # Control deterministic behavior
    deterministic(args.train_seed)

    # Freeze parameters or adjust layers as specified
    if args.freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        print('Fine-tuning the following parameters...', flush=True)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'\t{name}', flush=True)
        print(flush=True)

    # Re-define optimizer and lr_scheduler to update parameters optimized
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.target_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=10, threshold=1e-4, min_lr=1e-10,
        verbose=True)

    # Select a stratified subset of the training dataset to use
    if args.valid_fraction is not None:
        # subset_idx = get_subset_indices(target_train, args.n_target_samples,
        #                                 args.data_sampler_seed)
        # subset = torch.utils.data.Subset(target_train, subset_idx)
        subset = target_train
        # Split into train and validation sets and create PyTorch Dataloaders
        train_dataset, valid_dataset = torch.utils.data.random_split(
            subset,
            [int(np.ceil((1 - args.valid_fraction) * len(subset))),
             int(np.floor(args.valid_fraction * len(subset)))],
            generator=torch.Generator().manual_seed(args.train_seed))
    # Select stratified subsets to use for training (variable over iterations)
    # and validation (fixed over iterations)
    else:
        train_idx, valid_idx = get_train_valid_indices(
            target_train, args.n_target_samples, args.n_valid_samples,
            args.train_seed)
        train_dataset = torch.utils.data.Subset(target_train, train_idx)
        valid_dataset = torch.utils.data.Subset(target_train, valid_idx)
        # assert len(train_dataset) == args.n_target_samples
        # assert len(valid_dataset) == args.n_valid_samples

    # Create PyTorch Dataloaders
    if args.augment_data:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.target_batch_size, shuffle=True,
            sampler=ImbalancedDatasetSampler(train_dataset),
            drop_last=args.no_drop_last)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.target_batch_size, shuffle=True,
            sampler=ImbalancedDatasetSampler(train_dataset))
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.target_batch_size, shuffle=True,
            drop_last=args.no_drop_last)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.target_batch_size, shuffle=True)
    target_test_loader = torch.utils.data.DataLoader(
        target_test, batch_size=args.target_batch_size, shuffle=False)
    # source_test_loader = torch.utils.data.DataLoader(
    #     source_test, batch_size=args.target_batch_size, shuffle=False)
    dataloaders = {'train': train_loader, 'valid': valid_loader}

    # print(len(train_dataset), len(valid_dataset), len(target_test), len(source_test))

    # Train on target data
    model, final_model, target_results = train(
        args, 'target', model, criterion, optimizer, lr_scheduler, dataloaders,
        device, True)
    target_test_results = test(
        args, model, criterion, target_test_loader, device)
    # source_test_results = test(
    #     args, model, criterion, source_test_loader, device)
    target_results.update(
        {f'target_best_{k}': v for k, v in target_test_results.items()})
    # target_results.update(
    #     {f'source_best_{k}': v for k, v in source_test_results.items()})
    target_test_results = test(
        args, final_model, criterion, target_test_loader, device)
    # source_test_results = test(
    #     args, final_model, criterion, source_test_loader, device)
    target_results.update(
        {f'target_final_{k}': v for k, v in target_test_results.items()})
    # target_results.update(
    #     {f'source_final_{k}': v for k, v in source_test_results.items()})
    target_results_df = pd.DataFrame(columns=list(target_results.keys()))
    target_results_df = target_results_df.append(target_results,
                                                 ignore_index=True)
    target_results_df.to_parquet(
        os.path.join(exp_dir, f'target_results_{args.iter_idx}.parquet'),
        index=False)
    with open(os.path.join(exp_dir,f'target_results_{args.iter_idx}.json'), 'w') as fp:
            json.dump(target_results, fp, indent=4)
    fp.close()
    torch.save(model.state_dict(),
               os.path.join(exp_dir, f'target_checkpoint_{args.iter_idx}.pt'))
    torch.save(final_model.state_dict(),
               os.path.join(exp_dir,
                            f'target_checkpoint_final_{args.iter_idx}.pt'))