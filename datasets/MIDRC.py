import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import PIL.Image as Image
from imgaug import augmenters as iaa
import matplotlib.pylab as plt
import csv
import time
import os
from random import shuffle

# race2idx = {'White':0, 'Black or African American':1, 'Asian':2, 'Other':3}
race2idx = {'White':0, 'Black or African American':1, 'Other':2}
sex2idx = {'Male':0, 'Female':1}
states2idx = {'IL':0, 'NC':1, 'CA':2, 'IN':3, 'TX':4}

class MidrcDataset(Dataset):
    """Midrc dataset."""

    def __init__(self, csv_path, augment_times=1, transform=None, n_samples=None):
        """
        Args:        
            csv_path (string): The csv which contains the info we needed.
            augment_times (int): how many times we want to augment the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imgs = []
        self.labels = []
        self.transform = transform
        with open(csv_path, 'r') as csv_file:
            rows = csv_file.read().split('\n')[1:]
            count = 0
            shuffle(rows) 
            for row in rows:
                if row:
                    if n_samples and count >= n_samples:
                        break
                    if count % 1000 == 0:
                        print(count)
                    row = row.split(',')
                    is_covid = int(row[2] == 'Yes')
                    img_origin = Image.open(row[0]).convert('RGB')
                    count += 1
                    # if sex == 'Male':
                    # if race == 'Black or African American':
                    # if race == 'White':
                    for _ in range(augment_times):
                        if self.transform:
                            img = self.transform(img_origin)
                        self.labels.append(is_covid)
                        self.imgs.append(img)
                        
        print(len(self.labels))
        csv_file.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

class MidrcMLTDataset(Dataset):
    """Midrc Multi-task learning with demographic info dataset."""

    def __init__(self, csv_path, augment_times=1, transform=None, n_samples=None):
        """
        Args:        
            csv_path (string): The csv which contains the info we needed.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imgs = []
        self.labels = []
        self.races = []
        self.genders = []
        self.ages = []
        self.states = []
        self.transform = transform
        with open(csv_path, 'r') as csv_file:
            rows = csv_file.read().split('\n')[1:]
            count = 0
            # shuffle(rows)
            for row in rows:
                if row:
                    if n_samples and count >= n_samples:
                        break
                    if count % 1000 == 0:
                        print(count)
                    img_path, race, is_covid, sex, age, state = row.split(',')
                    img_origin = Image.open(img_path).convert('RGB')
                    race = race2idx[race] if race in race2idx else race2idx['Other']
                    count += 1
                    for _ in range(augment_times):
                        if self.transform:
                            img = self.transform(img_origin)
                        self.labels.append(int(is_covid == 'Yes'))
                        self.imgs.append(img)
                        self.races.append(race)
                        self.genders.append(sex2idx[sex])
                        self.ages.append(int(age)//10)
                        self.states.append(states2idx[state])
        csv_file.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):   
        return {'img': self.imgs[idx], 'targets': (self.labels[idx], self.races[idx], self.genders[idx], self.ages[idx])}
    
    def get_labels(self):
        return self.races

# class MidrcMLTDataset(Dataset):
#     """Midrc Multi-task learning dataset."""

#     def __init__(self, csv_path, transform=None):
#         """
#         Args:        
#             csv_path (string): The csv which contains the info we needed.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.img_paths = []
#         self.transform = transform
#         with open(csv_path, 'r') as csv_file:
#             rows = csv_file.read().split('\n')
#             self.label_names = rows[0].split(',')[1:]
#             self.num_labels = len(self.label_names)
#             self.labels = [[]] * self.num_labels
#             for row in rows[1:]:               
#                 if row:
#                     eles = row.split(',')
#                     for i in len(eles[1:]):
#                         self.labels[i].extend(eles[i+1])
#                     self.img_paths.append(eles[0])
#         csv_file.close()
#         for i in range(len(self.labels)):
#             self.labels[i] = str2label(self.labels[i])
        

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         img = Image.open(self.img_paths[idx])
       
#         if self.transform:
#             img = self.transform(img)

#         cur_labels = [self.labels[i][idx] for i in range(self.num_labels)]        

#         return img, cur_labels

# def str2label(labels):
#     kinds = sorted(list(set(labels)))
#     l2idx = dict(zip(kinds, [i for i in range(len(kinds))]))
#     return [l2idx[l] for l in labels]

# class MidrcDataAugRaceDataset(Dataset):
#     """Midrc dataset with augmentation on races."""

#     def __init__(self, csv_path, augment_times=1, transform=None):
#         """
#         Args:        
#             csv_path (string): The csv which contains the info we needed.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.imgs = []
#         paths = {}
#         self.labels = []
#         self.transform = transform
#         with open(csv_path, 'r') as csv_file:
#             rows = csv_file.read().split('\n')
#             for row in rows[1:]:
#                 if row:
#                     img_path, race, is_covid = row.split(',')
#                     # img = Image.open(img_path).convert('RGB')
#                     race = race2idx[race] if race in race2idx else 3
#                     # if self.transform:
#                     #     img = self.transform(img)
#                     is_covid = int(is_covid == 'Yes')
#                     if race not in paths:
#                         paths[race] = []
#                     paths[race].append((img_path, is_covid)) # get the path/label pair
    
#         # data augmentation
#         counts = [len(paths[i]) for i in range(4)]
#         max_count = max(counts)
#         for idx in range(4):
#             if len(path[i]) < max_count:
#                 cur_num = len(path[i])
#                 while cur_num < max_count:
#                     pass
#             else:
#                 for pair in path[i]:
#                     img = Image.open(img_path).convert('RGB')
#         csv_file.close()

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return {'img': self.imgs[idx], 'label': self.labels[idx]}

if __name__ == '__main__':
    csv_file = 'meta_data_info/MIDRC_cr_table_race_test.csv'
    transform = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,)),
			]
		)
    test = MidrcDataset(csv_file, transform)
    print(test.labels)
    for i in range (423, len(test), 100): 
        print(test[i]['img'].shape)
        print(test[i]['img'].max(), test[i]['img'].min())
        plt.imshow(test[i]['img'][0,:,:])
        plt.savefig('meta_data_info/test_img.png')
        break

    loader = torch.utils.data.DataLoader(test, batch_size=16)