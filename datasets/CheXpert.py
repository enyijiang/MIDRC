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

sex2idx = {'Male':0, 'Female':1}
dir2idx = {'Frontal':0, 'Lateral':1}
# states2idx = {'IL':0, 'NC':1, 'CA':2, 'IN':3, 'TX':4}

class ChexpertDataset(Dataset):
    """Chexpert dataset."""

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
            # shuffle(rows) 
            for row in rows:
                if row:
                    if n_samples and count >= n_samples:
                        break
                    if count % 1000 == 0:
                        print(count)
                    
                    row = row.split(',')
                    if os.path.exists(row[0]):
                        is_dignosed = int(row[-1] == 'Yes')
                        img_origin = Image.open(row[0]).convert('RGB')
                        count += 1
                        for _ in range(augment_times):
                            if self.transform:
                                img = self.transform(img_origin)
                            self.labels.append(is_dignosed)
                            self.imgs.append(img)
                        
        print(len(self.labels))
        csv_file.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

class ChexpertMLTDataset(Dataset):
    """Chexpert Multi-task learning with demographic info dataset."""

    def __init__(self, csv_path, augment_times=1, transform=None, n_samples=None):
        """
        Args:        
            csv_path (string): The csv which contains the info we needed.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imgs = []
        self.labels = []
        self.genders = []
        self.ages = []
        self.angles = []
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
                    img_path, sex, age, angle, is_dignosed = row.split(',')
                    if os.path.exists(img_path):
                        img_origin = Image.open(img_path).convert('RGB')
                        count += 1
                        for _ in range(augment_times):
                            if self.transform:
                                img = self.transform(img_origin)
                            self.labels.append(int(is_dignosed == 'Yes'))
                            self.imgs.append(img)
                            self.genders.append(sex2idx[sex])
                            self.ages.append((int(age)-18)//9)
                            self.angles.append(dir2idx[angle])
        csv_file.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):   
        return self.imgs[idx], (self.labels[idx], self.genders[idx], self.ages[idx], self.angles[idx])
    
    def get_labels(self):
        return self.races

# class ChexpertDataAugRaceDataset(Dataset):
#     """Chexpert dataset with augmentation on races."""

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
    csv_file = '../meta_data_info/chexpert_table_test.csv'
    transform = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,)),
			]
		)
    test = ChexpertMLTDataset(csv_file, transform = transform, n_samples=100)
    print(test.labels)
    for i in range (len(test)): 
        print(test[i]['img'].shape)
        print(test[i]['img'].max(), test[i]['img'].min())
        plt.imshow(test[i]['img'][0,:,:])
        plt.savefig('../meta_data_info/chexpert_test_img.png')
        break

    loader = torch.utils.data.DataLoader(test, batch_size=16)