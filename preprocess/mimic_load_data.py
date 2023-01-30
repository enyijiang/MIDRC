"""MIMIC-CXR dataloader."""

import os

import pandas as pd
import numpy as np
import datetime as dt

import torch
import torchvision

from PIL import Image

def load_data():
  """Load data from csv across different linked mimic versions"""

  metadata_path = "/shared/rsaas/oes2/mimic-cxr/physionet.org/" + \
    "mimic-cxr-2.0.0-metadata.csv"
  chexpert_path = "/shared/rsaas/oes2/mimic-cxr/physionet.org/" + \
    "mimic-cxr-2.0.0-chexpert.csv"
  icd9_path = "/shared/rsaas/oes2/mimic-cxr/physionet.org/" + \
    "diagnoses_icd.csv"
  patients_path = "/shared/rsaas/oes2/mimic-cxr/physionet.org/" + \
    "patients.csv"
  admissions_path = "/shared/rsaas/oes2/mimic-cxr/physionet.org/" + \
    "admissions.csv"

  metadata = pd.read_csv(metadata_path)
  # print("metadata size: ", metadata.shape)
  chexpert = pd.read_csv(chexpert_path).fillna(0.)
  # print("chexpert size: ", chexpert.shape)
  admissions = pd.read_csv(admissions_path).loc[:, ['subject_id', 'admission_type',
  'language', 'marital_status', 'race']].drop_duplicates('subject_id')
  # print("admissions size: ", admissions.shape)
  patients = pd.read_csv(patients_path).loc[:, ['subject_id',
  'gender', 'anchor_age']].drop_duplicates('subject_id')

  # print("patients size: ", patients.shape)

  CONDITIONS = list(chexpert.columns)[2:]
  # print("CONDITIONS: ", CONDITIONS)

  # print("merging with chexpert...")
  combined_data = metadata.merge(chexpert, how='inner',
  on=['subject_id', 'study_id'])
  # print("combined_data size: ", combined_data.shape)

  # print("merging with admissions...")
  combined_data = combined_data.merge(admissions, how='inner',
  on='subject_id')
  # print("combined_data size: ", combined_data.shape)

  # print("merging with patients...")
  combined_data = combined_data.merge(patients, how='inner',
    on='subject_id')
  # print("combined_data size: ", combined_data.shape)

  for condition in CONDITIONS:
    combined_data = combined_data[combined_data[condition] != -1]

  # Get days till admissions
  combined_data.admittime = combined_data.admittime.apply(lambda x:  \
    dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
  combined_data.dischtime= combined_data.dischtime.apply(lambda x:  \
    dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

  combined_data['len_stay'] = ((combined_data.dischtime -
  combined_data.admittime).dt.days <= 3).astype(int)
  combined_data['white'] = combined_data.race.str.contains('white',
    case=False).astype(int)

  combined_data = combined_data.loc[:, ['subject_id', 'study_id', 'len_stay',
    'white', 'race']]

  # Make race one-hot
  combined_data = pd.concat([combined_data,
    pd.get_dummies(combined_data.race)], 1).drop(['race'], 1)
  # print(combined_data.columns)
  # print("Final size: {}".format(combined_data.shape))

  return combined_data

def load_data_fl():
  """Load data from csv across different linked mimic versions"""

  metadata_path = "/shared/rsaas/oes2/mimic-cxr/physionet.org/" + \
    "mimic-cxr-2.0.0-metadata.csv"
  chexpert_path = "/shared/rsaas/oes2/mimic-cxr/physionet.org/" + \
    "mimic-cxr-2.0.0-chexpert.csv"
  # icd9_path = "/shared/rsaas/oes2/mimic-cxr/physionet.org/" + \
  #   "diagnoses_icd.csv"
  patients_path = "/shared/rsaas/oes2/mimic-cxr/physionet.org/" + \
    "patients.csv"
  admissions_path = "/shared/rsaas/oes2/mimic-cxr/physionet.org/" + \
    "admissions.csv"

  metadata = pd.read_csv(metadata_path).loc[:, ['dicom_id', 'subject_id', 'study_id']]
  print("metadata size: ", metadata.shape)
  chexpert = pd.read_csv(chexpert_path).fillna(0.)
  print("chexpert size: ", chexpert.shape)
  admissions = pd.read_csv(admissions_path).loc[:, ['subject_id', 'admission_type',
  'language', 'race']].drop_duplicates('subject_id')
  print("admissions size: ", admissions.shape)
  patients = pd.read_csv(patients_path).loc[:, ['subject_id',
  'gender', 'anchor_age']].drop_duplicates('subject_id')

  print("patients size: ", patients.shape)

  CONDITIONS = list(chexpert.columns)[2:]
  print("CONDITIONS: ", CONDITIONS)

  print("merging with chexpert...")
  combined_data = metadata.merge(chexpert, how='inner',
  on=['subject_id', 'study_id'])
  print("combined_data size: ", combined_data.shape)

  print("merging with admissions...")
  combined_data = combined_data.merge(admissions, how='inner',
  on='subject_id')
  print("combined_data size: ", combined_data.shape)

  print("merging with patients...")
  combined_data = combined_data.merge(patients, how='inner',
    on='subject_id')
  print("combined_data size: ", combined_data.shape)

  for condition in CONDITIONS:
    combined_data = combined_data[combined_data[condition] != -1]

  # print(combined_data)
  # weoitu
  # # Get days till admissions
  # combined_data.admittime = combined_data.admittime.apply(lambda x:  \
  #   dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
  # combined_data.dischtime= combined_data.dischtime.apply(lambda x:  \
  #   dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

  # combined_data['len_stay'] = ((combined_data.dischtime -
  # combined_data.admittime).dt.days <= 3).astype(int)
  # combined_data['white'] = combined_data.race.str.contains('white',
  #   case=False).astype(int)

  # combined_data = combined_data.loc[:, ['subject_id', 'study_id', 'len_stay',
  #   'white', 'race']]

  # # Make race one-hot
  # combined_data = pd.concat([combined_data,
  #   pd.get_dummies(combined_data.race)], 1).drop(['race'], 1)
  # print(combined_data.columns)
  # print("Final size: {}".format(combined_data.shape))
  combined_data.loc[combined_data['race'].str.contains('WHITE'), 'race'] = 'WHITE'
  combined_data.loc[combined_data['race'].str.contains('BLACK'), 'race'] = 'BLACK'
  combined_data.loc[combined_data['race'].str.contains('ASIAN'), 'race'] = 'ASIAN'
  combined_data.loc[combined_data['race'].str.contains('HISPANIC'), 'race'] = 'HISPANIC'
  
  print(combined_data['admission_type'].value_counts())
  print(combined_data['language'].value_counts())
  print(combined_data['race'].value_counts())
  print(combined_data['gender'].value_counts())

  return combined_data

class MIMICDataset(torch.utils.data.Dataset):
  """MIMIC-CXR."""
  def __init__(self, image_files_list, emb_dir, patient_data, img_transform=None):
    """MIMIC X, C, Y, W dataset.

      ARGS:
        image_files_list: list of paths to imges
        emb_dir: directory with embeding -- structures as e.g.
          ${emb_dir}/p10/p10000032/s5/s56699142_emb.txt
        patient_data: pandas dataframe with data info; output of load_data()
          above
        img_transform: torchvision.Transform
    """
    get_sub = lambda x: int(x.split('/')[-3][1:])
    get_study = lambda x: int(x.split('/')[-2][1:])

    self.img_transform = img_transform
    self.emb_dir = emb_dir
    self.patient_data = patient_data

    self.image_files_list = image_files_list

    self.images = []
    self.embeddings = []
    self.ys = []
    self.races = []
    self.Us = []

    print("loading MIMIC-CXR...")
    for img_name in image_files_list:
      img, emb, y, race, U = self.__load__(img_name)

      self.images.append(img)
      self.embeddings.append(emb)
      self.ys.append(y)
      self.Us.append(U)
      self.races.append(race)

    self.images = np.concatenate(self.images, 0)
    self.embeddings = np.concatenate(self.embeddings, 0)
    self.ys = np.concatenate(self.ys, 0)
    self.races = np.concatenate(self.races, 0)
    print("loaded MIMIC-CXR.")

  def __len__(self):
    return len(self.image_files_list)

  def __load__(self, img_name):
    # if torch.is_tensor(idx):
      # idx = idx.item()

    # img_name = self.image_files_list[idx]
    participant, study = img_name.split('/')[-3:-1]
    participant = int(participant[1:])
    study = int(study[1:])

    patient = self.patient_data[(self.patient_data.subject_id == participant) &
    (self.patient_data.study_id == study)]

    W = patient.iloc[0,4:].values.reshape(1,-1)
    y = np.asarray([[patient.iloc[0,2]]])
    U = np.asarray([[patient.iloc[0,3]]])
    print(U)

    emb_name = os.path.join(self.emb_dir, '/'.join(img_name.split('/')[-4:-1]) +
      '_emb.txt')
    image = Image.open(img_name).convert('RGB')
    emb = np.loadtxt(emb_name).reshape(1,-1)

    # May consider conditionally transposing so w < h.
    if self.img_transform:
      image = np.asarray(self.img_transform(image)).transpose(2,0,1)
    else:
      image = np.asarray(image).transpose(2,0,1)

    image = np.expand_dims(image, 0)

    return image, emb, y, W, U

  def __getitem__(self, idx):
    return self.images[idx], self.embeddings[idx], self.ys[idx].squeeze(), self.races[idx]

class UMIMICDataset(torch.utils.data.Dataset):
  """MIMIC-CXR."""
  def __init__(self, image_files_list, emb_dir, patient_data, img_transform=None):
    """MIMIC X, C, Y, W dataset.

      ARGS:
        image_files_list: list of paths to imges
        emb_dir: directory with embeding -- structures as e.g.
          ${emb_dir}/p10/p10000032/s5/s56699142_emb.txt
        patient_data: pandas dataframe with data info; output of load_data()
          above
        img_transform: torchvision.Transform
    """
    get_sub = lambda x: int(x.split('/')[-3][1:])
    get_study = lambda x: int(x.split('/')[-2][1:])

    self.img_transform = img_transform
    self.emb_dir = emb_dir
    self.patient_data = patient_data

    self.image_files_list = image_files_list

    self.images = []
    self.embeddings = []
    self.ys = []
    self.races = []
    self.Us = []

    print("loading MIMIC-CXR...")
    for img_name in image_files_list:
      img, emb, y, race, U = self.__load__(img_name)

      self.images.append(img)
      self.embeddings.append(emb)
      self.ys.append(y)
      self.Us.append(U)
      self.races.append(race)

    self.images = np.concatenate(self.images, 0)
    self.embeddings = np.concatenate(self.embeddings, 0)
    self.ys = np.concatenate(self.ys, 0)
    self.races = np.concatenate(self.races, 0)
    print("loaded MIMIC-CXR.")

  def __len__(self):
    return len(self.image_files_list)

  def __load__(self, img_name):
    # if torch.is_tensor(idx):
      # idx = idx.item()

    # img_name = self.image_files_list[idx]
    participant, study = img_name.split('/')[-3:-1]
    participant = int(participant[1:])
    study = int(study[1:])

    patient = self.patient_data[(self.patient_data.subject_id == participant) &
    (self.patient_data.study_id == study)]

    W = patient.iloc[0,4:].values.reshape(1,-1)
    y = np.asarray([[patient.iloc[0,2]]])
    U = np.asarray([[patient.iloc[0,3]]])

    emb_name = os.path.join(self.emb_dir, '/'.join(img_name.split('/')[-4:-1]) +
      '_emb.txt')
    image = Image.open(img_name).convert('RGB')
    emb = np.loadtxt(emb_name).reshape(1,-1)

    # May consider conditionally transposing so w < h.
    if self.img_transform:
      image = np.asarray(self.img_transform(image)).transpose(2,0,1)
    else:
      image = np.asarray(image).transpose(2,0,1)

    image = np.expand_dims(image, 0)

    return image, emb, y, W, U

  def __getitem__(self, idx):
    return self.images[idx], self.embeddings[idx], self.ys[idx].squeeze(),
    self.races[idx], self.Us[idx].squeeze()

class DomainDataset(torch.utils.data.Dataset):
  def __init__(self, datasetA, datasetB):
    self.datasetA = datasetA
    self.datasetB = datasetB

  def __getitem__(self, index):
    xA = self.datasetA[index]
    xB = self.datasetB[index]
    return xA, xB

  def __len__(self):
    return len(self.datasetA)

if __name__ == '__main__':
  ret_d = load_data_fl()
  ret_d.to_csv('meta_data_info/mimic_fl.csv')