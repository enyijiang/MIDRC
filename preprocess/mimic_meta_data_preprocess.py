import pandas as pd
import os
import numpy as np
import csv 
from random import shuffle
import re

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
#   print("metadata size: ", metadata.shape)
  chexpert = pd.read_csv(chexpert_path).fillna(0.)
#   print("chexpert size: ", chexpert.shape)
  admissions = pd.read_csv(admissions_path).loc[:, ['subject_id', 'admission_type', 'race']].drop_duplicates('subject_id')
#   print("admissions size: ", admissions.shape)
  patients = pd.read_csv(patients_path).loc[:, ['subject_id',
  'gender', 'anchor_age']].drop_duplicates('subject_id')

#   print("patients size: ", patients.shape)

  CONDITIONS = list(chexpert.columns)[2:]
  print("CONDITIONS: ", CONDITIONS)

  print("merging with chexpert...")
  combined_data = metadata.merge(chexpert, how='inner',
  on=['subject_id', 'study_id'])
#   print("combined_data size: ", combined_data.shape)

  print("merging with admissions...")
  combined_data = combined_data.merge(admissions, how='inner',
  on='subject_id')
#   print("combined_data size: ", combined_data.shape)

  print("merging with patients...")
  combined_data = combined_data.merge(patients, how='inner',
    on='subject_id')
#   print("combined_data size: ", combined_data.shape)

  for condition in CONDITIONS:
    combined_data = combined_data[combined_data[condition] != -1]

#   count = 0
#   combined_data['img_path'] = '' 
#   for img_name in image_files_list:
#     if count % 1000 == 0:
#         print(count)
#     participant, study, dicom = img_name.split('/')[-3:]
#     participant = int(participant[1:])
#     study = int(study[1:])
#     dicom = dicom.split('.')[0]

#     try:
#         combined_data.loc[(combined_data.subject_id == participant) &
#     (combined_data.study_id == study) & (combined_data.dicom_id == dicom), 'img_path'] =  img_name
#     except Exception as e: print(e)

#     count += 1

  combined_data.loc[combined_data['race'].str.contains('WHITE'), 'race'] = 'WHITE'
  combined_data.loc[combined_data['race'].str.contains('BLACK'), 'race'] = 'BLACK'
  combined_data.loc[combined_data['race'].str.contains('ASIAN'), 'race'] = 'ASIAN'
  combined_data.loc[combined_data['race'].str.contains('HISPANIC'), 'race'] = 'HISPANIC'

  white_data = combined_data.loc[combined_data['race'] == 'WHITE']
  black_data = combined_data.loc[combined_data['race'] == 'BLACK']
  asian_data = combined_data.loc[combined_data['race'] == 'ASIAN']
  hispanic_data = combined_data.loc[combined_data['race'] == 'HISPANIC']

  return white_data, black_data, asian_data, hispanic_data

base_path = '/shared/rsaas/enyij2/'
out_meta_dir = 'meta_data_info/mimic'
image_files_list = open('meta_data_info/mimic_image_files.txt', 'r').read().split('\n')[:-1]
sample_size = 4000
print(len(image_files_list))
os.makedirs(out_meta_dir, exist_ok=True)

white_data, black_data, asian_data, hispanic_data = load_data_fl()
race2data = {'White': white_data, 'Black': black_data, 'Asian': asian_data, 'Hispanic': hispanic_data}
# print(black_data['admission_type'].value_counts())
# print(black_data['race'].value_counts())
# print(black_data['gender'].value_counts())

out_obj = {}
for race in race2data:
    print(race)
    out_obj[race] = []
    df = race2data[race]    
    print(df['No Finding'].value_counts())
    for label in ['Yes', 'No']:
        cont = 0
        for _, row in df.iterrows():
            if cont % 100 == 0:
                print(cont)
            if cont == sample_size:
                continue
            no_finding = row['No Finding']
            if label == 'Yes' and no_finding == 0:
                r = re.compile(f".*/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg")
                img_path = list(filter(r.match, image_files_list))[0] # Read Note below           
                out_obj[race].append([img_path, row['admission_type'], row['gender'], row['anchor_age'], label])
                cont += 1
            elif label == 'No' and no_finding == 1.0:   
                r = re.compile(f".*/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg")
                img_path = list(filter(r.match, image_files_list))[0] # Read Note below           
                out_obj[race].append([img_path, row['admission_type'], row['gender'], row['anchor_age'], label])
                cont += 1
                # else:
                #     out_obj[race].append([os.path.join(base_path, row[0]), row[1], row[2], row[3], 'Yes'])


    shuffle(out_obj[race])
    print(len(out_obj[race]))
    split_idx = int(len(out_obj[race]) * 0.8)
    indices_tr = np.random.choice(len(out_obj[race]), split_idx, replace=False).tolist()
    indices_ts = list(set([i for i in range(len(out_obj[race]))]) - set(indices_tr))
    tr_out_obj = [out_obj[race][idx] for idx in indices_tr]
    ts_out_obj = [out_obj[race][idx] for idx in indices_ts]
    print(f'num of samples for training {len(tr_out_obj)}')
    print(f'num of samples for testing {len(ts_out_obj)}')
    out_obj[race] = {'train': tr_out_obj, 'test': ts_out_obj}

for race in race2data:
    for mode in ['train', 'test']:
        out_fp = f'{out_meta_dir}/mimic_table_{race}_{mode}.csv'
        with open(out_fp, 'w') as outcsv:   
            #configure writer to write standard csv file
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(['img_path', 'admission_type', 'gender', 'age', 'positive'])
            for item in out_obj[race][mode]:
                #Write item to outcsv
                writer.writerow(item)
            outcsv.close()