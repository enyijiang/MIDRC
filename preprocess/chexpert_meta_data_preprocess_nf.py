import pandas as pd
import os
import numpy as np
import csv 
from random import shuffle

base_path = '/shared/rsaas/enyij2/'
out_meta_dir = 'meta_data_info/chexpert2'
data = pd.read_csv(f'meta_data_info/chexpert_train.csv')
conds = data.columns.tolist()[5:-1]
# cur_cond = 'No Finding'
# os.makedirs(out_meta_dir, exist_ok=True)
# pos = 'Cardiomegaly'

lungs = ['Enlarged Cardiomediastinum', 'Fracture'] # 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis']
neg = 'No Finding'
# dignosis = {pos: 'Yes', neg: 'No'}
# out_fp = f'meta_data_info/chexpert_{cur_cond}_table.csv'


out_obj = {}
for cur_cond in lungs:
    print(cur_cond)
    out_obj[cur_cond] = []    
    pos_cont = 0
    for label in ['Yes', 'No']:
        cont = 0
        for _, row in data.iterrows():
            # if cont % 1000 == 0:
            #     print(cont)
            if cont == pos_cont and label == 'No':
                break
            cur = row[5:-1]
            if (label == 'Yes' and cur[cur_cond] == 1) or (label == 'No' and cur[neg] == 1):
                cont += 1           
                out_obj[cur_cond].append([os.path.join(base_path, row[0]), row[1], row[2], row[3], label])
                # else:
                #     out_obj[cur_cond].append([os.path.join(base_path, row[0]), row[1], row[2], row[3], 'Yes'])
        pos_cont = cont
        print(pos_cont)
    shuffle(out_obj[cur_cond])
    print(len(out_obj[cur_cond]))
    split_idx = int(len(out_obj[cur_cond]) * 0.8)
    indices_tr = np.random.choice(len(out_obj[cur_cond]), split_idx, replace=False).tolist()
    indices_ts = list(set([i for i in range(len(out_obj[cur_cond]))]) - set(indices_tr))
    tr_out_obj = [out_obj[cur_cond][idx] for idx in indices_tr]
    ts_out_obj = [out_obj[cur_cond][idx] for idx in indices_ts]
    print(f'num of samples for training {len(tr_out_obj)}')
    print(f'num of samples for testing {len(ts_out_obj)}')
    out_obj[cur_cond] = {'train': tr_out_obj, 'test': ts_out_obj}

for lung in lungs:
    for mode in ['train', 'test']:
        out_fp = f'{out_meta_dir}/chexpert_table_{lung}_{mode}.csv'
        with open(out_fp, 'w') as outcsv:   
            #configure writer to write standard csv file
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(['file_path', 'sex', 'age', 'frontal/lateral', 'positive'])
            for item in out_obj[lung][mode]:
                #Write item to outcsv
                writer.writerow(item)
            outcsv.close()