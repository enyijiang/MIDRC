import json
import csv
import numpy as np
import os

def write_csv(out_fp, out_obj):
    with open(out_fp, 'w') as outcsv:   
        #configure writer to write standard csv file
        writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(['file_path', 'race', 'covid19_positive', 'sex', 'age', 'state'])
        for item in out_obj:
            #Write item to outcsv
            writer.writerow(item)
    outcsv.close()

train_all = 'meta_data_info/states/MIDRC_cr_table_states_train.csv'
test_all = 'meta_data_info/states/MIDRC_cr_table_states_test.csv'
dir_out = 'meta_data_info/states'

train_d = open(train_all, 'r').read().split('\n')
test_d = open(test_all, 'r').read().split('\n')
data_all = {'train': train_d, 'test': test_d}
states = ['CA', 'IN', 'TX']
states_data = {}
for s in states:
    states_data.update({s:[]})

for mode in ['train', 'test']:
    for row in data_all[mode][1:]:
        if row:
            file_path, race, covid19_positive, sex, age, state = row.split(',')
            if state in states:
                states_data[state].append([file_path, race, covid19_positive, sex, age, state])

# randomly split training/testing set
for key in states_data:
    out_obj = states_data[key]
    split_idx = int(len(out_obj) * 0.8) + 1
    indices_tr = np.random.choice(len(out_obj), split_idx, replace=False).tolist()
    indices_ts = list(set([i for i in range(len(out_obj))]) - set(indices_tr))
    tr_out_obj = [out_obj[idx] for idx in indices_tr]
    ts_out_obj = [out_obj[idx] for idx in indices_ts]
    print(f'num of samples {len(out_obj)}')
    print(f'num of samples for training {len(tr_out_obj)}')
    print(f'num of samples for testing {len(ts_out_obj)}')
    out_obj = {'train': tr_out_obj, 'test': ts_out_obj}

    for mode in ['train', 'test']:
        out_fp = f'{dir_out}/MIDRC_table_{key}_{mode}.csv'
        # out_fp_NC = f'{dir_out}/MIDRC_table_NC_{mode}.csv'
        write_csv(out_fp, out_obj[mode])
        # write_csv(out_fp_NC, NC_data[mode])



