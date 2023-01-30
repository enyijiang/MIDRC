import json
import os
import shutil
import csv
import json
import os
import zipfile
from skimage.transform import resize
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pydicom
from skimage import exposure
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt

races = ['White', 'Black or African American', 'Asian', 'Native Hawaiian or other Pacific Islander', 'American Indian or Alaska Native', 'Other']
methods = ['cr', 'dx']
# m2num = {'cr': 2000, 'dx': 500}
root_dir = '/shared/rsaas/enyij2/midrc/data_subset'
IMG_PX_SIZE = 256
states2num = {}
age2num = {}
# read in zipcode csv file
zip2states = {}
zip3_csv = open('meta_data_info/zip2loc.csv', 'r').read().split('\n')
for row in zip3_csv[1:]:
    zipcode, city, state = row.split(',')
    zip2states[int(zipcode)] = state

def plot_bars(dict, x_label, method):
    names = list(dict.keys())
    values = list(dict.values())
    if x_label == 'Age':
        names = [n*10 for n in names]
    bars = plt.bar(range(len(dict)), values, tick_label=names)
    plt.xlabel(x_label)
    plt.ylabel('Number of Images')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, yval)
    
    plt.savefig(f'meta_data_info/states/{method}_{x_label}_dist.png')
    
    plt.clf()

def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and (dicom.PhotometricInterpretation == "MONOCHROME1"):
        # print('haha')
        data = np.amax(data) - data
    
    data = data - np.min(data)
        
    return data


for method in ['cr']:
    data_in = os.path.join(root_dir, method)
    data_out =  os.path.join(root_dir, method, 'clean_states')
    os.makedirs(data_out, exist_ok=True)

    # empty the original files
    files = glob.glob(os.path.join(data_out, '*'))
    # print(files)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    json_in = f'meta_data_info/MIDRC_{method}_table.json'
    # os.makedirs(data_out, exist_ok=True)
    obj = json.load(open(json_in, 'r'))
    out_obj = []
    cont = 0
    for case in obj:
        if cont % 1000 == 0:
            print(cont)
        # if cont > 10:
        #     break
        if 'file_name' in case:            
            race = case['race'][0]
            age = case['age_at_index'][0] if 'age_at_index' in case else None
            zipcode = case['zip'][0] if 'zip' in case else None
    
            if age and zipcode and zipcode > 0:              
                # print(age//10)
                if race in races:
                    # is_covid = int(case['covid19_positive'][0] == "Yes")
                    file_names = case['file_name']
                    if file_names:
                        state = zip2states[zipcode]
                        if state not in states2num:
                            states2num[state] = 0 
                        if age//10 not in age2num:
                            age2num[age//10] = 0
                        for fn in file_names:
                            file = fn[:-4].split('/')[-1]
                            dst_path = os.path.join(data_in, file)
                            if os.path.exists(os.path.join(data_in, fn)) and fn.split('.')[-1] == 'zip':
                                try:
                                    with zipfile.ZipFile(os.path.join(data_in, fn), 'r') as zip_ref:
                                        zip_ref.extractall(dst_path)
                                except:
                                    print("An exception occurred when unzipping")
                                imgs = os.listdir(os.path.join(dst_path, file))
                                for img in imgs:
                                    states2num[state] += 1
                                    age2num[age//10] += 1
                                    img_old_path = os.path.join(dst_path, file, img)
                                    img_new_path = os.path.join(data_out, img[:-4] + '.jpeg')
                                    img = read_xray(img_old_path)
                                    img = exposure.equalize_hist(img)
                                    # print(img.max(), img.min(), img.shape)
                                    resized_img = resize(img, (IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True)
                                    # print(resized_img.max(), resized_img.min(), resized_img.shape)
                                    im = Image.fromarray(resized_img*255).convert("L")
                                    im.save(img_new_path)
                                    out_obj.append([img_new_path, race, case['covid19_positive'][0], case['sex'][0], age, state])   
                                    cont += 1   

    split_idx = int(len(out_obj) * 0.8) + 1
    tr_out_obj = out_obj[:split_idx]
    ts_out_obj = out_obj[split_idx:]
    print(f'num of samples for training {len(tr_out_obj)}')
    print(f'num of samples for testing {len(ts_out_obj)}')
    out_obj = {'train': tr_out_obj, 'test': ts_out_obj}
    # out_fp_tr = f'meta_data_info/MIDRC_{method}_table_race_train_all.csv'
    # out_fp_ts = f'meta_data_info/MIDRC_{method}_table_race_test_all.csv'

    for mode in ['train', 'test']:
        out_fp = f'meta_data_info/states/MIDRC_{method}_table_states_{mode}.csv'
        with open(out_fp, 'w') as outcsv:   
            #configure writer to write standard csv file
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(['file_path', 'race', 'covid19_positive', 'sex', 'age', 'state'])
            for item in out_obj[mode]:
                #Write item to outcsv
                writer.writerow(item)
        outcsv.close()
    
    states2num = dict(sorted(states2num.items(), key=lambda item: item[1], reverse=True))
    age2num = dict(sorted(age2num.items(), key=lambda item: item[0]))
    print(states2num)
    for s in list(states2num):
        if states2num[s] < 10:
            del states2num[s]
    
    plot_bars(states2num, 'States', method)
    plot_bars(age2num, 'Age', method)

