import json
import os
import shutil
import csv
import pandas as pd
import json
import os
import zipfile
import cv2
from skimage.transform import resize
# from pydicom.pixel_data_handlers.util import apply_voi_lut
# import pydicom
from skimage import exposure
import PIL.Image as Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import time

IMG_PX_SIZE = 512
root_meta_data = 'meta_data_info/mimic/'
root_out_data = '/shared/rsaas/enyij2/mimic/'
out_meta_dir = 'meta_data_info/mimic_small/'
files = ['mimic_table_Black_test.csv']
os.makedirs(out_meta_dir, exist_ok=True)
os.makedirs(root_out_data, exist_ok=True)

for f in files:
    print(f)
    cont = 0
    df = pd.read_csv(os.path.join(root_meta_data, f))
    for idx, row in df.iterrows():
        if cont % 500 == 0:
            print(cont)
        img_path_old = row['img_path']
        lst = img_path_old.split('/')[-3:]
        
        img_path_new = os.path.join(root_out_data, '_'.join(lst))
        # print(img_path_old, img_path_new)
        # st = time.time()
        img = np.asarray(Image.open(img_path_old).convert('L'))
        # print(time.time() - st)
        # st = time.time()
        resized_img = resize(img, (IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True)
        # print(time.time() - st)
        # print(resized_img.max(), resized_img.min(), resized_img.shape)
        # st = time.time()
        im = Image.fromarray((resized_img*255).astype(np.uint8)).convert('L')
        im.save(img_path_new)
        # print(time.time() - st)
        df.loc[idx, 'img_path'] = img_path_new
        cont += 1
    
    df.to_csv(os.path.join(out_meta_dir, f))
        