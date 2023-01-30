import pydicom as dicom
from skimage.transform import resize
import matplotlib.pylab as plt
import time
import os
from matplotlib import image
import cv2
import numpy as np
from PIL import Image

img_path = '/shared/rsaas/enyij2/midrc/data_subset/dx/clean'
pics = os.listdir(img_path)
print(len(pics))
IMG_PX_SIZE = 256
st = time.time()
img = image.imread(os.path.join(img_path, pics[0]))
# print(time.time() - st)
print(img.shape)
plt.imshow(img)
# plt.savefig('meta_data_info/xyz.png')
# st = time.time()
resized_img = resize(img, (IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True)
# print(time.time() - st)

# plt.imshow(resized_img)
# plt.savefig('meta_data_info/xyz.png')



# img = cv2.imread(os.path.join(img_path, pics[0]), 1)
# converting to LAB color space
lab = cv2.cvtColor(resized_img., cv2.COLOR_GRAY2BGR)
lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)
l_channel, a, b = cv2.split(lab)

# Applying CLAHE to L-channel
# feel free to try different values for the limit and grid size:
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(l_channel)

# merge the CLAHE enhanced L-channel with the a and b channel
limg = cv2.merge((cl,a,b))

# Converting image from LAB Color model to BGR color spcae
enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Stacking the original image with the enhanced image
result = np.hstack((img, enhanced_img))
cv2.imwrite('meta_data_info/xyz_new.png', result)