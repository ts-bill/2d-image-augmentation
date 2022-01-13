import os
import cv2
import numpy as np
import imageio
from tqdm import tqdm
im_path = "output/spec/"
im_list = []

for x in os.listdir(im_path):
    if x.endswith(".png") or x.endswith(".jpg"):
        # Prints only text file present in My Folder
        print(x)
        im_list.append(x)
for picname in tqdm(im_list):
    filename = picname.split(".")
    pipe_image = imageio.imread(im_path + picname)
    re_image = cv2.resize(pipe_image, (640, 640))
    imageio.imwrite('output/spec_out/' + filename[0] +'.png', re_image)