import os
import cv2
import numpy as np
from PIL import Image, ExifTags
import piexif
import sys
import argparse

# python video_to_image.py --folder /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke --video-frame 0
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True)
args = parser.parse_args()

def align(s):
    while (len(s) < 4):
        s = '0' + s
    return s.lower()

img_id = 0
for image in os.listdir(args.folder):
    if ("jpg" in image.lower() or "jpeg" in image.lower()):
        img = Image.open(os.path.join(args.folder, image))
        img_arr = np.asarray(img)
        if img_arr.shape[0] > img_arr.shape[1]:
            img_arr = img_arr[:, ::-1, :].transpose(1, 0, 2)
        Image.fromarray(img_arr).save(os.path.join(args.folder, align(str(img_id) + ".jpg")))
        img_id += 1