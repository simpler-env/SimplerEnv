import os
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image, ExifTags
import glob
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--threshold', type=int, default=8000)
    args = parser.parse_args()
    
    flist = glob.glob(args.folder + '**/*.jpg', recursive=True) + glob.glob(args.folder + '**/*.png', recursive=True)
    
    for fname in flist:
        orig_image = np.array(Image.open(fname))
        if len(orig_image.shape) == 2:
            orig_image = orig_image[..., None]
            squeeze = True
        else:
            squeeze = False
        if orig_image.shape[-1] == 4:
            image = orig_image[..., -1]
        else:
            image = orig_image.sum(axis=-1)
        image_labeled, n_labels = ndimage.label(image)
        for label_id in range(1, n_labels + 1):
            label = (image_labeled == label_id)
            if label.sum() < args.threshold:
                orig_image[label] = 0
        if (orig_image.sum(axis=-1) > 0).sum() == 0:
            os.remove(fname)
        else:
            if squeeze:
                orig_image = orig_image.squeeze(-1)
            Image.fromarray(orig_image).save(fname)
    
    
    
if __name__ == '__main__':
    main()