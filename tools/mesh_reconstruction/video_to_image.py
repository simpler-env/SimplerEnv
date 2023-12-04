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
parser.add_argument('--video-frame', type=int, default=-1)
parser.add_argument('--remove-video', action='store_true')
args = parser.parse_args()

folder_path = args.folder
if "calibration" in folder_path:
    images = os.path.join(folder_path, "images")
else:
    images = folder_path
os.makedirs(images, exist_ok = True)

def align(s):
    while (len(s) < 4):
        s = '0' + s
    return s.lower()


# def delete_exif_and_resize(folder, name):
#     empty_exif = {}
#     exif_bytes = piexif.dump({"Exif": empty_exif})
#     piexif.insert(exif_bytes, os.path.join(folder, name))
#     image = Image.open(os.path.join(folder, name))
#     image = image.resize((1920, 1080))
#     os.remove(os.path.join(folder, name))
#     image.save(os.path.join(images, align(name)))


def average_video(folder, name, video_frame):
    cap = cv2.VideoCapture(os.path.join(folder, name))

    ave = None

    if video_frame >= 0:
        for i in range(video_frame + 1):
            success, frame = cap.read()
        assert success
        ave = frame.astype(np.uint8)
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(total_frames):
            success, frame = cap.read()
            if ave is None:
                ave = np.zeros(frame.shape, dtype=np.uint32)
            if success:
                ave += frame
        ave = (ave / total_frames).astype(np.uint8)
        
    cv2.imwrite(os.path.join(images, align(name[: -4] + ".jpg")), ave)
    cap.release()
    if args.remove_video:
        os.remove(os.path.join(folder, name))

for image in os.listdir(folder_path):
    if ("jpg" in image.lower()):
        # delete_exif_and_resize(folder_path, image)
        pass
    elif ("mp4" in image.lower()):
        average_video(folder_path, image, args.video_frame)
    elif not "images" in image.lower():
        os.remove(os.path.join(folder_path, image))

