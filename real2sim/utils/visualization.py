import mediapy as media
import numpy as np

def write_video(path, images, fps=5):
    # images: list of numpy arrays
    if not isinstance(images[0], np.ndarray):
        images_npy = [image.numpy() for image in images]
    else:
        images_npy = images
    media.write_video(path, images_npy, fps=fps)