from pathlib import Path
from typing import Union, List

import numpy as np
from PIL import Image


def is_path(path) -> bool:
    return isinstance(path, (str, Path))


def load_image_pils(
    image_paths: Union[str, List[str], np.ndarray, List[np.ndarray]]
) -> Union[List["Image"], bool]:
    """Load images from paths and return as a list of PIL.Image
    :return is_list: whether input is a list
    """
    # load image
    if is_path(image_paths):
        image_pil = Image.open(image_paths).convert("RGB")  # load image
        return [image_pil], False
    elif isinstance(image_paths, list) and is_path(image_paths[0]):
        image_pils = []
        for image_path in image_paths:
            image_pil = Image.open(image_path).convert("RGB")  # load image
            image_pils.append(image_pil)
        return image_pils, True
    elif isinstance(image_paths, list) and image_paths[0].ndim == 3 \
            or image_paths.ndim == 4:  # [n_images, H, W, 3]
        image_pils = []
        for image in image_paths:
            image_pil = Image.fromarray(image).convert("RGB")
            image_pils.append(image_pil)
        return image_pils, True
    elif image_paths.ndim == 3:  # [H, W, 3]
        image_pil = Image.fromarray(image_paths).convert("RGB")
        return [image_pil], False
    else:
        raise ValueError(f"Wrong format of image_paths: {type(image_paths)}")


def load_image_arrays(
    image_paths: Union[str, List[str], np.ndarray, List[np.ndarray]]
) -> Union[List[np.ndarray], bool]:
    """Load images and return as a list of [H, W, 3] np.ndarray
    :return is_list: whether input is a list
    """
    if is_path(image_paths) or \
       isinstance(image_paths, list) and is_path(image_paths[0]):
        image_pils, is_list = load_image_pils(image_paths)
        return [np.asarray(image_pil) for image_pil in image_pils], is_list
    elif isinstance(image_paths, list) and image_paths[0].ndim == 3 \
            or image_paths.ndim == 4:  # [n_images, H, W, 3]
        return list(image_paths), True
    elif image_paths.ndim == 3:  # [H, W, 3]
        return [image_paths], False
    else:
        raise ValueError(f"Wrong format of image_paths: {type(image_paths)}")
