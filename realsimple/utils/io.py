import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image


def is_path(path) -> bool:
    return isinstance(path, (str, Path))


def load_image_pils(image_paths: Union[str, List[str], np.ndarray, List[np.ndarray]]) -> Union[List["Image"], bool]:
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
    elif isinstance(image_paths, list) and image_paths[0].ndim == 3 or image_paths.ndim == 4:  # [n_images, H, W, 3]
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
    if is_path(image_paths) or isinstance(image_paths, list) and is_path(image_paths[0]):
        image_pils, is_list = load_image_pils(image_paths)
        return [np.asarray(image_pil) for image_pil in image_pils], is_list
    elif isinstance(image_paths, list) and image_paths[0].ndim == 3 or image_paths.ndim == 4:  # [n_images, H, W, 3]
        return list(image_paths), True
    elif image_paths.ndim == 3:  # [H, W, 3]
        return [image_paths], False
    else:
        raise ValueError(f"Wrong format of image_paths: {type(image_paths)}")


class DictAction(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple: The expanded list or tuple from the string.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count("(") == string.count(")")) and (
                string.count("[") == string.count("]")
            ), f"Imbalanced brackets exist in {string}"
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if (char == ",") and (pre.count("(") == pre.count(")")) and (pre.count("[") == pre.count("]")):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip("'\"").replace(" ", "")
        is_tuple = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif "," not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1 :]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split("=", maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)
