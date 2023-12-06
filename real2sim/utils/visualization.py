import os
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

import mediapy as media
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation
from matplotlib import pyplot as plt

FONT_PATH = str(Path(__file__) / "fonts/UbuntuMono-R.ttf")

_rng = np.random.RandomState(0)
_palette = ((_rng.random((3*255))*0.7+0.3)*255).astype(np.uint8).tolist()
_palette = [0, 0, 0]+_palette



def write_video(path, images, fps=5):
    # images: list of numpy arrays
    root_dir = Path(path).parent
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not isinstance(images[0], np.ndarray):
        images_npy = [image.numpy() for image in images]
    else:
        images_npy = images
    media.write_video(path, images_npy, fps=fps)
    
    
    
def plot_pred_and_gt_action_trajectory(predicted_actions, gt_actions, stacked_images):
    """
    Plot predicted and ground truth action trajectory
    Args:
        predicted_actions: list of dict with keys as ['terminate_episode', 'world_vector', 'rotation_delta', 'gripper_closedness_action']
        gt_actions: list of dict with keys as ['terminate_episode', 'world_vector', 'rotation_delta', 'gripper_closedness_action']
        stacked_images: np.array, [H, W * n_images, 3], uint8 (here n_images does not need to be the same as the length of predicted_actions or gt_actions)
    """
    
    action_name_to_values_over_time = defaultdict(list)
    predicted_action_name_to_values_over_time = defaultdict(list)
    figure_layout = ['terminate_episode_0', 'terminate_episode_1',
            'terminate_episode_2', 'world_vector_0', 'world_vector_1',
            'world_vector_2', 'rotation_delta_0', 'rotation_delta_1',
            'rotation_delta_2', 'gripper_closedness_action_0']
    action_order = ['terminate_episode', 'world_vector', 'rotation_delta', 'gripper_closedness_action']

    for i, action in enumerate(gt_actions):
        for action_name in action_order:
            for action_sub_dimension in range(action[action_name].shape[0]):
                # print(action_name, action_sub_dimension)
                title = f'{action_name}_{action_sub_dimension}'
                action_name_to_values_over_time[title].append(action[action_name][action_sub_dimension])
                predicted_action_name_to_values_over_time[title].append(predicted_actions[i][action_name][action_sub_dimension])
            
    figure_layout = [
        ['image'] * len(figure_layout),
        figure_layout
    ]

    plt.rcParams.update({'font.size': 12})

    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([45, 10])

    for i, (k, v) in enumerate(action_name_to_values_over_time.items()):

        axs[k].plot(v, label='ground truth')
        axs[k].plot(predicted_action_name_to_values_over_time[k], label='predicted action')
        axs[k].set_title(k)
        axs[k].set_xlabel('Time in one episode')

    axs['image'].imshow(stacked_images)
    axs['image'].set_xlabel('Time in one episode (subsampled)')

    plt.legend()
    plt.show()



def colorize_mask(pred_mask: np.ndarray) -> np.ndarray:
    """Colorize a predicted mask
    :param pred_mask: [H, W] bool/np.uint8 np.ndarray
    :return mask: colorized mask, [H, W, 3] np.uint8 np.ndarray
    """
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.asarray(save_mask)



def draw_mask(rgb_img, mask, alpha=0.5, id_countour=False) -> np.ndarray:
    """Overlay predicted mask on rgb image
    :param rgb_img: RGB image, [H, W, 3] np.uint8 np.ndarray
    :param mask: [H, W] bool/np.uint8 np.ndarray
    :param alpha: overlay transparency
    :return img_mask: mask-overlayed image, [H, W, 3] np.uint8 np.ndarray
    """
    img_mask = rgb_img.copy()
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        for id in obj_ids:
            # Overlay color on binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0, 0, 0]
            foreground = rgb_img * (1-alpha) + np.ones_like(rgb_img) * \
                alpha * np.asarray(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask != 0)
        countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
        foreground = rgb_img*(1-alpha) + colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours, :] = 0
    return img_mask



def draw_bbox(rgb_image: np.ndarray, labels: List[str], bboxes: np.ndarray,
              pred_indices: np.ndarray, pred_scores: np.ndarray,
              bbox_width=2, text_size=25, sort_by_score=True) -> np.ndarray:
    """Draw bbox predictions on rgb image

    :param rgb_image: RGB image, [H, W, 3] np.uint8 np.ndarray
    :param labels: list of label strings
    :param bboxes: bbox as XYXY pixel coordinates, [n_bbox, 4] np.float32 np.ndarray
    :param pred_indices: predicted label indices, [n_bbox,] integer np.ndarray
    :param pred_scores: predicted scores, [n_bbox,] np.float32 np.ndarray
    :param bbox_width: line width to draw bbox
    :param text_size: text size to write predicted label
    :param sort_by_score: plot bboxes with lower scores first
                          so bboxes with higher score are visible
    :return out_image: rgb_image with drawn bboxes, [H, W, 3] np.uint8 np.ndarray
    """
    font = ImageFont.truetype(FONT_PATH, text_size)

    H, W = rgb_image.shape[:2]
    rgb_im = Image.fromarray(rgb_image).convert("RGBA")
    # make a blank image for text, initialized to transparent text color
    txt_im = Image.new("RGBA", rgb_im.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(txt_im)

    if sort_by_score:
        sorted_idx = pred_scores.argsort()
        bboxes = bboxes[sorted_idx]
        pred_indices = pred_indices[sorted_idx]
        pred_scores = pred_scores[sorted_idx]

    def _pad_bbox(bbox: Tuple[float], pad: float) -> Tuple[float]:
        left, top, right, bottom = bbox
        return (left - pad, top - pad, right + pad, bottom + pad)

    for (x1, y1, x2, y2), pred_index, pred_score in zip(bboxes, pred_indices,
                                                        pred_scores):
        # draw bbox (left, top, right, bottom)
        d.rectangle([x1, y1, x2, y2], fill=None, outline=(255, 0, 0), width=bbox_width)

        # draw text
        text = f"{labels[pred_index]}: {pred_score:1.2f}"
        anchor_xy = [x1 + text_size*0.1, y2 + text_size*0.1 + 1]
        anchor = 'lt'
        text_bbox = d.textbbox(anchor_xy, text, font=font, anchor=anchor)
        text_bbox = _pad_bbox(text_bbox, text_size*0.1)
        if text_bbox[3] > H and text_bbox[2] > W:  # bottom-right
            anchor_xy = [x2 - text_size*0.1, y1 - text_size*0.1 - 1]
            anchor = 'rb'
            text_bbox = d.textbbox(anchor_xy, text, font=font, anchor=anchor)
            text_bbox = _pad_bbox(text_bbox, text_size*0.1)
        elif text_bbox[3] > H:  # bottom
            anchor_xy = [x1 + text_size*0.1, y1 - text_size*0.1 - 1]
            anchor = 'lb'
            text_bbox = d.textbbox(anchor_xy, text, font=font, anchor=anchor)
            text_bbox = _pad_bbox(text_bbox, text_size*0.1)
        elif text_bbox[2] > W:  # right
            anchor_xy = [x2 - text_size*0.1, y2 + text_size*0.1 + 1]
            anchor = 'rt'
            text_bbox = d.textbbox(anchor_xy, text, font=font, anchor=anchor)
            text_bbox = _pad_bbox(text_bbox, text_size*0.1)
        # draw text bbox (bg only)
        d.rectangle(text_bbox, fill=(255, 255, 255), outline=None, width=1)
        d.text(anchor_xy, text, fill=(0, 0, 0), font=font, anchor=anchor)

    out_im = Image.alpha_composite(rgb_im, txt_im).convert("RGB")
    return np.asarray(out_im).copy()  # copy makes it writable