from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers.image_transforms import center_to_corners_format
from transformers.models.owlvit.image_processing_owlvit import box_iou

from real2sim.utils.io import load_image_arrays


def post_process_object_detection(outputs, threshold=0.1, nms_threshold=0.3,
                                  target_sizes=None):
    logits, boxes = outputs.logits, outputs.pred_boxes

    if target_sizes is not None:
        if len(logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes "
                             "as the batch dimension of the logits")
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain "
                             "the size (h, w) of each image of the batch")

    probs = torch.max(logits, dim=-1)
    scores = torch.sigmoid(probs.values)
    labels = probs.indices

    # Convert to [x0, y0, x1, y1] format
    boxes = center_to_corners_format(boxes)

    # Apply non-maximum suppression (NMS)
    if nms_threshold < 1.0:
        for idx in range(boxes.shape[0]):  # batch
            for i in torch.argsort(-scores[idx]):
                if not scores[idx][i]:
                    continue

                ious = box_iou(boxes[idx][i, :].unsqueeze(0), boxes[idx])[0][0]
                ious[i] = -1.0  # Mask self-IoU.
                scores[idx][ious > nms_threshold] = 0.0

    # Convert from relative [0, 1] to absolute [0, height] coordinates
    if target_sizes is not None:
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

    results = []
    for s, l, b in zip(scores, labels, boxes):
        score = s[s > threshold]
        label = l[s > threshold]
        box = b[s > threshold]
        results.append({"scores": score, "labels": label, "boxes": box})
    return results


class Owl_ViT:
    """OWL_VIT for object detection"""

    VARIANTS = [
        "google/owlvit-base-patch32",
        "google/owlvit-base-patch16",
        "google/owlvit-large-patch14",
        "google/owlv2-base-patch16-ensemble",
        "google/owlv2-large-patch14-ensemble",
        # "google/owlv2-large-patch14-finetuned" # this is not as strong as ensemble
    ]

    def __init__(
        self,
        root_path: Union[str, Path] = "/rl_benchmark",
        model_variant="google/owlv2-large-patch14-ensemble",
        box_threshold=0.1,
        nms_threshold=1.0,
        device="cuda"
    ):
        """
        :param box_threshold: filtering threshold for bbox
        :param nms_threshold: NMS IoU threshold (1.0 is no NMS)
        """
        root_path = Path(root_path)
        assert model_variant in self.VARIANTS, f"Unknown {model_variant = }"
        self.model_variant = model_variant
        self.is_v2 = 'v2' in model_variant

        self.box_threshold = box_threshold
        self.nms_threshold = nms_threshold
        self.device = device

        self.load_model()

    def load_model(self):
        if not self.is_v2:
            self.model = OwlViTForObjectDetection.from_pretrained(self.model_variant)
            self.processor = OwlViTProcessor.from_pretrained(self.model_variant)
        else:
            self.model = Owlv2ForObjectDetection.from_pretrained(self.model_variant)
            self.processor = Owlv2Processor.from_pretrained(self.model_variant)

        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        images: Union[str, List[str], np.ndarray, List[np.ndarray]],
        prompts: List[str],
        return_on_cpu=False,
        verbose=False,
    ) -> Union[Tuple[torch.Tensor, np.ndarray, np.ndarray],
               Tuple[List[torch.Tensor], List[np.ndarray], List[np.ndarray]]]:
        """
        :param images: Input RGB images, can be
                       a string path, a list of string paths,
                       a [H, W, 3] np.uint8 np.ndarray,
                       a list of [H, W, 3] np.uint8 np.ndarray,
                       a [n_images, H, W, 3] np.uint8 np.ndarray
        :param prompts: a list of text detection prompts, same for all n_images
                        ["mug", "mug handle"]
        :param return_on_cpu: whether to return boxes as cuda Tensor or numpy array
        :param verbose: whether to print debug info
        :return boxes: (n_images) list of pred_bbox as XYXY pixel coordinates
                       [n_bbox, 4] torch.float32 cuda Tensor
        :return pred_indices: (n_images) list of [n_bbox,] integer np.ndarray
        :return pred_scores: (n_images) list of [n_bbox,] np.float32 np.ndarray
        """
        with torch.cuda.device(self.device):
            # Process images and prompts
            images, is_list = load_image_arrays(images)
            prompts = [prompts] * len(images)

            inputs = self.processor(
                text=prompts, images=images, return_tensors="pt"
            ).to(self.device)

            # Print input names and shapes
            if verbose:
                for key, val in inputs.items():
                    print(f"[ OWL_VIT ] {key}: {val.shape}")

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)

            if verbose:
                for k, val in outputs.items():
                    if k not in {"text_model_output", "vision_model_output"}:
                        print(f"[ OWL_VIT ] {k}: shape of {val.shape}")

                print("\n[ OWL_VIT ] Text model outputs")
                for k, val in outputs.text_model_output.items():
                    print(f"[ OWL_VIT ] {k}: shape of {val.shape}")

                print("\n[ OWL_VIT ] Vision model outputs")
                for k, val in outputs.vision_model_output.items():
                    print(f"[ OWL_VIT ] {k}: shape of {val.shape}")

            # Target image sizes (H, W) to rescale box predictions [batch_size, 2]
            if not self.is_v2:
                target_sizes = torch.Tensor(
                    [image.shape[:2] for image in images]
                ).to(self.device)
            else:
                # the bounding boxes are with respect to the padded image, 
                # so we want to recover the boxes with respect to the original image
                # TODO: If huggingface fixes this issue, we can remove this
                target_sizes = torch.Tensor(
                    [inputs.pixel_values.shape[-2:] for image in images]
                ).to(self.device)

            # Convert outputs (bounding boxes and class logits) to COCO API
            results = post_process_object_detection(
                outputs=outputs, threshold=self.box_threshold,
                nms_threshold=self.nms_threshold,
                target_sizes=target_sizes
            )
            
            if self.is_v2:
                # the bounding boxes are with respect to the padded image, 
                # so we want to recover the boxes with respect to the original image
                # TODO: If huggingface fixes this issue, we can remove this
                orig_h, orig_w = images[0].shape[:2]
                padded_h, padded_w = inputs.pixel_values.shape[-2:]
                if orig_h >= orig_w:
                    scale_coeff = padded_h / orig_h
                    padded_w = int(orig_w * scale_coeff)
                else:
                    scale_coeff = padded_w / orig_w
                    padded_h = int(orig_h * scale_coeff)
                for result in results:
                    result["boxes"][:, [0, 2]] = result["boxes"][:, [0, 2]] * orig_w / padded_w
                    result["boxes"][:, [1, 3]] = result["boxes"][:, [1, 3]] * orig_h / padded_h

            if verbose:
                for k, val in results[0].items():
                    print(f"[ OWL_VIT ] {k}: shape of {val.shape}")

        if is_list:
            # Reformat output
            boxes, pred_indices, pred_scores = [], [], []
            for result in results:
                boxes.append(result["boxes"].cpu().numpy()
                             if return_on_cpu else result["boxes"])
                pred_indices.append(result["labels"].cpu().numpy())
                pred_scores.append(result["scores"].cpu().numpy())

            return boxes, pred_indices, pred_scores
        else:
            result = results[0]
            return (
                result["boxes"].cpu().numpy() if return_on_cpu else result["boxes"],
                result["labels"].cpu().numpy(),
                result["scores"].cpu().numpy()
            )
