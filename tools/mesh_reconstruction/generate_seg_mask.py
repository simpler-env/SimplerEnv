import os, argparse, cv2
import numpy as np
import torch
from PIL import Image
from real2sim.utils.detection import Owl_ViT
from real2sim.utils.segmentation import SAM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--sam-root-path', type=str, default='/home/xuanlin/kolin_maniskill2/rl_benchmark/Segment-and-Track-Anything')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--min-seg-pixels', type=int, default=3000)
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.input_dir, 'seg'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'seg_image', 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'seg_image', 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'seg_image', 'test'), exist_ok=True)
    detector = Owl_ViT(box_threshold=0.1, nms_threshold=0.3, 
                       model_variant='google/owlv2-large-patch14-ensemble', device=args.device)
    segmenter = SAM(root_path=args.sam_root_path, 
                        model_variant='vit_b',
                        device=args.device)
    
    for image_name in os.listdir(args.input_dir):
        if image_name.endswith('.jpg'):
            image = cv2.cvtColor(cv2.imread(os.path.join(args.input_dir, image_name)), cv2.COLOR_BGR2RGB)
            image = np.asarray(image)
            
            boxes, pred_indices, pred_scores = detector(image, [args.prompt], return_on_cpu=True)
            max_score_id = np.argmax(pred_scores)
            box, pred_score = boxes[max_score_id], pred_scores[max_score_id]
            
            mask, miou = segmenter(image, boxes=box, return_on_cpu=True)
            seg_pixels = np.sum(mask)
            print(image_name, mask.shape, mask.dtype, miou, seg_pixels)
            
            mask = mask.astype(np.uint8) * 255
            if seg_pixels >= args.min_seg_pixels:
                Image.fromarray(mask).save(os.path.join(args.input_dir, 'seg', image_name[:-4] + '.jpg.png'))
                masked_image = image.copy().astype(np.uint8)
                masked_image[mask == 0] = 0
                Image.fromarray(masked_image).save(os.path.join(args.input_dir, 'seg_image', 'images', image_name[:-4] + '.jpg'))
                Image.fromarray(masked_image).save(os.path.join(args.input_dir, 'seg_image', 'train', image_name[:-4] + '.jpg'))
                Image.fromarray(masked_image).save(os.path.join(args.input_dir, 'seg_image', 'test', image_name[:-4] + '.jpg'))
                masked_image_rgba = np.concatenate([masked_image, mask[..., None]], axis=-1)
                assert masked_image_rgba.shape[-1] == 4
                Image.fromarray(masked_image_rgba).save(os.path.join(args.input_dir, 'seg_image', 'train', image_name[:-4] + '.jpg.png'))
                Image.fromarray(masked_image_rgba).save(os.path.join(args.input_dir, 'seg_image', 'test', image_name[:-4] + '.jpg.png'))
                # np.save(os.path.join(args.input_dir, 'seg', image_name[:-4]), mask)

if __name__ == '__main__':
    # main()
    
    from pathlib import Path
    dir_name = '/hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke/seg_image/images'
    for image_name in sorted(os.listdir(dir_name)):
        if image_name.endswith('.jpg'):
            img = Image.open(os.path.join(dir_name, image_name))
            img.save(Path(dir_name).parent / 'train' / (image_name[:-4] + '.jpg'))
            img.save(Path(dir_name).parent / 'test' / (image_name[:-4] + '.jpg'))
            # img = np.array(img, dtype=np.uint8)
            # seg = Image.open(Path(dir_name).parent.parent / 'seg' / (image_name[:-4] + '.jpg.png'))
            # seg = np.asarray(seg, dtype=np.uint8) * 255
            # img = np.concatenate([img, seg[..., None]], axis=-1)
            # assert img.shape[-1] == 4
            # Image.fromarray(img).save(Path(dir_name).parent / 'train' / (image_name[:-4] + '.jpg.png'))
            # Image.fromarray(img).save(Path(dir_name).parent / 'test' / (image_name[:-4] + '.jpg.png'))