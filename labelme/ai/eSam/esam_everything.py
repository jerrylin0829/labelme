
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import io
import cv2

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from .build_esam import build_efficient_sam_vits
GRID_SIZE = 10

from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms, box_area

class EfficientSAM_Everything:
    def __init__(self, model, grid_size=10, min_area=100, nms_thresh=0.7):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.grid_size = grid_size
        self.min_area = min_area
        self.nms_thresh = nms_thresh
        self.img = None
        
    def setImg(self, img):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    def setInferenceDev(self, num):
        self.device = torch.device(f"cuda:{num}" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device) 
         
    def setGridSize(self, grid_size):
        self.grid_size = grid_size
        
    def getGridSize(self):
        return self.grid_size
    
    def process_small_region(self, rles):
        new_masks = []
        scores = []
        for rle in rles:
            mask = rle_to_mask(rle[0])

            mask, changed = remove_small_regions(mask, self.min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, self.min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0).to(self.device))  
            scores.append(float(unchanged))

        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores).to(self.device),  
            torch.zeros_like(boxes[:, 0]).to(self.device),  
            iou_threshold=self.nms_thresh,
        )

        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0).to(self.device)  
                rles[i_mask] = mask_to_rle_pytorch(mask_torch)
        masks = [rle_to_mask(rles[i][0]) for i in keep_by_nms]
        return masks

    def get_predictions_given_embeddings_and_queries(self, img, points, point_labels):
        img = img.to(self.device)
        points = points.to(self.device)
        point_labels = point_labels.to(self.device)

        predicted_masks, predicted_iou = self.model(
            img[None, ...], points, point_labels
        )

        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou_scores = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)

        predicted_masks = torch.take_along_dim(
            predicted_masks, sorted_ids[..., None, None], dim=2
        )

        predicted_masks = predicted_masks[0]
        iou = predicted_iou_scores[0, :, 0]
        index_iou = iou > 0.7
        iou_ = iou[index_iou]
        masks = predicted_masks[index_iou]

        score = calculate_stability_score(masks, 0.0, 1.0)
        score = score[:, 0]
        index = score > 0.9
        score_ = score[index]
        masks = masks[index]
        iou_ = iou_[index]
        masks = torch.ge(masks, 0.0)
        return masks, iou_

    def run_everything(self, bbox):
        x1, y1, x2, y2 = bbox
        cropped_image = self.img[y1:y2, x1:x2]

        img_tensor = ToTensor()(cropped_image).to(self.device)
        _, original_image_h, original_image_w = img_tensor.shape

        xy = []
        for i in range(self.grid_size):
            curr_x = 0.5 + i / self.grid_size * original_image_w
            for j in range(self.grid_size):
                curr_y = 0.5 + j / self.grid_size * original_image_h
                xy.append([curr_x, curr_y])

        xy = torch.from_numpy(np.array(xy)).to(self.device)
        points = xy
        num_pts = xy.shape[0]

        point_labels = torch.ones(num_pts, 1).to(self.device)

        with torch.no_grad():
            predicted_masks, predicted_iou = self.get_predictions_given_embeddings_and_queries(
                img_tensor,
                points.reshape(1, num_pts, 1, 2),
                point_labels.reshape(1, num_pts, 1),
            )

        rle = [mask_to_rle_pytorch(m[0:1]) for m in predicted_masks]
        predicted_masks = self.process_small_region(rle)

        final_masks = []
        for mask in predicted_masks:
            full_image_mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=bool)
            full_image_mask[y1:y2, x1:x2] = mask
            final_masks.append(full_image_mask)

        return final_masks
    
    def show_anns(self, masks, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.set_autoscale_on(False)
        img = np.ones((masks[0].shape[0], masks[0].shape[1], 4))
        img[:, :, 3] = 0

        combined_mask = np.zeros((masks[0].shape[0], masks[0].shape[1]), dtype=bool)

        for ann in masks:
            combined_mask |= ann

        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[combined_mask] = color_mask

        ax.imshow(img)
        plt.show()
