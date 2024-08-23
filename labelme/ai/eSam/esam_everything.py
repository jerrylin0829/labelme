import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import io

import cv2
import threading
import sys
import os
from labelme.logger import logger
from ...widgets.msg_box import MessageBox

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from .build_esam import build_efficient_sam_vits
GRID_SIZE = 25

from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms, box_area

FILTER_MODE = ['Median','PERCENT']
msgBox = MessageBox()

class EfficientSAM_Everything:
    def __init__(
        self, model, dev, 
        grid_size = GRID_SIZE, 
        min_region_area = 200, 
        nms_thresh = 0.5, 
        fliter_mode = 0, 
        score = 0.9, 
        iou = 0.7, 
        pctUp = 95,
        pctLow = 10, 
        filter_delta = 200, 
        min_filter_area = 100
        ):
        if dev != None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{dev}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        self.grid_size = grid_size
        self.nms_thresh = nms_thresh
        self.min_region_area = min_region_area
        self.min_filter_area = min_filter_area
        self.score = score
        self.iou_thresh = iou
        self.pctUp = pctUp
        self.pctLow = pctLow
        self.filter_delta = filter_delta
        self.fliter_mode = fliter_mode
        self.img = None
        self.cropImg = None
        
        self.lock = threading.Lock()  
        self.thread = None 
        
    def wait_for_thread(self):
        if self.thread is not None:
            self.thread.join()

    def setImg(self, img):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def setInferenceDev(self, num):
        self.device = torch.device(
            f"cuda:{num}" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        
            
    def setGridSize(self, grid_size):
        self.grid_size = grid_size
        
    def setFliterMode(self,mode):
        self.fliter_mode = mode
        
    def setNMS(self,nms_thresh):
        self.nms_thresh = nms_thresh
        
    def setMinFilterArea(self,min_filter_area):
        self.min_filter_area = min_filter_area

    def setDelta(self,filter_delta):
        self.filter_delta = filter_delta
        
    def setScore(self,val):
        self.score = val
        
    def setIOU(self,val):
        self.iou_thresh = val
        
    def setPctUp(self, val):
        self.pctUp = val  
        
    def setPctLow(self, val):
        self.pctLow = val 
        
    def getGridSize(self):
        return self.grid_size
    
    def getInferenceDev(self):
        return self.device.index
    
    def getFliterMode(self):
        return self.fliter_mode
    
    def getNMS(self):
        return self.nms_thresh 
    
    def getMinFilterArea(self):
        return self.min_filter_area 
    
    def getDelta(self):
        return self.filter_delta 
    
    def getScore(self):
        return self.score
    
    def getIOU(self):   
        return self.iou_thresh   
    
    def getPctUp(self):
        return self.pctUp
        
    def getPctLow(self):
        return self.pctLow
    
    def process_small_region(self, rles):
        new_masks = []
        scores = []
        for rle in rles:
            mask = rle_to_mask(rle[0])
            mask, changed = remove_small_regions(mask, self.min_region_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, self.min_region_area, mode="islands")
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
        predicted_masks, predicted_iou = self.model(img[None, ...], points, point_labels)
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou_scores = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_masks = torch.take_along_dim(predicted_masks, sorted_ids[..., None, None], dim=2)
        predicted_masks = predicted_masks[0]
        iou = predicted_iou_scores[0, :, 0]
        index_iou = iou > self.iou_thresh
        iou_ = iou[index_iou]
        masks = predicted_masks[index_iou]
        score = calculate_stability_score(masks, 0.0, 1.0)
        score = score[:, 0]
        index = score > self.score
        masks = masks[index]
        iou_ = iou_[index]
        masks = torch.ge(masks, 0.0)
        return masks, iou_
    
    def filter_masks_by_area(self, masks):  # ! added by alvin
        mask_areas = [np.sum(mask) for mask in masks]  # todo :計算每個遮罩的面積
        sorted_mask_areas = sorted(mask_areas)
        if FILTER_MODE[self.fliter_mode] == 'Median':
            median = np.median(sorted_mask_areas)
            lower_bound = median - self.filter_delta
            upper_bound = median + self.filter_delta
            logger.info(f"mask_areas: {sorted_mask_areas}")
            
        elif FILTER_MODE[self.fliter_mode] == 'PERCENT':
            lower_bound = np.percentile(mask_areas, self.pctLow)
            upper_bound = np.percentile(mask_areas, self.pctUp)
            
        filtered_masks_with_areas = [
            (mask, area) for mask, area in zip(masks, mask_areas)
            if lower_bound <= area <= upper_bound
        ]
        
        filtered_masks, filtered_mask_areas = zip(*filtered_masks_with_areas) if filtered_masks_with_areas else ([], [])

        eliminated_mask = [
            mask for mask, area in zip(masks, mask_areas)
            if area < lower_bound or area > upper_bound
        ]

        if self.min_filter_area is not None:
            filtered_masks_with_areas = [
                (mask, area) for mask, area in zip(filtered_masks, filtered_mask_areas)
                if area > self.min_filter_area
            ]
            
            if filtered_masks_with_areas:
                filtered_masks, filtered_mask_areas = zip(*filtered_masks_with_areas)
            else:
                filtered_masks, filtered_mask_areas = [], []

        if len(filtered_masks) == 0:
            msgBox.showMessageBox(
                "warning","No masks found after filtering."
            )
            logger.warning("No masks found after filtering.")
        
        
        #logger.warning(f"Median: {median}")
        logger.warning(f"origin: { sorted([np.sum(mask) for mask in sorted_mask_areas] )}")
        logger.warning(f"filtered_masks: {sorted([np.sum(mask) for mask in filtered_masks])}")
        logger.warning(f"eliminated_mask: {[np.sum(mask) for mask in eliminated_mask]}")

        return filtered_masks, sorted_mask_areas, lower_bound, upper_bound
    
    def show_masks(self, masks, mask2=None):
        logger.info(f"total number num:{len(masks)}")
        fig, ax = plt.subplots(1, 2, figsize=(12, 12))
        ax[0].imshow(self.cropImg)
        self.show_anns(mask2, ax[0])
        ax[0].set_title("Before filter")
        ax[0].axis('off')

        ax[1].imshow(self.cropImg)
        self.show_anns(masks, ax[1])
        ax[1].set_title("After filter")
        ax[1].axis('off')
        plt.show()
    
    def show_anns(self, masks, ax):
        if len(masks) == 0:
            logger.warning("No masks to display.")
            return
        ax.set_autoscale_on(False)
        img = np.ones((masks[0].shape[0], masks[0].shape[1], 4))
        img[:, :, 3] = 0
        for ann in masks:
            m = ann
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
        ax.imshow(img)
        
    def run_everything(self, bbox):
        x1, y1, x2, y2 = bbox
        self.cropImg = self.img[y1:y2, x1:x2]
        
        img_tensor = ToTensor()(self.cropImg).to(self.device)
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
        
        filtered_masks, mask_areas, lower_bound, upper_bound = self.filter_masks_by_area(predicted_masks)

        final_masks = []
        for mask in filtered_masks:
            full_image_mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=bool)
            full_image_mask[y1:y2, x1:x2] = mask
            final_masks.append(full_image_mask)

        self.show_masks(filtered_masks, predicted_masks)
        return final_masks
    
    def run_in_background(self, bbox):
        def target_func():
            with self.lock:  
                logger.info("Running EfficientSAM in background thread.")
                self.run_everything(bbox)
        
        if self.thread is None or not self.thread.is_alive():  
            self.thread = threading.Thread(target=target_func)
            self.thread.start()
        else:
            logger.warning("Thread is already running. Please wait for it to complete.")