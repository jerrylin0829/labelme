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
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vits

GRID_SIZE = 28

from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms, box_area

def process_small_region(rles):
        new_masks = []
        scores = []
        min_area = 100
        nms_thresh = 0.7
        for rle in rles:
            mask = rle_to_mask(rle[0])

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                rles[i_mask] = mask_to_rle_pytorch(mask_torch)
        masks = [rle_to_mask(rles[i][0]) for i in keep_by_nms]
        return masks

def get_predictions_given_embeddings_and_queries(img, points, point_labels, model):
    predicted_masks, predicted_iou = model(
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

def run_everything_ours(img_path, model, bbox):
    device = torch.device(f"cuda:{6}" if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 提取矩形框内的圖像
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]

    img_tensor = ToTensor()(cropped_image).to(device)
    _, original_image_h, original_image_w = img_tensor.shape

    # 生成網格點，以每個網格點作為檢測的座標點
    xy = []
    for i in range(GRID_SIZE):
        curr_x = 0.5 + i / GRID_SIZE * original_image_w
        for j in range(GRID_SIZE):
            curr_y = 0.5 + j / GRID_SIZE * original_image_h
            xy.append([curr_x, curr_y])
    
    xy = torch.from_numpy(np.array(xy)).to(device)
    points = xy
    num_pts = xy.shape[0]

    # 創建與網格點數量相同的標籤，這裡全為 1（代表每個點都是目標點）
    point_labels = torch.ones(num_pts, 1).to(device)

    # 不計算梯度以節省内存和加速推理
    with torch.no_grad():
        # 使用預定義的函數進行預測，這裡會返回預測的遮罩和對應的 IOU 值
        predicted_masks, predicted_iou = get_predictions_given_embeddings_and_queries(
            img_tensor,
            points.reshape(1, num_pts, 1, 2),
            point_labels.reshape(1, num_pts, 1),
            model
        )

    # 將預測的遮罩轉換為 RLE 格式以便處理
    rle = [mask_to_rle_pytorch(m[0:1]) for m in predicted_masks]
    # 處理小區域，去除噪點
    predicted_masks = process_small_region(rle)

    # 將預測的遮罩重新定位到原始圖像的位置
    final_masks = []
    for mask in predicted_masks:
        full_image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        full_image_mask[y1:y2, x1:x2] = mask
        final_masks.append(full_image_mask)
    
    return final_masks

# def show_anns_ours(mask, ax):
#     ax.set_autoscale_on(False)
#     img = np.ones((mask[0].shape[0], mask[0].shape[1], 4))
#     img[:, :, 3] = 0
    
#     # 初始化一個空白的合併遮罩 (combined_mask)
#     combined_mask = np.zeros((mask[0].shape[0], mask[0].shape[1]), dtype=bool)
    
#     # 將所有的單一遮罩合併到 combined_mask
#     for ann in mask:
#         combined_mask |= ann
    
#     # 設定統一的顏色給所有的遮罩
#     color_mask = np.concatenate([np.random.random(3), [0.5]])
#     img[combined_mask] = color_mask
    
#     # 顯示合併後的遮罩
#     ax.imshow(img)

def show_anns_ours(mask, ax):
    ax.set_autoscale_on(False)
    img = np.ones((mask[0].shape[0], mask[0].shape[1], 4))
    img[:,:,3] = 0
    for ann in mask:
        m = ann
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
    ax.imshow(img)

efficient_sam_vits_model = build_efficient_sam_vits().to('cuda')
efficient_sam_vits_model.eval()

fig, ax = plt.subplots(1, 2, figsize=(30, 30))
image_path = r"figs/img.jpg" #! path
image = np.array(Image.open(image_path))
ax[0].imshow(image)
ax[0].title.set_text("Original")
ax[0].axis('off')

ax[1].imshow(image)

# 調整box的座標就可以在矩形框內做everything segmentation
bbox = (200, 200, 600, 600)
mask_efficient_sam_vits = run_everything_ours(image_path, efficient_sam_vits_model, bbox)
show_anns_ours(mask_efficient_sam_vits, ax[1])
ax[1].title.set_text("EfficientSAM")
ax[1].axis('off')


plt.show()