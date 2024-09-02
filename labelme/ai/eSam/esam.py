import os
import math
from typing import Any, List, Tuple, Type

import torch
import torch.nn.functional as F

from torch import nn, Tensor
from labelme.logger import logger
from .esam_decoder import MaskDecoder, PromptEncoder
from .esam_encoder import ImageEncoderViT
from .two_way_transformer import TwoWayAttentionBlock, TwoWayTransformer
from ...widgets.msg_box import MessageBox

from ...widgets.qdialog_generator import CustomDialog
from PyQt5.QtWidgets import QDialog
from  ...widgets.text_inputQDialog import StringInputDialog

BATCH = None
class EfficientSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        batch,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        decoder_max_num_input_points: int,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [0.485, 0.456, 0.406],
        pixel_std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.decoder_max_num_input_points = decoder_max_num_input_points
        self.mask_decoder = mask_decoder
        self.batch = batch
        batch = batch
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False
        )

    @torch.jit.export
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        multimask_output: bool,
        input_h: int,
        input_w: int,
        output_h: int = -1,
        output_w: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks given image embeddings and prompts. This only runs the decoder.

        Arguments:
          image_embeddings: A tensor of shape [B, C, H, W] or [B*max_num_queries, C, H, W]
          batched_points: A tensor of shape [B, max_num_queries, num_pts, 2]
          batched_point_labels: A tensor of shape [B, max_num_queries, num_pts]
        Returns:
          A tuple of two tensors:
            low_res_mask: A tensor of shape [B, max_num_queries, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """
        
        batch_size, max_num_queries, num_pts, _ = batched_points.shape
        print(f"============== batch_size[predict_masks] {batch_size} ==============")
        print(f"============== max_num_queries[predict_masks] {max_num_queries} ==============")
        print(f"============== num_pts[predict_masks] {num_pts} ==============")
        num_pts = batched_points.shape[2]
        rescaled_batched_points = self.get_rescaled_pts(batched_points, input_h, input_w)

        if num_pts > self.decoder_max_num_input_points:
            rescaled_batched_points = rescaled_batched_points[
                :, :, : self.decoder_max_num_input_points, :
            ]
            batched_point_labels = batched_point_labels[
                :, :, : self.decoder_max_num_input_points
            ]
        elif num_pts < self.decoder_max_num_input_points:
            rescaled_batched_points = F.pad(
                rescaled_batched_points,
                (0, 0, 0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )
            batched_point_labels = F.pad(
                batched_point_labels,
                (0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )

        sparse_embeddings = self.prompt_encoder(
            rescaled_batched_points.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points, 2
            ),
            batched_point_labels.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points
            ),
        )
        sparse_embeddings = sparse_embeddings.view(
            batch_size,
            max_num_queries,
            sparse_embeddings.shape[1],
            sparse_embeddings.shape[2],
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings,
            self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=multimask_output,
        )
        _, num_predictions, low_res_size, _ = low_res_masks.shape
        
        torch.cuda.empty_cache()
        
        if output_w > 0 and output_h > 0:
            output_masks = F.interpolate(
                low_res_masks, (output_h, output_w), mode="bicubic"
            )
            output_masks = torch.reshape(
                output_masks,
                (batch_size, max_num_queries, num_predictions, output_h, output_w),
            )
        else:
            output_masks = torch.reshape(
                low_res_masks,
                (
                    batch_size,
                    max_num_queries,
                    num_predictions,
                    low_res_size,
                    low_res_size,
                ),
            )
        iou_predictions = torch.reshape(
            iou_predictions, (batch_size, max_num_queries, num_predictions)
        )
        return output_masks, iou_predictions
    def setBatchQuery(self,val):
        self.grid = val
        
    def getBatchQuery(self):
        return self.grid   
        
    def get_rescaled_pts(self, batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * self.image_encoder.img_size / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * self.image_encoder.img_size / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )

    @torch.jit.export
    def get_image_embeddings(self, batched_images) -> torch.Tensor:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
        Returns:
          List of image embeddings each of of shape [B, C(i), H(i), W(i)].
          The last embedding corresponds to the final layer.
        """
        batched_images = self.preprocess(batched_images)
        return self.image_encoder(batched_images)

    def forward(
        self,
        batched_images: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        scale_to_original_image_size: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
          batched_points: A tensor of shape [B, num_queries, max_num_pts, 2]
          batched_point_labels: A tensor of shape [B, num_queries, max_num_pts]
          max_queries_per_batch: Maximum number of queries per batch

        Returns:
          A list tuples of two tensors where the ith element is by considering the first i+1 points.
            low_res_mask: A tensor of shape [B, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """
        max_queries_per_batch = self.batch
        
        batch_size, num_queries, max_num_pts, _ = batched_points.shape
        image_embeddings = self.get_image_embeddings(batched_images)
        

        all_masks = []
        all_ious = []

        for i in range(0, num_queries, max_queries_per_batch):
  
            batch_points = batched_points[:, i:i+max_queries_per_batch, :, :]
            batch_labels = batched_point_labels[:, i:i+max_queries_per_batch, :]

            masks, iou_predictions = self.predict_masks(
                image_embeddings,
                batch_points,
                batch_labels,
                multimask_output=True,
                input_h=batched_images.shape[2],
                input_w=batched_images.shape[3],
                output_h=batched_images.shape[2] if scale_to_original_image_size else -1,
                output_w=batched_images.shape[3] if scale_to_original_image_size else -1,
            )

            all_masks.append(masks)
            all_ious.append(iou_predictions)
            torch.cuda.empty_cache()
            
        all_masks = torch.cat(all_masks, dim=1)  
        all_ious = torch.cat(all_ious, dim=1)
        
        torch.cuda.empty_cache()
        return all_masks, all_ious

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if (
            x.shape[2] != self.image_encoder.img_size
            or x.shape[3] != self.image_encoder.img_size
        ):
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
            )
        return (x - self.pixel_mean) / self.pixel_std

def request_checkpoint_path():
    msg_box = MessageBox()
    filepthBox = StringInputDialog(title='File path') 
    result = filepthBox.exec_()
    
    if result == QDialog.Accepted:
        try:
            file_path = filepthBox.get_user_input()
            return file_path
        except AttributeError as e:
            msg_box.showMessageBox("Error", f"Error retrieving file path: {e}")
            logger.fatal(f"Error: {e}")
            return None
    else:
        logger.info("User cancelled the input.")
        return None


def check_cuda_and_device(dev: int = None):

    msgbox = MessageBox()
    
    if not torch.cuda.is_available():
        msgbox.showMessageBox("Error", "CUDA is not available on this system.")
        raise RuntimeError("CUDA is not available on this system.")
    device_count = torch.cuda.device_count()
    if dev is not None and (not isinstance(dev, int) or dev < 0 or dev >= device_count):
        msgbox.showMessageBox(
            "Error", f"Invalid CUDA device index: {dev}. Available devices: {device_count}"
        )
        raise ValueError(f"Invalid CUDA device index: {dev}. Available devices: {device_count}")

def load_model_checkpoint(sam: nn.Module, checkpoint: str = None, dev: int = None):
    msgbox = MessageBox()

    while checkpoint is None or not os.path.exists(checkpoint) or os.path.isdir(checkpoint):
        if checkpoint and os.path.isdir(checkpoint):
            msgbox.showMessageBox("Error", f"Provided path is a directory, not a file: {checkpoint}")
        elif checkpoint is None or not os.path.exists(checkpoint):
            msgbox.showMessageBox("Error", f"Checkpoint file '{checkpoint}' does not exist." if checkpoint else "No checkpoint file specified.")

        checkpoint = request_checkpoint_path()

        if checkpoint is None:
            msgbox.showMessageBox("Error", "No valid checkpoint file provided or user cancelled input.")
            raise FileNotFoundError("No valid checkpoint file provided or user cancelled input.")

    try:
        map_location = f"cuda:{dev}" if dev is not None else "cuda:0"
        state_dict = torch.load(
                checkpoint,
                map_location=map_location,
                weights_only=True
            )
        print(f"Loaded state_dict keys: {list(state_dict.keys())}")  
    except Exception as e:
        msgbox.showMessageBox("Error", f"Error loading checkpoint: {e}")
        raise RuntimeError(f"Error loading checkpoint: {e}")

    if "model" in state_dict:
        try:
            sam.load_state_dict(state_dict["model"])
        except Exception as e:
            msgbox.showMessageBox("Error", f"Error loading model state_dict: {e}")
            raise RuntimeError(f"Error loading model state_dict: {e}")
    else:
        try:
            sam.load_state_dict(state_dict)  
        except Exception as e:
            msgbox.showMessageBox("Error", f"Error loading state_dict: {e}")
            raise RuntimeError(f"Error loading state_dict: {e}")
   
def build_efficient_sam(dev, encoder_patch_embed_dim, encoder_num_heads, batch, checkpoint=None):
    img_size = 1024
    encoder_patch_size = 16
    encoder_depth = 12
    encoder_mlp_ratio = 4.0
    encoder_neck_dims = [256, 256]
    decoder_max_num_input_points = 6
    decoder_transformer_depth = 2
    decoder_transformer_mlp_dim = 2048
    decoder_num_heads = 8
    decoder_upscaling_layer_dims = [64, 32]
    num_multimask_outputs = 3
    iou_head_depth = 3
    iou_head_hidden_dim = 256
    activation = "gelu"
    normalization_type = "layer_norm"
    normalize_before_activation = False

    activation_fn = nn.GELU if activation == "gelu" else nn.ReLU

    image_encoder = ImageEncoderViT(
        img_size=img_size,
        patch_size=encoder_patch_size,
        in_chans=3,
        patch_embed_dim=encoder_patch_embed_dim,
        normalization_type=normalization_type,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=encoder_mlp_ratio,
        neck_dims=encoder_neck_dims,
        act_layer=activation_fn,
    )

    image_embedding_size = image_encoder.image_embedding_size
    encoder_transformer_output_dim = image_encoder.transformer_output_dim

    sam = EfficientSam(
        batch=batch,
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=encoder_transformer_output_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
        ),
        decoder_max_num_input_points=decoder_max_num_input_points,
        mask_decoder=MaskDecoder(
            transformer_dim=encoder_transformer_output_dim,
            transformer=TwoWayTransformer(
                depth=decoder_transformer_depth,
                embedding_dim=encoder_transformer_output_dim,
                num_heads=decoder_num_heads,
                mlp_dim=decoder_transformer_mlp_dim,
                activation=activation_fn,
                normalize_before_activation=normalize_before_activation,
            ),
            num_multimask_outputs=num_multimask_outputs,
            activation=activation_fn,
            normalization_type=normalization_type,
            normalize_before_activation=normalize_before_activation,
            iou_head_depth=iou_head_depth - 1,
            iou_head_hidden_dim=iou_head_hidden_dim,
            upscaling_layer_dims=decoder_upscaling_layer_dims,
        ),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
    )

    check_cuda_and_device(dev)
    load_model_checkpoint(sam, checkpoint, dev)
    torch.cuda.empty_cache()
    return sam
