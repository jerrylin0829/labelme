# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .esam import build_efficient_sam

def build_efficient_sam_vitt(batch,dev=None):
    return build_efficient_sam(
        batch = batch,
        dev=dev,
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint="labelme/ai/weights/efficient_sam_vitt.pt",
    ).eval()


def build_efficient_sam_vits(batch,dev=None):
    return build_efficient_sam(
        batch = batch,
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        dev=dev,
        checkpoint=r"labelme/ai/weights/efficient_sam_vits.pt"
    ).eval()
