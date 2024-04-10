#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.clip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork
from utils.general_utils import safe_state
from utils.camera_utils import unproject
from utils.mask_utils import MaskDataset
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils.image_utils import psnr

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator
                

def project(dataset, pipe):
    print(dataset.source_path)
    masks_path = os.path.join(dataset.source_path, dataset.images.replace('image', 'mask'), 'Annotations')
    gaussians = GaussianModel(3)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
    mask_dataset = MaskDataset(dataset.source_path, scene.getTrainCameras().copy(), mask_dir=dataset.images.replace('image', 'mask'))
    clip_embeddings = mask_dataset.clip_embeddings
    instance_num = len(mask_dataset.instance_colors)
    print(f'instance num: {instance_num}')
    weights = torch.zeros((gaussians._opacity.shape[0], instance_num)).cuda()
    weights_cnt = torch.zeros((gaussians._opacity.shape[0], instance_num), dtype=torch.int32).cuda()
    viewpoint_stack = scene.getTrainCameras().copy()
    with torch.no_grad():
        for i in tqdm(range(len(viewpoint_stack))):
            cur_cam = viewpoint_stack[i]
            masks = mask_dataset[i]
            mask_index = torch.unique(masks)
            for index in mask_index:
                mask = masks == index
                # print(index)
                # print((torch.cat([mask]*3, dim=-1).cpu().numpy()* 255).astype(np.uint8).shape)
                # Image.fromarray((torch.cat([mask]*3, dim=-1).cpu().numpy()* 255).astype(np.uint8)).save('1.jpg')
                mask = mask.permute(2, 0, 1)
                # print(mask)
                weight = torch.zeros_like(gaussians._opacity)
                weight_cnt = torch.zeros_like(gaussians._opacity, dtype=torch.int32)
                gaussians.apply_weights(cur_cam, weight, weight_cnt, mask.float())

                # print(weight.shape)
                # print(weights.shape)
                weights[:, index] += weight.squeeze(-1)
                # print(weights[:, i])
                # print(weight)
                weights_cnt[:, index] += weight_cnt.squeeze(-1)
        weights /= weights_cnt + 1e-7
        print(weights.max())

    save_states = {'weights': weights.cpu(), 'clip_embeddings': clip_embeddings}
    save_path = os.path.join(scene.model_path, "point_cloud/iteration_30000", 'gs_states.pt')
    torch.save(save_states, save_path)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])  
    project(lp.extract(args), pp.extract(args))