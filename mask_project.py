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
# from scene.clip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork
from utils.general_utils import safe_state
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from ultralytics import YOLO
                

def project(dataset):
    gaussians = GaussianModel(3)
    scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
    efficientvit_sam = create_sam_model(
        name="xl1", weight_url="ckpts/xl1.pt",
    )
    efficientvit_sam = efficientvit_sam.cuda().eval()
    efficientvit_sam.requires_grad_(False)
    efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
    yolo = YOLO('ckpts/yolov8x-worldv2.pt')  # or choose yolov8m/l-world.pt
    yolo.set_classes(["flower"])
    # registry = sam_model_registry[dataset.sam_model_type]
    # sam = registry(checkpoint=dataset.sam_model_ckpt)
    # sam = sam.to('cuda')
    # sam_mask_generator = SamAutomaticMaskGenerator(model=sam)
    viewpoint_stack = scene.getTrainCameras().copy()
    # masks = torch.zeros((gaussians._opacity.shape[0], 1), device='cuda')
    weights = torch.zeros_like(gaussians._opacity).cuda()
    weights_cnt = torch.zeros_like(gaussians._opacity, dtype=torch.int32).cuda()
    save_path = 'tmp'
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for i in tqdm(range(len(viewpoint_stack))):
            try:
                cur_cam = viewpoint_stack[i]
                image_np = (cur_cam.original_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                image_pil = Image.fromarray(image_np)
                efficientvit_sam_predictor.set_image(image_np)
                result = yolo.predict(image_pil)[0]
                box = result.boxes.xyxy.cpu().numpy()
                result.save(f'tmp/{i}.jpg')
                # image_pil = Image.fromarray(image_np)
                mask = torch.from_numpy(efficientvit_sam_predictor.predict(box=box, multimask_output=False)[0]).cuda()
                
                gaussians.apply_weights(cur_cam, weights, weights_cnt, mask.float())
                torch.cuda.empty_cache()
            except:
                pass
        weights = weights / (weights_cnt + 1e-7)


    save_path = os.path.join(scene.model_path, "point_cloud/iteration_30000", 'mask.pt')
    torch.save((weights.cpu() > 0.9), save_path)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    # parser.add_argument('--debug_from', type=int, default=-1)
    # parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    # parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    # parser.add_argument("--start_checkpoint", type=str, default = None)
    # parser.add_argument("--comp", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    
    # print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)
    
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    project(lp.extract(args))

    # All done
    # print("\nTraining complete.")