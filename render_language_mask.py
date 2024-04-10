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
import torch
from torch import nn
import open_clip
from scene import Scene
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Type, Tuple
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.clip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, mask_background, args, masks):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    mask_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask_renders")
    mask_image_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask_image_renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(mask_render_path, exist_ok=True)
    makedirs(mask_image_render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # mask_index = args.mask_index
    # instance_embedding = gaussians.instance_embeddings[mask_index]
    # clip_embedding = gaussians.clip_embeddings[mask_index]
    # encoder_embeddings = torch.stack([instance_embedding, clip_embedding]).unsqueeze(0)
    # triplane = gaussians.triplane_tokens[mask_index]
        # triplane = pc.triplane_upsample(triplane_lowres)
        # masks_precomp = pc.triplane_encoder(xyz)
    # masks_precomp = triplane_sample(triplane.embeddings, xyz)
    # rendered_feature = render(viewpoint_cam, gaussians, pipe, feature_bg_color, render_feature=True, encoder_hidden_states=encoder_embeddings)['render_feature']

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        rendering_masks = torch.zeros((1, rendering.shape[1], rendering.shape[2]), dtype=torch.bool, device='cuda')
        for mask in masks:
            rendering_mask = render(view, gaussians, pipeline, background, override_color=mask.unsqueeze(-1).float().repeat(1, 3))["render"]
            rendering_mask = rendering_mask[[0]]
            rendering_mask = rendering_mask > 0
            rendering_masks = rendering_mask | rendering_masks
        
        # rendering_mask = render(view, gaussians, pipeline, background, override_color=masks.float().repeat(1, 3))["render"]
        # rendering_mask = rendering_mask[[0]]
        # rendering_mask = rendering_mask > 0
        # rendering_masks = rendering_mask
        
            # rendering_masks = rendering_masks | rendering_mask
        rendered_mask_image = rendering.clone()
        rendered_mask_image[:, ~rendering_masks[0]] = rendered_mask_image[:, ~rendering_masks[0]] / 2
        rendering_masks = rendering_masks[None].float()
        # rendering_mask = rendering_mask > 0.8  # 1, H, W
        
        # rendering_mask = rendering_mask[None, None]
        # rendered_masks = (torch.norm(rendering_mask, dim=0) > 0.5)
        # rendered_masks = rendering_mask > 0.3
        # rendered_masks = []
        # for mask_index in masks_index:
        #     rendered_mask = render(view, gaussians, pipeline, mask_background, render_feature=True, triplane_index=mask_index)['render_feature']
        #     rendered_mask = rendered_mask > 0.7
        #     rendered_masks.append(rendered_mask)
        # rendered_masks = torch.cat(rendered_masks, 0)
        # rendered_masks = torch.sum(rendered_masks, dim=0) > 0
        
        # rendered_mask_image[:, ~rendering_mask] = torch.tensor([[1, 1, 1]], device='cuda', dtype=torch.float32).permute(1, 0)
        
        # rendering_mask = rendering_mask[None].float()
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering_masks, os.path.join(mask_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendered_mask_image, os.path.join(mask_image_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(3)
        scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
        # params_path = os.path.join(dataset.model_path,
        #                     "point_cloud",
        #                     "iteration_" + str(30000),
        #                     f"mask_triplane_{1000}.pt")
        gs_states_path = os.path.join(dataset.model_path,
                            "point_cloud",
                            "iteration_" + str(30000),
                            f"gs_states.pt")
        gs_states = torch.load(gs_states_path)
        weights = gs_states['weights'].cuda()
        clip_embeddings = gs_states['clip_embeddings'].cuda()

        # gaussians.load_feature_params(params_path)
        
        clip = OpenCLIPNetwork(OpenCLIPNetworkConfig())
        clip.set_positives(args.text_prompt.split(','))
        relevancy = clip.get_relevancy(clip_embeddings, 0)[..., 0]
        print(relevancy.max())
        print(relevancy.min())
        # mask = relevancy > 0.66
        # mask_index = torch.argmax(relevancy).unsqueeze(0)
        
        # if args.top_one:
        #     masks_index = torch.argmax(relevancy).unsqueeze(0)
        # else:
        masks_index = torch.nonzero(relevancy > args.mask_threshold)
        masks_index = torch.nonzero(relevancy > 0.90).squeeze(1)
        non_zero_indices = torch.nonzero(masks_index).squeeze()

        masks_index = masks_index[non_zero_indices]
        # print(masks_index.ndim)
        if masks_index.ndim == 0:
            masks_index = masks_index.unsqueeze(0)
        # masks_index = torch.argmax(relevancy).unsqueeze(0)
        print(masks_index)
        # print(masks_index.shape)
        weight = weights[:, masks_index]
        # weight = weights[:, [mask_index]]
        # print(weight.max())
        # print(weight.min())
        # print(weight.shape)
        masks = (weight > 0.90).permute(1, 0)
        # masks = (weight > 0.95)
        # print(masks.shape)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        mask_bg_color = [0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        mask_background = torch.tensor(mask_bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, mask_background, args, masks)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, mask_background, args, masks)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--mask_threshold", default=0.5, type=float)
    parser.add_argument("--top_one", default=True, type=bool)
    parser.add_argument("--text_prompt", default='flower', type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)