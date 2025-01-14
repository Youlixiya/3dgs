#fern
CUDA_VISIBLE_DEVICES=1 python train.py --source_path data/nerf_llff_data/fern --images images_4 --model_path output/llff/fern
CUDA_VISIBLE_DEVICES=1 python language3dgs_project.py --source_path data/nerf_llff_data/fern --images images_4 --model_path output/llff/fern
CUDA_VISIBLE_DEVICES=1 python mask_project.py --source_path data/nerf_llff_data/fern --images images_4 --model_path output/llff/fern
CUDA_VISIBLE_DEVICES=1 python train_latent.py --source_path data/nerf_llff_data/fern --images latents_4 --model_path output/llff/latent_fern  --gs_source output/llff/latent_fern/point_cloud/iteration_30000/point_cloud.ply
CUDA_VISIBLE_DEVICES=1 python train_copy.py --source_path data/nerf_llff_data/fern --images images_32 --model_path output/llff/fern_32
CUDA_VISIBLE_DEVICES=1 python train_mask.py --gs_source output/llff/fern/point_cloud/iteration_30000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4
# CUDA_VISIBLE_DEVICES=0 python render.py -m output/llff/fern -s data/nerf_llff_data/fern
CUDA_VISIBLE_DEVICES=1 python render.py -m output/llff/fern -s data/nerf_llff_data/fern --images images_4 --text_prompt bench
CUDA_VISIBLE_DEVICES=1 python render.py -m output/llff/fern -s data/nerf_llff_data/fern --images images_4 --text_prompt fern

CUDA_VISIBLE_DEVICES=1 python train.py --source_path data/nerf_llff_data/flower --images images_4 --model_path output/llff/flower
CUDA_VISIBLE_DEVICES=1 python mask_project.py --source_path data/nerf_llff_data/flower --images images_4 --model_path output/llff/flower
CUDA_VISIBLE_DEVICES=1 python language3dgs_project.py --source_path data/nerf_llff_data/flower --images images_4 --model_path output/llff/flower
CUDA_VISIBLE_DEVICES=1 python instance3dgs_project.py --source_path data/nerf_llff_data/flower --images images_4 --model_path output/llff/flower
CUDA_VISIBLE_DEVICES=1 python render_mask.py -m output/llff/flower -s data/nerf_llff_data/flower --images images_4 --text_prompt flower
CUDA_VISIBLE_DEVICES=1 python render_language_mask.py -m output/llff/flower -s data/nerf_llff_data/flower --images images_4 --text_prompt flower