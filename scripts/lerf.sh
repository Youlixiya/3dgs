#waldo_kitchen
CUDA_VISIBLE_DEVICES=1 python train.py --source_path data/lerf_data/waldo_kitchen --images images --model_path output/lerf/waldo_kitchen
CUDA_VISIBLE_DEVICES=1 python instance3dgs_project.py --source_path data/lerf_data/waldo_kitchen --images images --model_path output/lerf/waldo_kitchen

CUDA_VISIBLE_DEVICES=1 python render_mask.py -m output/llff/flower -s data/nerf_llff_data/flower --images images_4 --text_prompt flower
CUDA_VISIBLE_DEVICES=1 python render_language_mask.py -m output/lerf/waldo_kitchen -s data/lerf_data/waldo_kitchen --images images --text_prompt 'waldo'

#teatime
CUDA_VISIBLE_DEVICES=3 python train.py --source_path data/lerf_data/teatime --images images --model_path output/lerf/teatime
CUDA_VISIBLE_DEVICES=3 python instance3dgs_project.py --source_path data/lerf_data/teatime --images images --model_path output/lerf/teatime

CUDA_VISIBLE_DEVICES=1 python render_mask.py -m output/llff/flower -s data/nerf_llff_data/flower --images images_4 --text_prompt flower
CUDA_VISIBLE_DEVICES=3 python render_language_mask.py -m output/lerf/teatime -s data/lerf_data/teatime --images images --text_prompt 'bear'