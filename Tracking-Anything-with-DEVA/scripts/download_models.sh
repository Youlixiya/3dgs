wget -P ./saves/ https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth
# wget -P ./saves/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -P ./saves/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# wget -P ./saves/ https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/mobile_sam.pt
# wget -P ./saves/ https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/GroundingDINO_SwinT_OGC.py


python demo/demo_automatic.py --chunk_size 4 --img_path lerf_data/teatime/images --amp --temporal_setting semionline --size 480 --output lerf_data/teatime/masks  --suppress_small_objects