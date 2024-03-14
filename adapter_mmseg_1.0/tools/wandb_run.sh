wandb login c5c472fac02971379915fc188989f349182a673d; \
python tools/analysis_tools/feature_map_visual.py \
    data/cityscapes/leftImg8bit/val/munster/munster_000005_000019_leftImg8bit.png \
    configs/segformer/segformer_mit-b0_8x1_768x768_160k_cityscapes.py \
    checkpoints/segformer_mit-b0_768x768_cityscapes-43099604.pth \
    --out-file visual_results/test.png

# --gt-mask data/cityscapes/gtFine/val/munster/munster_000005_000019_gtFine_labelTrainIds.png \