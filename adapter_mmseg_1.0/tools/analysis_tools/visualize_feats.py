# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn
from typing import Type

from argparse import ArgumentParser
from mmseg.apis import inference_model, init_model
from mmseg.visualization import SegLocalVisualizer


class Recorder:
    """record the forward output feature map and save to data_buffer."""

    def __init__(self) -> None:
        self.in_data_buffer = list()
        self.out_data_buffer = list()

    def __enter__(self, ):
        self._data_buffer = list()

    def record_data_hook(self, model: nn.Module, input: Type, output: Type):
        self.in_data_buffer.append(input)
        self.out_data_buffer.append(output)

    def __exit__(self, *args, **kwargs):
        pass


source = [
    'backbone.norm0',
    # 'backbone.norm1',
    # 'backbone.norm2',
    # 'backbone.norm3'
]
source = dict.fromkeys(source)


def visualize(args, model, recorder, result):
    seg_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='WandbVisBackend')],
        save_dir='temp_dir',
        alpha=0.5)
    seg_visualizer.dataset_meta = dict(
        classes=model.dataset_meta['classes'],
        palette=model.dataset_meta['palette'])

    image = mmcv.imread(args.img, 'color')

    seg_visualizer.add_datasample(
        name='predict',
        image=image,
        data_sample=result,
        draw_gt=False,
        draw_pred=True,
        wait_time=0,
        out_file=None,
        show=False)

    # add feature map to wandb visualizer
    for i in range(len(recorder.data_buffer)):
        feature = recorder.data_buffer[i][0]  # remove the batch
        drawn_img = seg_visualizer.draw_featmap(
            feature, image, channel_reduction='select_max')
        seg_visualizer.add_image(f'feature_map{i}', drawn_img)

    seg_visualizer.add_image('image', image)


def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    # # parser.add_argument('--out-file', default=None, help='Path to output file')
    # parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    args.device = 'cuda:0'
    args.img = 'demo/ade_val/ADE_val_00001967.jpg'
    base_root = '/home/peng/code/adapter_mmseg/'
    args.checkpoint = base_root+'work_dirs/mlp_convnext_base_512x512_160k_ade20k/iter_160000.pth'
    args.config = base_root+'work_dirs/mlp_convnext_base_512x512_160k_ade20k/mlp_convnext_base_512x512_160k_ade20k.py'
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # for name, module in model.named_modules():
    #     print(name)

    count = 0
    recorder = Recorder()
    # registry the forward hook
    for name, module in model.named_modules():
        if name in source:
            print("{} is registered forward_hook".format(name))
            count += 1
            module.register_forward_hook(recorder.record_data_hook)
            if count == len(source):
                break
    with recorder:
        result = inference_model(model, args.img)
    # visualize(args, model, recorder, result)

    # for feats in recorder.in_data_buffer:
    #     for feat in feats:
    #         print(feat.shape)


if __name__ == '__main__':
    main()
