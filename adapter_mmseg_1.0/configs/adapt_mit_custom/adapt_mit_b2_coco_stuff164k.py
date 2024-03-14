_base_ = [
    '../_base_/models/adapt_segformer_b0.py',
    '../_base_/datasets/coco-stuff164k.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_80k.py'
]
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

num_classes = 171
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
sem_norm_cfg = dict(type='LN2d', requires_grad=True, eps=1e-6)

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint, 
    backbone=dict(
        embed_dims=64, 
        num_layers=[3, 4, 6, 3], 
        aux_ratio=[3, 3, 3, 3],
        num_aux_layers=[1, 1, 1, 1], 
        sem_norm_cfg=sem_norm_cfg,
        num_classes=num_classes),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=num_classes))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
            'aux_layers': dict(lr_mult=7.),
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader