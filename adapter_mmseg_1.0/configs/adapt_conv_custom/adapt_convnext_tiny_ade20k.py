_base_ = [
    '../_base_/models/adapter_convnext_tiny.py',
    '../_base_/datasets/ade20k_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_180k.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth '
num_classes = 150
crop_size = (512, 512)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ConvNeXtAdapter',
        num_classes=num_classes,
        arch='tiny',
        aux_ratio=[3, 3, 3, 3],
        num_aux_layers=[1, 1, 1, 1],
        layer_scale=[1e-5, 1e-4, 1e-3, 1e-2],
        enhance=[False] * 4,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        num_classes=num_classes,
        in_channels=[96, 192, 384, 768],
        channels=256,
        dropout_ratio=0.1),)

optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 6,
        'custom_keys': {
            'head': dict(lr_mult=10.),
            'aux_layers': dict(lr_mult=7.)},
    })

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.1,
    min_lr=1e-8,
    by_epoch=False)

data = dict(samples_per_gpu=4, workers_per_gpu=4)
# # fp16 settings
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# # fp16 placeholder
# fp16 = dict()


