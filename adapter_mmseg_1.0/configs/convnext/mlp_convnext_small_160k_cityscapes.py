_base_ = [
    '../_base_/models/mlp_convnext_tiny.py',
    '../_base_/datasets/cityscapes_768x768_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'
num_classes = 19
crop_size = (768, 768)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ConvNeXt',
        arch='small',
        drop_path_rate=0.3,
        layer_scale_init_value=1.0,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        num_classes=num_classes,
        in_channels=[96, 192, 384, 768],
        channels=256),
    auxiliary_head=dict(in_channels=384, num_classes=num_classes),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader