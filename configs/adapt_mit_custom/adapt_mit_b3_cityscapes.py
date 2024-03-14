_base_ = [
    '../_base_/models/adapt_segformer_b0.py',
    '../_base_/datasets/cityscapes_768x768.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]

checkpoint = 'checkpoints/segformer_mit_b3_768x768_cityscapes-75d22848.pth'
num_classes = 19
crop_size = (768, 768)
data_preprocessor = dict(size=crop_size)
sem_norm_cfg = dict(type='LN2d', requires_grad=True, eps=1e-6)

model = dict(
    data_preprocessor = data_preprocessor,
    pretrained=None, 
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 4, 18, 3], 
        aux_ratio=[3, 3, 3, 3],
        num_aux_layers=[1, 1, 1, 1],
        sem_norm_cfg=sem_norm_cfg,
        num_classes=num_classes,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.')),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=num_classes),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

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

train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader