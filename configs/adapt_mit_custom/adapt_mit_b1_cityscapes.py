_base_ = [
    '../_base_/models/adapt_segformer_b0.py',
    '../_base_/datasets/cityscapes_768x768.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa
num_classes = 19
crop_size = (768, 768)
data_preprocessor = dict(size=crop_size)

model = dict( 
            data_preprocessor=data_preprocessor,
            pretrained=checkpoint,
            backbone=dict(
                embed_dims=64, 
                aux_ratio=[3, 3, 3, 3],
                num_aux_layers=[1, 1, 1, 1], 
                num_classes=num_classes),
            decode_head=dict(
                in_channels=[64, 128, 320, 512],
                num_classes=num_classes),
            test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
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