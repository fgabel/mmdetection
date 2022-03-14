wandb_project = 'mmdetection'
wandb_experiment_name = 'detection: staircase: yolox_tiny_default'

_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

img_scale = (416, 416)
batch_size = 8
# model settings

init_cfg = dict(type='Kaiming',
                layer='Conv2d',
                a=2.23606797749979,  # sqrt(5)
                distribution='uniform',
                mode='fan_in',
                nonlinearity='leaky_relu')
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=1000000,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.375, init_cfg=init_cfg),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        init_cfg=init_cfg),
    bbox_head=dict(
        type='YOLOXHead', num_classes=1, in_channels=96, feat_channels=96, init_cfg=init_cfg),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
data_root = 'data/'
dataset_type = 'CocoDataset'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(
    #    type='MixUp',
    #    img_scale=img_scale,
    #    ratio_range=(0.8, 1.6),
    #    pad_val=114.0),
    dict(type='PhotoMetricDistortion', brightness_delta = 10, contrast_range = (0.9, 1.1), saturation_range=(1.0, 1.0), hue_delta = 0),
    dict(
        type='AutoAugment',
        policies=[[{
            'type': 'BrightnessTransform',
            'prob': 0.5,
            'level': 3
        }],
                  [{
                      'type': 'Translate',
                      'prob': 0.5,
                      'level': 2,
                      'img_fill_val': 255
                  }]]),
    dict(type='RandomFlip', direction = ["horizontal", "vertical", "diagonal"], flip_ratio=0.5),
    #dict(type='RandomAffine', scaling_ratio_range=(0.95, 1.05), border_val=(255, 255, 255), max_shear_degree=0),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_coco.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline,
        filter_empty_gt=True,
    )

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    persistent_workers=True,
    train=train_dataset, 
    classes=('Staircase'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_coco.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test_coco.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.0002,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

#max_epochs = 10
num_last_epochs = 50000
resume_from = None
load_from = None#'work_dirs/togal_yolox_s_8x8_300e_coco_staircase/latest.pth'
interval = 3000

# learning policy
lr_config = dict(
    _delete_=True,
    policy='poly',
    power=0.99997,
    by_epoch=False,
    min_lr=3e-5)

runner = dict(type='IterBasedRunner', max_iters=50000)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=2000,
    #dynamic_intervals=[(max_iters - num_last_epochs, 1)],
    metric='bbox')
log_config = dict(interval=100,
                  hooks=[
                      dict(type='WandbLoggerHook',
                           init_kwargs=dict(
                               project=wandb_project,
                               name=wandb_experiment_name)),
                            dict(type='TextLoggerHook'),

                           ]
                      )
