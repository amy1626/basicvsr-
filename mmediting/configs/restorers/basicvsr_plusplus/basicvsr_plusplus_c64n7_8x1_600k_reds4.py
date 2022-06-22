exp_name = 'basicvsr_plusplus_c64n7_8x1_600k_reds4'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlus',
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=5000)#迭代次数是5000
test_cfg = dict(metrics=['PSNR'], crop_border=0)
#test_cfg = dict(metrics=['PSNR','SSIM'], crop_border=0)


# dataset settings
train_dataset_type = 'SRREDSMultipleGTDataset'
val_dataset_type = 'SRREDSMultipleGTDataset'
test_dataset_type = 'SRVid4Dataset'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        #type='LoadPairedImageFromFile',
        io_backend='disk',
        key='lq',#键入结果以找到对应的路径
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        #type='LoadPairedImageFromFile',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),#Rescale images from [0, 255] to [0, 1]
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),#Flip images
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),#Random transpose h and w for images
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    #workers_per_gpu=1,#每个gpu分配的线程数 samples_per_gpu:每个gpu计算的样本数量
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),  # 8 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            # lq_folder='data/REDS/train_sharp_bicubic/X4',
            # gt_folder='data/REDS/train_sharp',

            # lq_folder='data/reds/traffic/train/trans_bicubic/X4',
            # gt_folder='data/reds/traffic/train/trans',
            
            lq_folder='data/4K/train/lr',
            gt_folder='data/4K/train/gt',

            num_input_frames=30,
            pipeline=train_pipeline,
            scale=4,
            #val_partition='test',
            val_partition='REDS4',
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        # lq_folder='data/REDS/train_sharp_bicubic/X4',
        # gt_folder='data/REDS/train_sharp',

        # lq_folder='data/reds/traffic/train/trans_bicubic/X4',
        # gt_folder='data/reds/traffic/train/trans',

        lq_folder='data/4K/valid_crop/lr',
        gt_folder='data/4K/valid_crop/gt',
        
        num_input_frames=100,
        pipeline=test_pipeline,
        scale=4,
        #val_partition='test',
        val_partition='REDS4',
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,

        lq_folder='data/test_amy/Bicubic4xLR',
        gt_folder='data/test_amy/GT',
        #ann_file='data/test_amy/meta_info_Vid4_GT.txt',
        num_input_frames=30,
        #num_input_frames=100,
        val_partition='test',
        pipeline=test_pipeline,
        scale=4,
        #val_partition='REDS4',
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=1e-4,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)})))

# learning policy
total_iters = 600000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[600000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=50000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,#打印日志的间隔
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),#用于记录训练过程的logger
        # dict(type='TensorboardLoggerHook'),#还支持 Tensorboard 记录器
    ])
visual_config = None#可视化配置，我们不使用它。

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None#从给定路径加载模型作为预训练模型。 这不会恢复训练
resume_from = 'work_dirs/basicvsr_plusplus_c64n7_8x1_600k_reds4/iter_400000.pth'
#resume_from = None#从给定路径恢复检查点，当检查点被保存时，训练将从迭代中恢复
workflow = [('train', 1)]#[('train', 1)] 表示只有一个工作流，名为'train'的工作流执行一次。
find_unused_parameters = True
