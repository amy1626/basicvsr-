# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmedit.apis import multi_gpu_test, set_random_seed, single_gpu_test
from mmedit.core.distributed_wrapper import DistributedDataParallelWrapper
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester')
    parser.add_argument('--config', type=str,default='./configs/restorers/basicvsr/basicvsr_vimeo90k_bi.py',help='test config file path')#配置文件路径
    parser.add_argument('--checkpoint', type=str,default='basicvsr_reds4_20120409-0e599677.pth',help='checkpoint file')#加载模型
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
        #将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。
        #如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
    parser.add_argument('--out', type=str,default='./work_dirs/example_exp/result.pkl',help='output result pickle file')#输出的
    parser.add_argument(
        '--gpu-collect',#是否使用gpu收集结果？不懂
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--save-path',
        default='./work_dirs/result',
        type=str,
        help='path to store images and if not given, will not save image')#输出的超分结果
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],#使用并行运算的方式
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()#参数
    
    cfg = mmcv.Config.fromfile(args.config)#读取配置文件
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True#在一般场景下，只要简单地在 PyTorch 程序开头将其值设置为 True，就可以大大提升卷积神经网络的运行速度。
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':#单卡
        distributed = False
    else:
        distributed = True#多卡
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()#rank，world_size
    
    # # set random seeds
    # if args.seed is not None:
        # if rank == 0:
            # print('set random seed to', args.seed)
        # set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)#构建测试数据集

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
        **dict(
            samples_per_gpu=1,
            drop_last=False,
            shuffle=False,
            dist=distributed),
        **cfg.data.get('test_dataloader', {})
    }
    #print(loader_cfg)
    # {'workers_per_gpu': 1, 'samples_per_gpu': 1,
    # 'drop_last': False, 'shuffle': False, 'dist': False}

    data_loader = build_dataloader(dataset, **loader_cfg)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)#INFO - Use load_from_http loader
    #print('model:',model)
    args.save_image = args.save_path is not None#是否保存输出结果

    empty_cache = cfg.get('empty_cache', False)
    if not distributed:#※
        _ = load_checkpoint(model, args.checkpoint, map_location='cpu')#加载预训练权重
        model = MMDataParallel(model, device_ids=[0])#单进程多线程
        outputs = single_gpu_test(#单gpu
            model,
            data_loader,
            save_path=args.save_path,
            save_image=args.save_image)
    else:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = DistributedDataParallelWrapper(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

        device_id = torch.cuda.current_device()
        _ = load_checkpoint(
            model,
            args.checkpoint,
            map_location=lambda storage, loc: storage.cuda(device_id))
        outputs = multi_gpu_test(#多gpu
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            save_path=args.save_path,
            save_image=args.save_image,
            empty_cache=empty_cache)        

    # if rank == 0 and 'eval_result' in outputs[0]:
        # print('')
        # # print metrics
        # stats = dataset.evaluate(outputs)
        # for stat in stats:
            # print('Eval-{}: {}'.format(stat, stats[stat]))

        # # save result pickle
        # if args.out:
            # print('writing results to {}'.format(args.out))
            # mmcv.dump(outputs, args.out)

import time
if __name__ == '__main__':
    T1=time.time()
    main()
    T2=time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))