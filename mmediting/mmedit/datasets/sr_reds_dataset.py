# Copyright (c) OpenMMLab. All rights reserved.
from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRREDSDataset(BaseSRDataset):
    """REDS dataset for video super resolution.

    The dataset loads several LQ (Low-Quality) frames and a center GT
    (Ground-Truth) frame. Then it applies specified transforms and finally
    returns a dict containing paired data and other information.
    数据集加载若干低分辨率帧和一个中间帧。然后应用具体的transforms，最后返回包含一对数据和其他信息的字典。

    It reads REDS keys from the txt file.
    Each line contains:
    1. image name; 2, image shape, separated by a white space.
    Examples:

        000/00000000.png (720, 1280, 3)
        000/00000001.png (720, 1280, 3)

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder. 低分辨率的路径
        gt_folder (str | :obj:`Path`): Path to a gt folder. gt的路径
        ann_file (str | :obj:`Path`): Path to the annotation file. anno文件的路径
        num_input_frames (int): Window size for input frames. 滑动窗口的大小
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio. 上采样的倍数
        val_partition (str): Validation partition mode. Choices ['official' or
        'REDS4']. Default: 'official'.验证集选择官方的还是自己设置的
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 ann_file,
                 num_input_frames,
                 pipeline,
                 scale,
                 val_partition='official',
                 test_mode=False):
        super().__init__(pipeline, scale, test_mode)
        assert num_input_frames % 2 == 1, (
            f'num_input_frames should be odd numbers, '
            f'but received {num_input_frames }.')
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.ann_file = str(ann_file)
        self.num_input_frames = num_input_frames
        self.val_partition = val_partition
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for REDS dataset.
        加载reds数据集的annno，返回低分辨率和高分辨率的图像对作为字典。
        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        # get keys
        with open(self.ann_file, 'r') as fin:
            keys = [v.strip().split('.')[0] for v in fin]

        if self.val_partition == 'REDS4':
            val_partition = ['270', '271', '272', '273']
        elif self.val_partition == 'official':#官方的验证集
            #val_partition = [f'{v:03d}' for v in range(090, 099)]
            val_partition = [f'{v:03d}' for v in range(270, 274)]
        else:
            raise ValueError(
                f'Wrong validation partition {self.val_partition}.'
                f'Supported ones are ["official", "REDS4"]')

        if self.test_mode:
            keys = [v for v in keys if v.split('/')[0] in val_partition]
        else:
            keys = [v for v in keys if v.split('/')[0] not in val_partition]

        data_infos = []
        for key in keys:
            data_infos.append(
                dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=key,
                    max_frame_num=100,  # REDS has 100 frames for each clip
                    num_input_frames=self.num_input_frames))

        return data_infos
