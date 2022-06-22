# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class BICUNet(nn.Module):
    """
    bicubic双三次下采样直接
    Args:
        channels (tuple[int]): A tuple of channel numbers for each layer
            including channels of input and output . Default: (3, 64, 32, 3).
        kernel_sizes (tuple[int]): A tuple of kernel sizes for each conv layer.
            Default: (9, 1, 5).
        upscale_factor (int): Upsampling factor. Default: 4.
    """

    def __init__(self,
                 channels=(3, 64, 32, 3),
                 kernel_sizes=(9, 1, 5),
                 upscale_factor=4):
        super().__init__()
        assert len(channels) == 4, ('The length of channel tuple should be 4, '
                                    f'but got {len(channels)}')
        assert len(kernel_sizes) == 3, (
            'The length of kernel tuple should be 3, '
            f'but got {len(kernel_sizes)}')
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        # n, t, c, h, w = lrs.size()
        # outputs = []
        # for i in range(t):
            # out = lrs[:, i, :, :, :]
            # out=self.img_upsampler(out)
            # outputs[i] = out
        # return torch.stack(outputs, dim=1)
        
        out = self.img_upsampler(x)
        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
