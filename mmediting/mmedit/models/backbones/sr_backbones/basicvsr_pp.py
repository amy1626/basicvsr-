# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.runner import load_checkpoint

from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv, SPyNet)
from mmedit.models.common import PixelShufflePack, flow_warp
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class BasicVSRPlusPlus(nn.Module):
    """BasicVSR++ network structure.

    Support either x4 upsampling or same size output. Since DCN is used in this
    model, it can only be used with CUDA enabled. If CUDA is not enabled,
    feature alignment will be skipped.

    Paper:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation
        and Alignment

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_pretrained=None,
                 cpu_cache_length=100):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    2 * mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deform_groups=16,
                    max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn(
                'Deformable alignment module is not added. '
                'Probably your CUDA is not configured correctly. DCN can only '
                'be used with CUDA enabled. Alignment is skipped now.')

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.
        ?????????????????????
        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.
        ??????SPyNet????????????????????????????????????
        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.
        ????????????????????????flows_forward???????????????????????????flows_backward????????????
        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).??????

        Return:????????????flows_backward,flows_forward???
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.
            ????????????????????????  ????????????

        Args:
            feats dict(list[tensor]): Features from previous branches. Each ?????????????????????
                component is a list of tensors with shape (n, c, h, w). ????????????????????????????????????????????????
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).   ??????
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.???????????????????????????

        Return:
            dict(list[tensor]): A dictionary containing all the propagated  ?????????????????????????????????
                features. Each key in the dictionary corresponds to a   ??????????????????key???????????????????????????????????????????????????
                propagation branch, which is represented by a list of tensors.
        """
        #print(module_name)#backward_1   forward_1   backward_2  forward_2

        n, t, _, h, w = flows.size()#???????????????size
        #print('flows:',flows.size())#torch.Size([1, 29, 2, 180, 320])
        frame_idx = range(0, t + 1) #????????????
        flow_idx = range(-1, t)     #????????????
        #print('frame_idx:',frame_idx)#frame_idx: range(0, 30)
        #print('flow_idx:',flow_idx)#flow_idx: range(-1, 29)
        mapping_idx = list(range(0, len(feats['spatial'])))#mapping??????????????????
        #print('mapping_idx1:',mapping_idx)#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        mapping_idx += mapping_idx[::-1]
        #print('mapping_idx2:',mapping_idx)#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx        #???29???-1???????????????-1?????????????????????-1
            #print('frame_idx:',frame_idx)#frame_idx: range(29, -1, -1)
            #print('flow_idx:',flow_idx)#flow_idx: range(29, -1, -1)

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)#????????????????????????0
        #print('feat_prop.size():',feat_prop.size())#torch.Size([1, 64, 180, 320])
        for i, idx in enumerate(frame_idx):
            #print('i:',i)       #0  1  ?????? 28    29 0 1 ?????? 28 29    0  1  ?????? 28 29   0 1 ?????? 28 29
            #print('idx:',idx)   #29 28 ?????? 1     0  0 1 ?????? 28 29    29 28 ?????? 1  0    0 1 ?????? 28 29
            feat_current = feats['spatial'][mapping_idx[idx]]#????????????????????????
            #(feat_current.size())#torch.Size([1, 64, 180, 320])
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:#??????????????????
                flow_n1 = flows[:, flow_idx[i], :, :, :]#??????????????????????????????
                #print('flow_n1.size():',flow_n1.size())#torch.Size([1, 2, 180, 320])
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))#??????
                #print('cond_n1.size():',cond_n1.size())#torch.Size([1, 64, 180, 320])

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features    
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2,
                                                  flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                #print('cond.size():',cond.size())#torch.Size([1, 192, 180, 320])
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                #print('feat_prop1:',feat_prop.size())#feat_prop1: torch.Size([1, 128, 180, 320])
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)
                #print('feat_prop2:',feat_prop.size())#feat_prop2: torch.Size([1, 64, 180, 320])


            '''for k in feats:
                print(k)
            #spatial    backward_1
            #spatial    backward_1  forward_1
            #spatial    backward_1  forward_1   backward_2
            #spatial    backward_1  forward_1   backward_2  forward_2   '''
            
            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]#???????????????a?????????+a???
            ] + [feat_prop]
            #print('feat.size():',feat.size())#'list' object
            if self.cpu_cache:  
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            
            #print('feat_prop',feat_prop.size())#torch.Size([1, 64, 180, 320])
            feat_prop = feat_prop + self.backbone[module_name](feat)
            #print('feat_prop',feat_prop.size())#torch.Size([1, 64, 180, 320])
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        outputs = []
        num_outputs = len(feats['spatial'])
        #print('num_outputs:',num_outputs)#30
        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            #pop() ??????????????????   ??????   ?????????????????????????????????????????????????????????????????????????????????
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            #insert()?????????????????????????????????  ??????  ??????????????????
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()
            #print('hr.size:',hr.size())#torch.Size([1, 3, 720, 1280])

            outputs.append(hr)
        #print('12345',(torch.stack(outputs, dim=1)).size())#torch.Size([1, 30, 3, 720, 1280])
        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lqs.size()#???????????????size
        #print(lqs.size())   #torch.Size([1, 30, 3, 180, 320])

        # whether to cache the features in CPU  cpu??????????????????  30<100?????????false
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:#?????????????????????????????? 
            lqs_downsample = lqs.clone()
        else:#?????????????????????
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25,
                mode='bicubic').view(n, t, c, h // 4, w // 4)
        #print(lqs_downsample.size())   #torch.Size([1, 30, 3, 180, 320])      

        # check whether the input is an extended sequence   ?????????????????????
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features  ??????????????????
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))#??????????????????????????????
            #print('feats_.size():',feats_.size())#torch.Size([30, 64, 180, 320])
            h, w = feats_.shape[2:]#?????? ??????????????? ??? ??? ???
            feats_ = feats_.view(n, t, -1, h, w)
            #print('feats_.size():',feats_.size())#torch.Size([1, 30, 64, 180, 320])
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]
            #print('len(feats):',len(feats))#len(feats): 1
            #print('len(feats[spatial]):',len(feats['spatial']))#len(feats[spatial]): 30
            #?????????????????????????????????feats['spatial']???t????????????????????????

        # compute optical flow using the low-res inputs ???????????????????????????????????????????????????
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)
        #print('flows_forward',flows_forward.size())#torch.Size([1, 29, 2, 180, 320])
        #print('flows_backward',flows_backward.size())#torch.Size([1, 29, 2, 180, 320])
        
        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                #print(module)#  backward_1    forward_1   backward_2  forward_2


                feats[module] = []
                #print(feats[module])#[]

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)#?????????feats???????????????feats['backward']
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        #print('flow_1.size():',flow_1.size())#torch.Size([1, 2, 180, 320])
        #print('flow_2.size():',flow_2.size())#torch.Size([1, 2, 180, 320])
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        #print(extra_feat.size())    #torch.Size([1, 196, 180, 320])
        out = self.conv_offset(extra_feat)
        #print(out.size())   #torch.Size([1, 432, 180, 320])
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        # print(o1.size())    #torch.Size([1, 144, 180, 320])
        # print(o2.size())    #torch.Size([1, 144, 180, 320])
        # print(mask.size())  #torch.Size([1, 144, 180, 320])

        '''torch.chunk(input, chunks, dim = 0) 
        ???????????????????????????input????????????????????????dim????????????????????????????????????????????????chunks??????
        ????????????????????????????????????????????????chunks???int???- ??????????????????????????????'''

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        #print('offset.size()???',offset.size())#torch.Size([1, 288, 180, 320])
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        #print('offset1.size()???',offset_1.size())#torch.Size([1, 144, 180, 320])
        #print('offset2.size()???',offset_2.size())#torch.Size([1, 144, 180, 320])

        # #??????offset???????????????
        # import mmcv
        # from mmedit.core import tensor2img
        # mmcv.imwrite(tensor2img(offset),'/data/dataset1/wj/mmediting/work_dirs/offset.png')

        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        #print('offset1.size()???',offset_1.size())#torch.Size([1, 144, 180, 320])
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        #print('offset2.size()???',offset_2.size())#torch.Size([1, 144, 180, 320])
        offset = torch.cat([offset_1, offset_2], dim=1)
        #print(offset.size())    #torch.Size([1, 288, 180, 320])
        # mask
        mask = torch.sigmoid(mask)
        #print(mask.size())      #torch.Size([1, 144, 180, 320])
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
