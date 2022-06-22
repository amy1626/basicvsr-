# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.runner import load_checkpoint
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from torch.nn.modules.utils import _pair

from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv, SPyNet)
from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  flow_warp,make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger

@BACKBONES.register_module()
class TVSRNet(nn.Module):
    
    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_pretrained=None,
                 cpu_cache_length=100,
                 num_frames=5,
                 center_frame_idx=2,
                 with_tsa=True):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        
        # optical flow
        self.spynet = SPyNet(pretrained=spynet_pretrained)


        # feature extraction module
        #特征提取模块:输入lr进行特征提取，再进行对齐、融合、上采样等等操作
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
        modules = ['backward', 'forward']
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

        # fusion
        if self.with_tsa:
            self.fusion = TemporalAttentionFusion(
                mid_channels=mid_channels,
                num_frames=num_frames,
                center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frames * mid_channels, mid_channels, 1,


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

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn(
                'Deformable alignment module is not added. '
                'Probably your CUDA is not configured correctly. DCN can only '
                'be used with CUDA enabled. Alignment is skipped now.')

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.
            在整个序列中传播  潜在特征

        Args:
            feats dict(list[tensor]): Features from previous branches. Each 以前分支的特征
                component is a list of tensors with shape (n, c, h, w). 每个组件都是一个张量列表，形状为
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).   光流
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.传播分支的模块名称

        Return:
            dict(list[tensor]): A dictionary containing all the propagated  包含所有传播特征的字典
                features. Each key in the dictionary corresponds to a   字典中的每个key对应一个传播分支，由张量列表表示。
                propagation branch, which is represented by a list of tensors.
        """
        #print(module_name)#backward_1   forward_1   backward_2  forward_2

        n, t, _, h, w = flows.size()#获取光流的size
        #print('flows:',flows.size())#torch.Size([1, 29, 2, 180, 320])
        frame_idx = range(0, t + 1) #帧的下标
        flow_idx = range(-1, t)     #流的下标
        #print('frame_idx:',frame_idx)#frame_idx: range(0, 30)
        #print('flow_idx:',flow_idx)#flow_idx: range(-1, 29)
        mapping_idx = list(range(0, len(feats['spatial'])))#mapping的索引、下标
        #print('mapping_idx1:',mapping_idx)#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        mapping_idx += mapping_idx[::-1]
        #print('mapping_idx2:',mapping_idx)#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx        #从29到-1（但不包括-1），每次步长为-1
            #print('frame_idx:',frame_idx)#frame_idx: range(29, -1, -1)
            #print('flow_idx:',flow_idx)#flow_idx: range(29, -1, -1)

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)#传播特征，张量为0
        #print('feat_prop.size():',feat_prop.size())#torch.Size([1, 64, 180, 320])
        for i, idx in enumerate(frame_idx):
            #print('i:',i)       #0  1  …… 28    29 0 1 …… 28 29    0  1  …… 28 29   0 1 …… 28 29
            #print('idx:',idx)   #29 28 …… 1     0  0 1 …… 28 29    29 28 …… 1  0    0 1 …… 28 29
            feat_current = feats['spatial'][mapping_idx[idx]]#当前帧的空间特征
            #(feat_current.size())#torch.Size([1, 64, 180, 320])
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:#包含对齐模块
                flow_n1 = flows[:, flow_idx[i], :, :, :]#流序列中的其中一个流
                #print('flow_n1.size():',flow_n1.size())#torch.Size([1, 2, 180, 320])
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))#扭曲
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
                for k in feats if k not in ['spatial', module_name]#当模块名为a时，不+a；
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
            #pop() 函数用于移除   列表   中的一个元素（默认最后一个元素），并且返回该元素的值。
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            #insert()函数用于将指定对象插入  列表  的指定位置。
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

    def forward(self,lqs):

        n, t, c, h, w = lqs.size()#输入的低分辨率的图像

        # whether to cache the features in CPU
        # 是否缓存特征在cpu内，如果t>100，cpu_cache就为true
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25,
                mode='bicubic').view(n, t, c, h // 4, w // 4)


        feats = {}
        # compute spatial features计算序列帧的特征
        if self.cpu_cache:#在cpu有缓存
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]
            #print('len(feats):',len(feats))

        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')#低分辨率帧的宽高必须大于64像素
        ##前向传播和后向传播的spynet对齐后的特征，也就是光流引导可变形卷积的预对齐特征
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)


        # feature propgation

        for direction in ['backward', 'forward']:
            module = f'{direction}_{iter_}'
            #print(module)#  backward_1    forward_1   backward_2  forward_2


            feats[module] = []
            print(feats[module])#[]

            if direction == 'backward':
                flows = flows_backward
            else:
                flows = flows_forward

            feats = self.propagate(feats, flows, module)#这里的feats应该指的是feats['backward']
            if self.cpu_cache:
                del flows
                torch.cuda.empty_cache()


        if self.with_tsa:
            feats = self.fusion(feats)
        else:
            aligned_feat = feats.view(n, -1, h, w)
            feats = self.fusion(feats)

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
            if self.with_tsa:
                for module in [
                        self.fusion.feat_fusion, self.fusion.spatial_attn1,
                        self.fusion.spatial_attn2, self.fusion.spatial_attn3,
                        self.fusion.spatial_attn4, self.fusion.spatial_attn_l1,
                        self.fusion.spatial_attn_l2,
                        self.fusion.spatial_attn_l3,
                        self.fusion.spatial_attn_add1
                ]:
                    kaiming_init(
                        module.conv,
                        a=0.1,
                        mode='fan_out',
                        nonlinearity='leaky_relu',
                        bias=0,
                        distribution='uniform')
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class SecondOrderDeformableAlignment(nn.Module):

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
        #x:lr feature extra_feat:previous feature
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)#warp
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)#得到了Co，Cm

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)#DCN offests

        # mask
        mask = torch.sigmoid(mask)#DCN mask

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class TemporalAttentionFusion(nn.Module):

    def __init__(self,
                 mid_channels=64,
                 num_frames=5,
                 center_frame_idx=2,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super().__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.temporal_attn2 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.feat_fusion = ConvModule(
            num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)


        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)


    def forward(self, aligned_feat):
        """Forward function for TSAFusion.

        Args:
            aligned_feat (Tensor): Aligned features with shape (n, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (n, c, h, w).
        """
        n, t, c, h, w = aligned_feat.size()
        #print(aligned_feat.size())#torch.Size([1, 5, 64, 180, 320])

        # temporal attention
        embedding_ref = self.temporal_attn1(#参考帧是第二帧
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        #print(embedding_ref.size())    #torch.Size([1, 64, 180, 320])
        emb = self.temporal_attn2(aligned_feat.view(-1, c, h, w))#aligned_feat.view(-1, c, h, w)=(n*t,c,h,w)
        #print(emb.size())  #torch.Size([5, 64, 180, 320])
        emb = emb.view(n, t, -1, h, w)  # (n, t, c, h, w)     -1是需要估算的
        #print(emb.size())  #torch.Size([1, 5, 64, 180, 320])

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = emb[:, i, :, :, :]
            #print(emb_neighbor.size())#torch.Size([1, 64, 180, 320])
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (n, h, w)#点乘，对应位置的元素相乘
            #print((emb_neighbor * embedding_ref).size())#torch.Size([1, 64, 180, 320])
            #print(corr.size())#torch.Size([1, 180, 320])
            corr_l.append(corr.unsqueeze(1))  # (n, 1, h, w)   unsqueeze升维第i维度

        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (n, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(n, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(n, -1, h, w)  # (n, t*c, h, w)
        #contiguous方法改变了多维数组在内存中的存储顺序，以便配合view方法使用
        aligned_feat = aligned_feat.view(n, -1, h, w) * corr_prob
        #print(aligned_feat.size())  #torch.Size([1, 320, 180, 320])
        # fusion
        feat = self.feat_fusion(aligned_feat)
        # print(feat.size())    #torch.Size([1, 64, 180, 320])

        return feat
