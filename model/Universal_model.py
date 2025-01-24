from typing import Sequence, Tuple, Type, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from model.SwinUNETR import SwinUNETR
from model.Unet import UNet3D
from model.DiNTS import TopologyInstance, DiNTS
from model.Unetpp import BasicUNetPlusPlus
from functools import partial
from model.adapter_block import Attention
from  model.adapter import Adapter

class Universal_model(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, backbone = 'swinunetr', encoding = 'rand_embedding'):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'swinunetr':
            self.linear = nn.Linear(34, 2)
            self.backbone = SwinUNETR(img_size=img_size,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False,
                        )
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 48),
                nn.ReLU(inplace=True),
                nn.Conv3d(48, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 768),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)), ##无论输入张量的大小如何，输出张量的大小都将是 (1,1,1)
                nn.Conv3d(768, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'unet':
            self.backbone = UNet3D()
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'dints':
            ckpt = torch.load('./model/arch_code_cvpr.pth')
            node_a = ckpt["node_a"]
            arch_code_a = ckpt["arch_code_a"]
            arch_code_c = ckpt["arch_code_c"]

            dints_space = TopologyInstance(
                    channel_mul=1.0,
                    num_blocks=12,
                    num_depths=4,
                    use_downsample=True,
                    arch_code=[arch_code_a, arch_code_c]
                )

            self.backbone = DiNTS(
                    dints_space=dints_space,
                    in_channels=1,
                    num_classes=3,
                    use_downsample=True,
                    node_a=node_a,
                )
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'unetpp':
            self.backbone = BasicUNetPlusPlus(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 256),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
            )
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone))

        self.encoding = encoding
        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8*1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.controller = nn.Conv3d(256+256, sum(weight_nums+bias_nums), kernel_size=1, stride=1, padding=0)## 512,253
        if self.encoding == 'rand_embedding':
            self.organ_embedding = nn.Embedding(out_channels, 256)
        elif self.encoding == 'word_embedding':
            self.register_buffer('organ_embedding', torch.randn(out_channels, 512))
            self.text_to_vision = nn.Linear(512, 256)
        self.class_num = out_channels


    def forward(self, x_in):
        dec4, out = self.backbone(x_in) ## x_in torch.Size([4, 1, 96, 96, 96]) out.shape [4,48,96,96,96],
        # dec4.shape [4,768,3,3,3]
        if self.encoding == 'rand_embedding':
            task_encoding = self.organ_embedding.weight.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        elif self.encoding == 'word_embedding':##self.organ_embedding:[34, 512]
            task_encoding = F.relu(self.text_to_vision(self.organ_embedding)) ## task_encoding torch.Size([34, 256]) 
            ## 将task_encodeing 变为2维的    
            # transposed_task_encoding = task_encoding.transpose(0, 1)
            # transposed_task_encoding = self.linear(transposed_task_encoding)  # shape is now [512, 2]
            # Transpose the tensor back
            # task_encoding_shink = transposed_task_encoding.transpose(0, 1)  # shape is now [2, 512]
            task_encoding = task_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2) # task_encoding torch.Size([34, 256, 1, 1, 1])
            # print(f"task_encoding shape: {torch.unique(task_encoding)}")
        x_feat = self.GAP(dec4) ##[4, 256, 1, 1, 1]
        b = x_feat.shape[0]
        logits_array = []
        for i in range(b):
            x_cond = torch.cat([x_feat[i].unsqueeze(0).repeat(self.class_num,1,1,1,1), task_encoding], 1) ## 都变化为[34,256 -->512,1,1,1] 两个按照第一个维度拼接
            params = self.controller(x_cond) ## torch.Size([34, 153, 1, 1, 1]) 经过了一个controller 卷积
            params.squeeze_(-1).squeeze_(-1).squeeze_(-1) ##[34, 153]
            
            head_inputs = self.precls_conv(out[i].unsqueeze(0)) ## input:[1,48,96,96,96] output:[1,8,96,96,96]
            head_inputs = head_inputs.repeat(self.class_num,1,1,1,1)  ## [34,8,96,96,96]
            N, _, D, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, D, H, W) ##[1,272,96,96,96]
            # print(head_inputs.shape, params.shape)
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

            logits = self.heads_forward(head_inputs, weights, biases, N)   ## 将参数作为卷积核的参数得到最终的结果 [1,34,96,96,96]
            logits_array.append(logits.reshape(1, -1, D, H, W)) ##[1,34,96,96,96]  batch 张图片的预测结果叠加起来
        #print(weights[0].shape,weights[1].shape,weights[2].shape)
        # torch.Size([272, 8, 1, 1, 1]) torch.Size([272, 8, 1, 1, 1]) torch.Size([34, 8, 1, 1, 1])
        # print(biases[0].shape,biases[1].shape,biases[2].shape)
        # torch.Size([272]) torch.Size([272]) torch.Size([34])
        out = torch.cat(logits_array,dim=0)
        # print(out.shape)
        return out
    
    def load_params(self, model_dict):
        if self.backbone_name == 'swinunetr':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out' not in key:
                    store_dict[key] = model_dict[key]

            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')
        elif self.backbone_name == 'unet':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out_tr' not in key:
                    store_dict[key.replace("module.", "")] = model_dict[key]
            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 7))
        for i in range(N):
            task_encoding[i, task_id[i]]=1
        return task_encoding.cuda()
    
    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2 ## torch.Size([34, 153]) channels = 8
        assert len(weight_nums) == len(bias_nums) ## weight_nums[64,64,8]
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1  ## 列表的拼接：[64,64,8,8,8,1] 按照第1维进行切分，每一个向量都是[34,x],x是前面的列表
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1) ## [272,8,1,1,1]
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


class Universal_model_only_txt(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, backbone = 'swinunetr', encoding = 'rand_embedding'):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'swinunetr':
            self.backbone = SwinUNETR(img_size=img_size,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False,
                        )
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 48),
                nn.ReLU(inplace=True),
                nn.Conv3d(48, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 768),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)), ##无论输入张量的大小如何，输出张量的大小都将是 (1,1,1)
                nn.Conv3d(768, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'unet':
            self.backbone = UNet3D()
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'dints':
            ckpt = torch.load('./model/arch_code_cvpr.pth')
            node_a = ckpt["node_a"]
            arch_code_a = ckpt["arch_code_a"]
            arch_code_c = ckpt["arch_code_c"]

            dints_space = TopologyInstance(
                    channel_mul=1.0,
                    num_blocks=12,
                    num_depths=4,
                    use_downsample=True,
                    arch_code=[arch_code_a, arch_code_c]
                )

            self.backbone = DiNTS(
                    dints_space=dints_space,
                    in_channels=1,
                    num_classes=3,
                    use_downsample=True,
                    node_a=node_a,
                )
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'unetpp':
            self.backbone = BasicUNetPlusPlus(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 256),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
            )
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone))

        self.encoding = encoding
        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8*1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.controller = nn.Conv3d(256+256, sum(weight_nums+bias_nums), kernel_size=1, stride=1, padding=0)## 512,253
        if self.encoding == 'rand_embedding':
            self.organ_embedding = nn.Embedding(out_channels, 256)
        elif self.encoding == 'word_embedding':
            self.register_buffer('organ_embedding', torch.randn(out_channels, 512))
            self.text_to_vision = nn.Linear(512, 256)
        self.class_num = out_channels


    def forward(self, x_in):
        dec4, out = self.backbone(x_in) ## x_in torch.Size([4, 1, 96, 96, 96]) out.shape [4,48,96,96,96],
        # dec4.shape [4,768,3,3,3]
        if self.encoding == 'rand_embedding':
            task_encoding = self.organ_embedding.weight.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        elif self.encoding == 'word_embedding':##self.organ_embedding:[34, 512]
            task_encoding = F.relu(self.text_to_vision(self.organ_embedding)) ## task_encoding torch.Size([34, 256]) 
            ## 将task_encodeing 变为2维的    
            # transposed_task_encoding = task_encoding.transpose(0, 1)
            # transposed_task_encoding = self.linear(transposed_task_encoding)  # shape is now [512, 2]
            # Transpose the tensor back
            # task_encoding_shink = transposed_task_encoding.transpose(0, 1)  # shape is now [2, 512]
            task_encoding = task_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2) # task_encoding torch.Size([34, 256, 1, 1, 1])
            # print(f"task_encoding shape: {torch.unique(task_encoding)}")
        x_feat = self.GAP(dec4) ##[4, 256, 1, 1, 1]
        b = x_feat.shape[0]
        logits_array = []
        for i in range(b):
            x_cond = torch.cat([task_encoding, task_encoding], 1) ## 都变化为[34,256 -->512,1,1,1] 两个按照第一个维度拼接
            params = self.controller(x_cond) ## torch.Size([34, 153, 1, 1, 1]) 经过了一个controller 卷积
            params.squeeze_(-1).squeeze_(-1).squeeze_(-1) ##[34, 153]
            
            head_inputs = self.precls_conv(out[i].unsqueeze(0)) ## input:[1,48,96,96,96] output:[1,8,96,96,96]
            head_inputs = head_inputs.repeat(self.class_num,1,1,1,1)  ## [34,8,96,96,96]
            N, _, D, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, D, H, W) ##[1,272,96,96,96]
            # print(head_inputs.shape, params.shape)
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

            logits = self.heads_forward(head_inputs, weights, biases, N)   ## 将参数作为卷积核的参数得到最终的结果 [1,34,96,96,96]
            logits_array.append(logits.reshape(1, -1, D, H, W)) ##[1,34,96,96,96]  batch 张图片的预测结果叠加起来
        #print(weights[0].shape,weights[1].shape,weights[2].shape)
        # torch.Size([272, 8, 1, 1, 1]) torch.Size([272, 8, 1, 1, 1]) torch.Size([34, 8, 1, 1, 1])
        # print(biases[0].shape,biases[1].shape,biases[2].shape)
        # torch.Size([272]) torch.Size([272]) torch.Size([34])
        out = torch.cat(logits_array,dim=0)
        # print(out.shape)
        return out
    
    def load_params(self, model_dict):
        if self.backbone_name == 'swinunetr':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out' not in key:
                    store_dict[key] = model_dict[key]

            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')
        elif self.backbone_name == 'unet':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out_tr' not in key:
                    store_dict[key.replace("module.", "")] = model_dict[key]
            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 7))
        for i in range(N):
            task_encoding[i, task_id[i]]=1
        return task_encoding.cuda()
    
    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2 ## torch.Size([34, 153]) channels = 8
        assert len(weight_nums) == len(bias_nums) ## weight_nums[64,64,8]
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1  ## 列表的拼接：[64,64,8,8,8,1] 按照第1维进行切分，每一个向量都是[34,x],x是前面的列表
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1) ## [272,8,1,1,1]
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


class Universal_model_attention(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, backbone = 'swinunetr', encoding = 'rand_embedding',is_concat = True):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.backbone_name = backbone
        self.is_concat = is_concat
        if backbone == 'swinunetr':
            self.linear = nn.Linear(34, 2)
            self.backbone = SwinUNETR(img_size=img_size,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False,
                        )
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 48),
                nn.ReLU(inplace=True),
                nn.Conv3d(48, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 768),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)), ##无论输入张量的大小如何，输出张量的大小都将是 (1,1,1)
                nn.Conv3d(768, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'unet':
            self.backbone = UNet3D()
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'dints':
            ckpt = torch.load('./model/arch_code_cvpr.pth')
            node_a = ckpt["node_a"]
            arch_code_a = ckpt["arch_code_a"]
            arch_code_c = ckpt["arch_code_c"]

            dints_space = TopologyInstance(
                    channel_mul=1.0,
                    num_blocks=12,
                    num_depths=4,
                    use_downsample=True,
                    arch_code=[arch_code_a, arch_code_c]
                )

            self.backbone = DiNTS(
                    dints_space=dints_space,
                    in_channels=1,
                    num_classes=3,
                    use_downsample=True,
                    node_a=node_a,
                )
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'unetpp':
            self.backbone = BasicUNetPlusPlus(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 256),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
            )
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone))

        self.encoding = encoding
        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8*1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.controller_256 = nn.Conv3d(256, sum(weight_nums+bias_nums), kernel_size=1, stride=1, padding=0)## 512,253
        self.controller = nn.Conv3d(256+256, sum(weight_nums+bias_nums), kernel_size=1, stride=1, padding=0)## 512,253
        if self.encoding == 'rand_embedding':
            self.organ_embedding = nn.Embedding(out_channels, 256)
        elif self.encoding == 'word_embedding':
            self.register_buffer('organ_embedding', torch.randn(out_channels, 512))
            self.text_to_vision = nn.Linear(512, 256)
        self.class_num = out_channels


    def forward(self, x_in):
        dec4, out = self.backbone(x_in) ## x_in torch.Size([4, 1, 96, 96, 96]) out.shape [4,48,96,96,96],
        # dec4.shape [4,768,3,3,3]
        if self.encoding == 'rand_embedding':
            task_encoding = self.organ_embedding.weight.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        elif self.encoding == 'word_embedding':##self.organ_embedding:[34, 512]
            task_encoding = F.relu(self.text_to_vision(self.organ_embedding)) ## task_encoding torch.Size([34, 256]) 
            ## 将task_encodeing 变为2维的    
            # transposed_task_encoding = task_encoding.transpose(0, 1)
            # transposed_task_encoding = self.linear(transposed_task_encoding)  # shape is now [512, 2]
            # Transpose the tensor back
            # task_encoding_shink = transposed_task_encoding.transpose(0, 1)  # shape is now [2, 512]
            task_encoding = task_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2) # task_encoding torch.Size([34, 256, 1, 1, 1])
        
        x_feat = self.GAP(dec4) ##[4, 256, 1, 1, 1]
        b = x_feat.shape[0]
        logits_array = []
        for i in range(b):
            # x_repeat = x_feat[i].unsqueeze(0).repeat(self.class_num,1,1,1,1).T  ## [256,4]
            x_repeat = x_feat[i].squeeze().unsqueeze(0).repeat(self.class_num,1)
            attention = torch.mm(task_encoding.squeeze() , task_encoding.squeeze().T)## [4,4]
            #####normalize attention
            min_val = torch.min(attention)
            max_val = torch.max(attention)
            attention = (attention - min_val) / (max_val - min_val)
            #######normalize attention
            x_attention_feat = torch.matmul(attention,x_repeat).unsqueeze(2).unsqueeze(3).unsqueeze(4)#.T
            if self.is_concat:
                ## solution 1 : concat
                x_cond = torch.cat([x_attention_feat, task_encoding], 1) ## 都变化为[34,256 -->512,1,1,1] 两个按照第一个维度拼接
                params = self.controller(x_cond) ## torch.Size([34, 153, 1, 1, 1]) 经过了一个controller 卷积
            else:
                ## solution 2 : no concat
                x_cond = x_attention_feat
                params = self.controller_256(x_cond)

            params.squeeze_(-1).squeeze_(-1).squeeze_(-1) ##[34, 153]
            head_inputs = self.precls_conv(out[i].unsqueeze(0)) ## input:[1,48,96,96,96] output:[1,8,96,96,96]
            head_inputs = head_inputs.repeat(self.class_num,1,1,1,1)  ## [34,8,96,96,96]
            N, _, D, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, D, H, W) ##[1,272,96,96,96]
            # print(head_inputs.shape, params.shape)
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

            logits = self.heads_forward(head_inputs, weights, biases, N)   ## 将参数作为卷积核的参数得到最终的结果 [1,34,96,96,96]
            logits_array.append(logits.reshape(1, -1, D, H, W)) ##[1,34,96,96,96]  batch 张图片的预测结果叠加起来
        #print(weights[0].shape,weights[1].shape,weights[2].shape)
        # torch.Size([272, 8, 1, 1, 1]) torch.Size([272, 8, 1, 1, 1]) torch.Size([34, 8, 1, 1, 1])
        # print(biases[0].shape,biases[1].shape,biases[2].shape)
        # torch.Size([272]) torch.Size([272]) torch.Size([34])
        out = torch.cat(logits_array,dim=0)
        # print(out.shape)
        return out
    
    def load_params(self, model_dict):
        if self.backbone_name == 'swinunetr':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out' not in key:
                    store_dict[key] = model_dict[key]

            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')
        elif self.backbone_name == 'unet':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out_tr' not in key:
                    store_dict[key.replace("module.", "")] = model_dict[key]
            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 7))
        for i in range(N):
            task_encoding[i, task_id[i]]=1
        return task_encoding.cuda()
    
    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2 ## torch.Size([34, 153]) channels = 8
        assert len(weight_nums) == len(bias_nums) ## weight_nums[64,64,8]
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1  ## 列表的拼接：[64,64,8,8,8,1] 按照第1维进行切分，每一个向量都是[34,x],x是前面的列表
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1) ## [272,8,1,1,1]
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x



class Universal_model_2_txt_encoder(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, backbone = 'swinunetr', encoding = 'rand_embedding'):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'swinunetr':
            self.linear = nn.Linear(34, 2)
            self.backbone = SwinUNETR(img_size=img_size,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False,
                        )
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 48),
                nn.ReLU(inplace=True),
                nn.Conv3d(48, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 768),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)), ##无论输入张量的大小如何，输出张量的大小都将是 (1,1,1)
                nn.Conv3d(768, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'unet':
            self.backbone = UNet3D()
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'dints':
            ckpt = torch.load('./model/arch_code_cvpr.pth')
            node_a = ckpt["node_a"]
            arch_code_a = ckpt["arch_code_a"]
            arch_code_c = ckpt["arch_code_c"]

            dints_space = TopologyInstance(
                    channel_mul=1.0,
                    num_blocks=12,
                    num_depths=4,
                    use_downsample=True,
                    arch_code=[arch_code_a, arch_code_c]
                )

            self.backbone = DiNTS(
                    dints_space=dints_space,
                    in_channels=1,
                    num_classes=3,
                    use_downsample=True,
                    node_a=node_a,
                )
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'unetpp':
            self.backbone = BasicUNetPlusPlus(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 256),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
            )
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone))

        self.encoding = encoding
        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8*1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.controller = nn.Conv3d(256+256, sum(weight_nums+bias_nums), kernel_size=1, stride=1, padding=0)## 512,253
        if self.encoding == 'rand_embedding':
            self.organ_embedding = nn.Embedding(out_channels, 256)
        elif self.encoding == 'word_embedding':
            self.register_buffer('organ_embedding', torch.randn(out_channels, 512))
            self.text_to_vision = nn.Linear(512, 256)
        self.class_num = out_channels


    def forward(self, x_in):
        dec4, out = self.backbone(x_in) ## x_in torch.Size([4, 1, 96, 96, 96]) out.shape [4,48,96,96,96],
        # dec4.shape [4,768,3,3,3]
        if self.encoding == 'rand_embedding':
            task_encoding = self.organ_embedding.weight.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        elif self.encoding == 'word_embedding':##self.organ_embedding:[34, 512]
            task_encoding = F.relu(self.text_to_vision(self.organ_embedding)) ## task_encoding torch.Size([34, 256]) 
            ## 将task_encodeing 变为2维的    
            # transposed_task_encoding = task_encoding.transpose(0, 1)
            # transposed_task_encoding = self.linear(transposed_task_encoding)  # shape is now [512, 2]
            # Transpose the tensor back
            # task_encoding_shink = transposed_task_encoding.transpose(0, 1)  # shape is now [2, 512]
            task_encoding = task_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2) # task_encoding torch.Size([34, 256, 1, 1, 1])
        
        x_feat = self.GAP(dec4) ##[4, 256, 1, 1, 1]
        b = x_feat.shape[0]
        logits_array = []
        for i in range(b):
            x_cond = torch.cat([x_feat[i].unsqueeze(0).repeat(2,1,1,1,1), task_encoding], 1) ## 都变化为[34,256 -->512,1,1,1] 两个按照第一个维度拼接
            params = self.controller(x_cond) ## torch.Size([34, 153, 1, 1, 1]) 经过了一个controller 卷积
            params.squeeze_(-1).squeeze_(-1).squeeze_(-1) ##[34, 153]
            
            head_inputs = self.precls_conv(out[i].unsqueeze(0)) ## input:[1,48,96,96,96] output:[1,8,96,96,96]
            head_inputs = head_inputs.repeat(self.class_num,1,1,1,1)  ## [34,8,96,96,96]
            N, _, D, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, D, H, W) ##[1,272,96,96,96]
            # print(head_inputs.shape, params.shape)
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

            logits = self.heads_forward(head_inputs, weights, biases, N)   ## 将参数作为卷积核的参数得到最终的结果 [1,34,96,96,96]
            logits_array.append(logits.reshape(1, -1, D, H, W)) ##[1,34,96,96,96]  batch 张图片的预测结果叠加起来
        #print(weights[0].shape,weights[1].shape,weights[2].shape)
        # torch.Size([272, 8, 1, 1, 1]) torch.Size([272, 8, 1, 1, 1]) torch.Size([34, 8, 1, 1, 1])
        # print(biases[0].shape,biases[1].shape,biases[2].shape)
        # torch.Size([272]) torch.Size([272]) torch.Size([34])
        out = torch.cat(logits_array,dim=0)
        # print(out.shape)
        return out
    
    def load_params(self, model_dict):
        if self.backbone_name == 'swinunetr':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out' not in key:
                    store_dict[key] = model_dict[key]

            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')
        elif self.backbone_name == 'unet':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out_tr' not in key:
                    store_dict[key.replace("module.", "")] = model_dict[key]
            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 7))
        for i in range(N):
            task_encoding[i, task_id[i]]=1
        return task_encoding.cuda()
    
    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2 ## torch.Size([34, 153]) channels = 8
        assert len(weight_nums) == len(bias_nums) ## weight_nums[64,64,8]
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1  ## 列表的拼接：[64,64,8,8,8,1] 按照第1维进行切分，每一个向量都是[34,x],x是前面的列表
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1) ## [272,8,1,1,1]
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x
