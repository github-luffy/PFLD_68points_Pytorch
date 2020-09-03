# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional
import math
from collections import OrderedDict


class WingLoss(nn.Module):

    def __init__(self, wing_w=10.0, wing_epsilon=2.0):
        super(WingLoss, self).__init__()
        self.wing_w = wing_w
        self.wing_epsilon = wing_epsilon
        self.wing_c = self.wing_w * (1.0 - math.log(1.0 + self.wing_w / self.wing_epsilon))

    def forward(self, targets, predictions, euler_angle_weights=None):
        abs_error = torch.abs(targets - predictions)
        loss = torch.where(torch.le(abs_error, self.wing_w),
                           self.wing_w * torch.log(1.0 + abs_error / self.wing_epsilon), abs_error - self.wing_c)
        loss_sum = torch.sum(loss, 1)
        if euler_angle_weights is not None:
            loss_sum *= euler_angle_weights
        return torch.mean(loss_sum)


class LinearBottleneck(nn.Module):

    def __init__(self, input_channels, out_channels, expansion, stride=1, activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.expansion_channels = input_channels * expansion

        self.conv1 = nn.Conv2d(input_channels, self.expansion_channels, stride=1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.expansion_channels)

        self.depth_conv2 = nn.Conv2d(self.expansion_channels, self.expansion_channels, stride=stride, kernel_size=3,
                                     groups=self.expansion_channels, padding=1)
        self.bn2 = nn.BatchNorm2d(self.expansion_channels)

        self.conv3 = nn.Conv2d(self.expansion_channels, out_channels, stride=1, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.activation = activation(inplace=True)  # inplace=True
        self.stride = stride
        self.input_channels = input_channels
        self.out_channels = out_channels

    def forward(self, input):
        residual = input

        out = self.conv1(input)
        out = self.bn1(out)
        # out = self.activation(out)

        out = self.depth_conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.input_channels == self.out_channels:
            out += residual
        return out


class AuxiliaryNet(nn.Module):

    def __init__(self, input_channels, nums_class=3, activation=nn.ReLU, first_conv_stride=2):
        super(AuxiliaryNet, self).__init__()
        self.input_channels = input_channels
        # self.num_channels = [128, 128, 32, 128, 32]
        self.num_channels = [512, 512, 512, 512, 1024]
        self.conv1 = nn.Conv2d(self.input_channels, self.num_channels[0], kernel_size=3, stride=first_conv_stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels[0])

        self.conv2 = nn.Conv2d(self.num_channels[0], self.num_channels[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels[1])

        self.conv3 = nn.Conv2d(self.num_channels[1], self.num_channels[2], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels[2])

        self.conv4 = nn.Conv2d(self.num_channels[2], self.num_channels[3], kernel_size=7, stride=1, padding=3)
        self.bn4 = nn.BatchNorm2d(self.num_channels[3])

        self.fc1 = nn.Linear(in_features=self.num_channels[3], out_features=self.num_channels[4])
        self.fc2 = nn.Linear(in_features=self.num_channels[4], out_features=nums_class)

        self.activation = activation(inplace=True)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):

        out = self.conv1(input)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.activation(out)

        out = functional.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
        #print(out.size())
        # out = out.view(out.size(0), -1)
        out = self.fc1(out)
        euler_angles_pre = self.fc2(out)

        return euler_angles_pre


# class AuxiliaryNet(nn.Module):
#
#     def __init__(self, input_channels, nums_class=3, activation=nn.ReLU):
#         super(AuxiliaryNet, self).__init__()
#         self.input_channels = input_channels
#         # self.num_channels = [128, 128, 32, 128, 32]
#         self.num_channels = [512, 512, 512, 512, 1024]
#         self.conv1 = nn.Conv2d(self.input_channels, self.num_channels[0], kernel_size=3, stride=2,
#                                padding=1)
#         self.bn1 = nn.BatchNorm2d(self.num_channels[0])
#
#         self.conv2 = nn.Conv2d(self.num_channels[0], self.num_channels[0], kernel_size=3, stride=2,
#                                padding=1)
#         self.bn2 = nn.BatchNorm2d(self.num_channels[0])
#
#         self.conv3 = nn.Conv2d(self.num_channels[0], self.num_channels[1], kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(self.num_channels[1])
#
#         self.conv4 = nn.Conv2d(self.num_channels[1], self.num_channels[2], kernel_size=3, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(self.num_channels[2])
#
#         self.conv5 = nn.Conv2d(self.num_channels[2], self.num_channels[3], kernel_size=7, stride=1, padding=3)
#         self.bn5 = nn.BatchNorm2d(self.num_channels[3])
#
#         self.fc1 = nn.Linear(in_features=self.num_channels[3], out_features=self.num_channels[4])
#         self.fc2 = nn.Linear(in_features=self.num_channels[4], out_features=nums_class)
#
#         self.activation = activation(inplace=True)
#
#         self.init_params()
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, input):
#
#         out = self.conv1(input)
#         out = self.bn1(out)
#         out = self.activation(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.activation(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#         out = self.activation(out)
#
#         out = self.conv4(out)
#         out = self.bn4(out)
#         out = self.activation(out)
#
#         out = self.conv5(out)
#         out = self.bn5(out)
#         out = self.activation(out)
#
#         out = functional.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
#
#         # out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         euler_angles_pre = self.fc2(out)
#
#         return euler_angles_pre


class MobileNetV2(nn.Module):

    def __init__(self, input_channels=3, num_of_channels=None, nums_class=136, activation=nn.ReLU6):
        super(MobileNetV2, self).__init__()
        assert num_of_channels is not None
        self.num_of_channels = num_of_channels
        self.conv1 = nn.Conv2d(input_channels, self.num_of_channels[0], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_of_channels[0])

        self.depth_conv2 = nn.Conv2d(self.num_of_channels[0], self.num_of_channels[0], kernel_size=3, stride=1,
                                     padding=1, groups=self.num_of_channels[0])
        self.bn2 = nn.BatchNorm2d(self.num_of_channels[0])

        self.stage0 = self.make_stage(self.num_of_channels[0], self.num_of_channels[0], stride=2, stage=0, times=5,
                                      expansion=2, activation=activation)

        self.stage1 = self.make_stage(self.num_of_channels[0], self.num_of_channels[1], stride=2, stage=1, times=7,
                                      expansion=4, activation=activation)

        self.linear_bottleneck_end = nn.Sequential(LinearBottleneck(self.num_of_channels[1], self.num_of_channels[2],
                                                                    expansion=2, stride=1, activation=activation))

        self.conv3 = nn.Conv2d(self.num_of_channels[2], self.num_of_channels[3], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_of_channels[3])

        self.conv4 = nn.Conv2d(self.num_of_channels[3], self.num_of_channels[4], kernel_size=7, stride=1)
        self.bn4 = nn.BatchNorm2d(self.num_of_channels[4])

        self.activation = activation(inplace=True)

        self.in_features = 14 * 14 * self.num_of_channels[2] + 7 * 7 * self.num_of_channels[3] + 1 * 1 * self.num_of_channels[4]
        self.fc = nn.Linear(in_features=self.in_features, out_features=nums_class)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def make_stage(self, input_channels, out_channels, stride, stage, times, expansion, activation=nn.ReLU6):
        modules = OrderedDict()
        stage_name = 'LinearBottleneck{}'.format(stage)

        module = LinearBottleneck(input_channels, out_channels, expansion=2,
                                  stride=stride, activation=activation)
        modules[stage_name+'_0'] = module

        for i in range(times - 1):
            module = LinearBottleneck(out_channels, out_channels, expansion=expansion, stride=1,
                                      activation=activation)
            module_name = stage_name+'_{}'.format(i+1)
            modules[module_name] = module

        return nn.Sequential(modules)

    def forward(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.activation(out)
        # print(out.size())

        out = self.depth_conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        # print(out.size())

        out = self.stage0(out)
        # print(out.size())

        out1 = self.stage1(out)
        # print(out1.size())

        out1 = self.linear_bottleneck_end(out1)
        # print(out1.size())

        out2 = self.conv3(out1)
        out2 = self.bn3(out2)
        out2 = self.activation(out2)
        # print(out2.size())

        out3 = self.conv4(out2)
        out3 = self.bn4(out3)
        out3 = self.activation(out3)
        # print(out3.size())

        out1 = out1.view(out1.size(0), -1)
        # print(out1.size())
        out2 = out2.view(out2.size(0), -1)
        # print(out2.size())
        out3 = out3.view(out3.size(0), -1)
        # print(out3.size())

        multi_scale = torch.cat([out1, out2, out3], 1)
        # print(multi_scale.size())

        assert multi_scale.size(1) == self.in_features

        pre_landmarks = self.fc(multi_scale)
        return pre_landmarks, out


# class MobileNetV2(nn.Module):
#
#     def __init__(self, input_channels=3, num_of_channels=None, nums_class=136, activation=nn.ReLU6):
#         super(MobileNetV2, self).__init__()
#         assert num_of_channels is not None
#         self.num_of_channels = num_of_channels
#         self.conv1 = nn.Conv2d(input_channels, self.num_of_channels[0], kernel_size=3, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(self.num_of_channels[0])
#
#         self.depth_conv2 = nn.Conv2d(self.num_of_channels[0], self.num_of_channels[0], kernel_size=3, stride=1,
#                                      padding=1, groups=self.num_of_channels[0])
#         self.bn2 = nn.BatchNorm2d(self.num_of_channels[0])
#
#         self.stage0 = self.make_stage(self.num_of_channels[0], self.num_of_channels[0], stride=2, stage=0, times=3,
#                                       expansion=2, activation=activation)
#
#         self.stage1 = self.make_stage(self.num_of_channels[0], self.num_of_channels[1], stride=2, stage=1, times=5,
#                                       expansion=4, activation=activation)
#
#         self.linear_bottleneck_end = nn.Sequential(LinearBottleneck(self.num_of_channels[1], self.num_of_channels[2],
#                                                                     expansion=2, stride=1, activation=activation))
#
#         self.conv3 = nn.Conv2d(self.num_of_channels[2], self.num_of_channels[3], kernel_size=3, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(self.num_of_channels[3])
#
#         self.conv4 = nn.Conv2d(self.num_of_channels[3], self.num_of_channels[4], kernel_size=7, stride=1)
#         self.bn4 = nn.BatchNorm2d(self.num_of_channels[4])
#
#         self.activation = activation(inplace=True)
#
#         self.avg_pool1 = nn.AvgPool2d(kernel_size=14, stride=1)
#
#         self.avg_pool2 = nn.AvgPool2d(kernel_size=7, stride=1)
#
#         self.in_features = self.num_of_channels[2] + self.num_of_channels[3] + self.num_of_channels[4]
#         self.fc = nn.Linear(in_features=self.in_features, out_features=nums_class)
#
#         self.init_params()
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def make_stage(self, input_channels, out_channels, stride, stage, times, expansion, activation=nn.ReLU6):
#         modules = OrderedDict()
#         stage_name = 'LinearBottleneck{}'.format(stage)
#
#         module = LinearBottleneck(input_channels, out_channels, expansion=2,
#                                   stride=stride, activation=activation)
#         modules[stage_name+'_0'] = module
#
#         for i in range(times - 1):
#             module = LinearBottleneck(out_channels, out_channels, expansion=expansion, stride=1,
#                                       activation=activation)
#             module_name = stage_name+'_{}'.format(i+1)
#             modules[module_name] = module
#
#         return nn.Sequential(modules)
#
#     def forward(self, input):
#         out = self.conv1(input)
#         out = self.bn1(out)
#         out = self.activation(out)
#         # print(out.size())
#
#         out = self.depth_conv2(out)
#         out = self.bn2(out)
#         out = self.activation(out)
#         # print(out.size())
#
#         out = self.stage0(out)
#         # print(out.size())
#
#         out1 = self.stage1(out)
#         # print(out1.size())
#
#         out1 = self.linear_bottleneck_end(out1)
#         # print(out1.size())
#
#         out2 = self.conv3(out1)
#         out2 = self.bn3(out2)
#         out2 = self.activation(out2)
#         # print(out2.size())
#
#         out3 = self.conv4(out2)
#         out3 = self.bn4(out3)
#         out3 = self.activation(out3)
#         # print(out3.size())
#
#         out1 = self.avg_pool1(out1)
#
#         out2 = self.avg_pool2(out2)
#
#         out1 = out1.view(out1.size(0), -1)
#         # print(out1.size())
#         out2 = out2.view(out2.size(0), -1)
#         # print(out2.size())
#         out3 = out3.view(out3.size(0), -1)
#         # print(out3.size())
#
#         multi_scale = torch.cat([out1, out2, out3], 1)
#         # print(multi_scale.size())
#
#         assert multi_scale.size(1) == self.in_features
#
#         pre_landmarks = self.fc(multi_scale)
#         return pre_landmarks, out


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super(BlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        if stride > 1:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=5, stride=stride, padding=2,
                      groups=in_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        out = (branch1+self.shortcut(x)) if self.use_pool else (branch1+x)
        return self.relu(out)


class DoubleBlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super(DoubleBlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        if stride > 1:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=stride,
                      padding=2, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=5, stride=1, padding=2,
                      groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        out = (branch1 + self.shortcut(x)) if self.use_pool else (branch1 + x)
        return self.relu(out)

 
class ASPP(nn.Module):

    def __init__(self, in_channels=96, out_channels=96):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=5, dilation=5)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)

        self.conv_1x1_3 = nn.Conv2d(in_channels*5, out_channels, kernel_size=1) # (480 = 5*96)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2] 
        feature_map_w = feature_map.size()[3] 

        out_1x1 = functional.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) 
        out_3x3_1 = functional.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) 
        out_3x3_2 = functional.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) 
        out_3x3_3 = functional.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map) 
        out_img = functional.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) 
        out_img = functional.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear", align_corners=True) 

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = functional.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

        return out
    

class BlazeLandMark(nn.Module):
    def __init__(self, nums_class=136):
        super(BlazeLandMark, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        
            # nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(24),
            # nn.ReLU(inplace=True),
        )
        
        self.blazeBlock = nn.Sequential(
            BlazeBlock(in_channels=24, out_channels=24),
            BlazeBlock(in_channels=24, out_channels=24),
            BlazeBlock(in_channels=24, out_channels=48, stride=2),
            BlazeBlock(in_channels=48, out_channels=48),
            BlazeBlock(in_channels=48, out_channels=48),
        )
        
        self.doubleBlazeBlock1 = nn.Sequential(
            DoubleBlazeBlock(in_channels=48, out_channels=96, mid_channels=24, stride=2),
            DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24),
            DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24))
        
        self.doubleBlazeBlock2 = nn.Sequential(
            DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24, stride=2),
            DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24),
            DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24),
        )

        # self.firstconv = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(12),
        #     nn.ReLU(inplace=True),
        # )

        # self.blazeBlock = nn.Sequential(
        #     BlazeBlock(in_channels=12, out_channels=12),
        #     BlazeBlock(in_channels=12, out_channels=12),
        #     BlazeBlock(in_channels=12, out_channels=24, stride=2),
        #     BlazeBlock(in_channels=24, out_channels=24),
        #     BlazeBlock(in_channels=24, out_channels=24),
        # )

        # self.doubleBlazeBlock1 = nn.Sequential(
        #     DoubleBlazeBlock(in_channels=24, out_channels=48, mid_channels=12, stride=2),
        #     DoubleBlazeBlock(in_channels=48, out_channels=48, mid_channels=12),
        #     DoubleBlazeBlock(in_channels=48, out_channels=48, mid_channels=12))

        # self.doubleBlazeBlock2 = nn.Sequential(
        #     DoubleBlazeBlock(in_channels=48, out_channels=48, mid_channels=12, stride=2),
        #     DoubleBlazeBlock(in_channels=48, out_channels=48, mid_channels=12),
        #     DoubleBlazeBlock(in_channels=48, out_channels=48, mid_channels=12),
        # )

        self.secondconv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=7, stride=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        
        self.aspp = ASPP(in_channels=96, out_channels=96)
        #self.in_features = 48 + 96 + 192

        self.in_features = 96*7*7

        self.fc = nn.Linear(in_features=self.in_features, out_features=nums_class)

        self.init_params()

    # def initialize(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        fisrt_out = self.firstconv(input)

        #out = self.blazeBlock(input)
        #print(out.size())
        #
        #out1 = self.doubleBlazeBlock1(out)
        # print(out1.size())
        #
        #out2 = self.doubleBlazeBlock2(out1)
        # print(out2.size())
        #
        #out3 = self.secondconv(out2)
        #
        # assert out1.size(2) == 14 and out1.size(3) == 14
        # out1 = self.avg_pool1(out1)
        #
        # assert out2.size(2) == 7 and out2.size(3) == 7
        # out2 = self.avg_pool2(out2)

        block_out1 = self.blazeBlock(fisrt_out)
        # print(out1.size())

        block_out2 = self.doubleBlazeBlock1(block_out1)
        # print(out2.size())

        block_out3 = self.aspp(self.doubleBlazeBlock2(block_out2))
        # print(out3.size())
        # assert out1.size(2) == 14 and out1.size(3) == 14
        # out1_ = self.avg_pool1(out1)
        # assert out2.size(2) == 7 and out2.size(3) == 7
        # out2_ = self.avg_pool2(out2)
        # assert out3.size(2) == 4 and out3.size(3) == 4
        # out3_ = self.avg_pool3(out3)
        # out1_ = out1_.view(out1_.size(0), -1)
        # out2_ = out2_.view(out2_.size(0), -1)
        # out3_ = out3_.view(out3_.size(0), -1)
        # multi_scale = torch.cat([out1_, out2_, out3_], 1)
        # assert multi_scale.size(1) == self.in_features
        # pre_landmarks = self.fc(multi_scale)

        # functional.adaptive_avg_pool2d: global avg_pool
        # 1 means 1*1 feature map no matter what input size
        # squeeze(-1): delete dims that number is 1
        #block_out1_ = functional.adaptive_avg_pool2d(block_out1, 1).squeeze(-1).squeeze(-1)
        #block_out2_ = functional.adaptive_avg_pool2d(block_out2, 1).squeeze(-1).squeeze(-1)
        #block_out3_ = functional.adaptive_avg_pool2d(block_out3, 1).squeeze(-1).squeeze(-1)
        #print(block_out3_.size())
        block_out3 = block_out3.view(block_out3.size(0), -1)
        assert block_out3.size(1) == self.in_features
        pre_landmarks = self.fc(block_out3)
        
        return pre_landmarks, block_out1

 
#-----------------Efficient-------------------------
from efficientnet.model import EfficientNet


class EFFNet(nn.Module):

    def __init__(self, compound_coef=0, load_weights=False):
        super(EFFNet, self).__init__()
        print(f'efficientnet-b{compound_coef}')
        model = EfficientNet.from_pretrained(f'efficientnet-b{compound_coef}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)

        feature_maps = []
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate)
            #print('x', x.size())
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)

            last_x = x
            #print('last_x', last_x.size())
        del last_x
        return feature_maps[1:]


class EfficientLM(nn.Module):

    def __init__(self, nums_class=136, compound_coef=0, load_weights=False):
        super(EfficientLM, self).__init__()
        self.nums_class = nums_class
        self.compound_coef = compound_coef
        self.backbone_coef = [0, 1, 2, 3, 4, 5, 6, 7]
        self.inchannels_list = [320, 320, 352, 384, 448, 512, 576, 640]
        self.p8_outchannels_list = [40, 40, 48, 48, 56, 64, 72, 80]
        self.p8_outchannels = self.p8_outchannels_list[self.compound_coef]

        self.backbone = EFFNet(self.backbone_coef[self.compound_coef], load_weights=load_weights)

        self.in_features = self.inchannels_list[self.compound_coef] * 4 * 4 # 112*112
        self.fc = nn.Linear(in_features=self.in_features, out_features=self.nums_class)

    def forward(self, x):
        p4, p8, p16, p32 = self.backbone(x)
        p32 = p32.view(p32.size(0), -1)
        output = self.fc(p32)
        return output, p8


# HRNetV2
import numpy as np
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


blocks_dict = {
    'BASIC':BasicBlock,
    'BOTTLENECK':BottleNeck
}


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=[height_output, width_output], mode='bilinear', align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse



class HighResolutionNet(nn.Module):

    def __init__(self, nums_class=136):
        super(HighResolutionNet, self).__init__()

        self.num_branches1 = 2
        self.num_branches2 = 3
        self.num_branches3 = 4

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        

        self.layer1 = self._make_layer(BottleNeck, 64, 64, 4)
        layer1_out_channel = BottleNeck.expansion*64

        num_channels1 = [48, 96]
        num_channels_expansion1 = [num_channels1[i] * BasicBlock.expansion for i in range(len(num_channels1))]
        self.transition1 = self._make_transition_layer([layer1_out_channel], num_channels_expansion1)
        # layer_config['NUM_MODULES'] layer_config['NUM_BRANCHES'] layer_config['NUM_BLOCKS']
        # layer_config['NUM_CHANNELS'] blocks_dict[layer_config['BLOCK']] layer_config['FUSE_METHOD']
        self.stage2, pre_stage_channels = self._make_stage(1, self.num_branches1, [4, 4], num_channels1, BasicBlock, 'SUM', num_channels_expansion1)

        num_channels2 = [48, 96, 192]
        num_channels_expansion2 = [num_channels2[i] * BasicBlock.expansion for i in range(len(num_channels2))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels_expansion2)
        self.stage3, pre_stage_channels = self._make_stage(4, self.num_branches2, [4, 4, 4], num_channels2, BasicBlock, 'SUM', num_channels_expansion2)

        num_channels3 = [48, 96, 192, 384]
        num_channels_expansion3 = [num_channels3[i] * BasicBlock.expansion for i in range(len(num_channels3))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels_expansion3)
        self.stage4, pre_stage_channels = self._make_stage(3, self.num_branches3, [4, 4, 4, 4], num_channels3, BasicBlock, 'SUM', num_channels_expansion3, multi_scale_output=True)
        
        last_inp_channels = np.int(np.sum(pre_stage_channels))
        
        self.FINAL_CONV_KERNEL = 1
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=False),
            # nn.Conv2d(in_channels=last_inp_channels, out_channels=nums_class, kernel_size=self.FINAL_CONV_KERNEL,
            #     stride=1, padding=1 if self.FINAL_CONV_KERNEL == 3 else 0)
        )

        self.in_features = last_inp_channels * 28 * 28
        self.fc = nn.Linear(in_features=self.in_features, out_features=nums_class)

        self.init_weights()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)   

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=False)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, num_modules, num_branches, num_blocks, num_channels, block, fuse_method, num_inchannels, multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def init_weights(self):
        # print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        axn_input = self.relu(self.bn2(self.conv2(x)))
        out = self.layer1(axn_input)

        x_list = []
        for i in range(self.num_branches1):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](out))
            else:
                x_list.append(out)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.num_branches2):
            if self.transition2[i] is not None:
                if i < self.num_branches1:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.num_branches3):
            if self.transition3[i] is not None:
                if i < self.num_branches2:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        out = self.stage4(x_list)
        # print(out[0].size(), out[1].size(), out[2].size(), out[3].size())
        # Upsampling
        x0_h, x0_w = out[0].size(2), out[0].size(3)
        x1 = F.interpolate(out[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(out[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(out[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        out = torch.cat([out[0], x1, x2, x3], 1)

        out = self.last_layer(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        # print(out.size(), axn_input.size())
        return out, axn_input

# resNest50
from resnest.torch import resnest50
class MyResNest50(nn.Module):
    
    def __init__(self, nums_class=136):
        super(MyResNest50, self).__init__()

        self.resnest = resnest50(pretrained=True)
        self.resnest_backbone1 = nn.Sequential(*list(self.resnest.children())[:-6])
        self.resnest_backbone_end = nn.Sequential(*list(self.resnest.children())[-6:-2])
        
        self.in_features = 2048 * 4 * 4
        self.fc = nn.Linear(in_features=self.in_features, out_features=nums_class)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        auxnet = self.resnest_backbone1(x)
        #print(auxnet.size())
        out = self.resnest_backbone_end(auxnet)
        #print(out.size())
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, auxnet

