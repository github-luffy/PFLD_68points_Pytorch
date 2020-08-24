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




