# -*- coding: utf-8 -*-

import torch
from model2 import MobileNetV2, BlazeLandMark

coefficient = 0.25
print(coefficient)
num_of_channels = [int(64 * coefficient), int(128 * coefficient), int(16 * coefficient), int(32 * coefficient),
                    int(128 * coefficient)]
# model = MobileNetV2(num_of_channels=num_of_channels, nums_class=136)
model = BlazeLandMark(nums_class=136)
path = './model_37.pth'
dst = './pfld_37.onnx'
model = torch.load(path)

x = torch.rand(1, 3, 56, 56).cuda()

torch_out = torch.onnx._export(model, x, dst, export_params=True)

import onnx
print("==> Loading and checking exported model from '{}'".format(dst))
onnx_model = onnx.load(dst)
onnx.checker.check_model(onnx_model)  # assuming throw on error
print("==> Passed")

