# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional
import math
from collections import OrderedDict

import tensorflow as tf
import tensorflow.contrib.slim as slim


def mobilenet_v2(input, weight_decay, batch_norm_params):
    features = {}
    with tf.variable_scope('Mobilenet'):
        with slim.arg_scope([slim.convolution2d, slim.separable_conv2d], \
                            activation_fn=tf.nn.relu6,\
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            padding='SAME'):
            print('Mobilnet input shape({}): {}'.format(input.name, input.get_shape()))

            # 96*96*3   112*112*3
            conv_1 = slim.convolution2d(input, 32, [3, 3], stride=2, scope='conv_1')
            print(conv_1.name, conv_1.get_shape())

            # 48*48*32  56*56*32
            conv2_1 = slim.separable_convolution2d(conv_1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv2_1/dwise')
            print(conv2_1.name, conv2_1.get_shape())
            conv2_1 = slim.convolution2d(conv2_1, 16, [1, 1], stride=1, activation_fn=None,
                                         scope='conv2_1/linear')
            print(conv2_1.name, conv2_1.get_shape())
            features['feature2'] = conv2_1
            # 48*48*16  56*56*16
            conv3_1 = slim.convolution2d(conv2_1, 96, [1, 1], stride=1, scope='conv3_1/expand')
            print(conv3_1.name, conv3_1.get_shape())
            conv3_1 = slim.separable_convolution2d(conv3_1, num_outputs=None, stride=2, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_1/dwise')
            print(conv3_1.name, conv3_1.get_shape())
            conv3_1 = slim.convolution2d(conv3_1, 24, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_1/linear')
            print(conv3_1.name, conv3_1.get_shape())

            conv3_2 = slim.convolution2d(conv3_1, 144, [1, 1], stride=1, scope='conv3_2/expand')
            print(conv3_2.name, conv3_2.get_shape())
            conv3_2 = slim.separable_convolution2d(conv3_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_2/dwise')
            print(conv3_2.name, conv3_2.get_shape())
            conv3_2 = slim.convolution2d(conv3_2, 24, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_2/linear')
            print(conv3_2.name, conv3_2.get_shape())
            block_3_2 = conv3_1 + conv3_2
            print(block_3_2.name, block_3_2.get_shape())

            features['feature3'] = block_3_2
            features['pfld'] = block_3_2
            # 24*24*24   28*28*24
            conv4_1 = slim.convolution2d(block_3_2, 144, [1, 1], stride=1, scope='conv4_1/expand')
            print(conv4_1.name, conv4_1.get_shape())
            conv4_1 = slim.separable_convolution2d(conv4_1, num_outputs=None, stride=2, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv4_1/dwise')
            print(conv4_1.name, conv4_1.get_shape())
            conv4_1 = slim.convolution2d(conv4_1, 32, [1, 1], stride=1, activation_fn=None,
                                         scope='conv4_1/linear')
            print(conv4_1.name, conv4_1.get_shape())

            conv4_2 = slim.convolution2d(conv4_1, 192, [1, 1], stride=1, scope='conv4_2/expand')
            print(conv4_2.name, conv4_2.get_shape())
            conv4_2 = slim.separable_convolution2d(conv4_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv4_2/dwise')
            print(conv4_2.name, conv4_2.get_shape())
            conv4_2 = slim.convolution2d(conv4_2, 32, [1, 1], stride=1, activation_fn=None,
                                         scope='conv4_2/linear')
            print(conv4_2.name, conv4_2.get_shape())
            block_4_2 = conv4_1 + conv4_2
            print(block_4_2.name, block_4_2.get_shape())

            conv4_3 = slim.convolution2d(block_4_2, 192, [1, 1], stride=1, scope='conv4_3/expand')
            print(conv4_3.name, conv4_3.get_shape())
            conv4_3 = slim.separable_convolution2d(conv4_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv4_3/dwise')
            print(conv4_3.name, conv4_3.get_shape())
            conv4_3 = slim.convolution2d(conv4_3, 32, [1, 1], stride=1, activation_fn=None,
                                         scope='conv4_3/linear')
            print(conv4_3.name, conv4_3.get_shape())
            block_4_3 = block_4_2 + conv4_3
            print(block_4_3.name, block_4_3.get_shape())

            # 12*12*32   14*14*32
            features['feature4'] = block_4_3
            conv5_1 = slim.convolution2d(block_4_3, 192, [1, 1], stride=1, scope='conv5_1/expand')
            print(conv5_1.name, conv5_1.get_shape())
            conv5_1 = slim.separable_convolution2d(conv5_1, num_outputs=None, stride=2, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_1/dwise')
            print(conv5_1.name, conv5_1.get_shape())
            conv5_1 = slim.convolution2d(conv5_1, 64, [1, 1], stride=1,activation_fn=None,
                                         scope='conv5_1/linear')
            print(conv5_1.name, conv5_1.get_shape())

            conv5_2 = slim.convolution2d(conv5_1, 384, [1, 1], stride=1, scope='conv5_2/expand')
            print(conv5_2.name, conv5_2.get_shape())
            conv5_2 = slim.separable_convolution2d(conv5_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_2/dwise')
            print(conv5_2.name, conv5_2.get_shape())
            conv5_2 = slim.convolution2d(conv5_2, 64, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_2/linear')
            print(conv5_2.name, conv5_2.get_shape())
            block_5_2 = conv5_1 + conv5_2
            print(block_5_2.name, block_5_2.get_shape())

            conv5_3 = slim.convolution2d(block_5_2, 384, [1, 1], stride=1, scope='conv5_3/expand')
            print(conv5_3.name, conv5_3.get_shape())
            conv5_3 = slim.separable_convolution2d(conv5_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_3/dwise')
            print(conv5_3.name, conv5_3.get_shape())
            conv5_3 = slim.convolution2d(conv5_3, 64, [1, 1], stride=1,  activation_fn=None,
                                         scope='conv5_3/linear')
            print(conv5_3.name, conv5_3.get_shape())
            block_5_3 = block_5_2 + conv5_3
            print(block_5_3.name, block_5_3.get_shape())

            conv5_4 = slim.convolution2d(block_5_3, 384, [1, 1], stride=1, scope='conv5_4/expand')
            print(conv5_4.name, conv5_4.get_shape())
            conv5_4 = slim.separable_convolution2d(conv5_4, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_4/dwise')
            print(conv5_4.name, conv5_4.get_shape())
            conv5_4 = slim.convolution2d(conv5_4, 64, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_4/linear')
            print(conv5_4.name, conv5_4.get_shape())
            block_5_4 = block_5_3 + conv5_4
            print(block_5_4.name, block_5_4.get_shape())

            # 6*6*64    7*7*64
            conv6_1 = slim.convolution2d(block_5_4, 384, [1, 1], stride=1, scope='conv6_1/expand')
            print(conv6_1.name, conv6_1.get_shape())
            conv6_1 = slim.separable_convolution2d(conv6_1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv6_1/dwise')
            print(conv6_1.name, conv6_1.get_shape())
            conv6_1 = slim.convolution2d(conv6_1, 96, [1, 1], stride=1, activation_fn=None,
                                         scope='conv6_1/linear')
            print(conv6_1.name, conv6_1.get_shape())

            conv6_2 = slim.convolution2d(conv6_1, 576, [1, 1], stride=1, scope='conv6_2/expand')
            print(conv6_2.name, conv6_2.get_shape())
            conv6_2 = slim.separable_convolution2d(conv6_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv6_2/dwise')
            print(conv6_2.name, conv6_2.get_shape())
            conv6_2 = slim.convolution2d(conv6_2, 96, [1, 1], stride=1, activation_fn=None,
                                         scope='conv6_2/linear')
            print(conv6_2.name, conv6_2.get_shape())
            block_6_2 = conv6_1 + conv6_2
            print(block_6_2.name, block_6_2.get_shape())

            conv6_3 = slim.convolution2d(block_6_2, 576, [1, 1], stride=1, scope='conv6_3/expand')
            print(conv6_3.name, conv6_3.get_shape())
            conv6_3 = slim.separable_convolution2d(conv6_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv6_3/dwise')
            print(conv6_3.name, conv6_3.get_shape())
            conv6_3 = slim.convolution2d(conv6_3, 96, [1, 1], stride=1, activation_fn=None,
                                         scope='conv6_3/linear')
            print(conv6_3.name, conv6_3.get_shape())
            block_6_3 = block_6_2 + conv6_3
            print(block_6_3.name, block_6_3.get_shape())

            features['feature5'] = block_6_3
            # 6*6*96    7*7*96
            conv7_1 = slim.convolution2d(block_6_3, 576, [1, 1], stride=1, scope='conv7_1/expand')
            print(conv7_1.name, conv7_1.get_shape())
            conv7_1 = slim.separable_convolution2d(conv7_1, num_outputs=None, stride=2, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv7_1/dwise')
            print(conv7_1.name, conv7_1.get_shape())
            conv7_1 = slim.convolution2d(conv7_1, 160, [1, 1], stride=1, activation_fn=None,
                                         scope='conv7_1/linear')
            print(conv7_1.name, conv7_1.get_shape())

            conv7_2 = slim.convolution2d(conv7_1, 960, [1, 1], stride=1, scope='conv7_2/expand')
            print(conv7_2.name, conv7_2.get_shape())
            conv7_2 = slim.separable_convolution2d(conv7_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv7_2/dwise')
            print(conv7_2.name, conv7_2.get_shape())
            conv7_2 = slim.convolution2d(conv7_2, 160, [1, 1], stride=1, activation_fn=None,
                                         scope='conv7_2/linear')
            print(conv7_2.name, conv7_2.get_shape())
            block_7_2 = conv7_1 + conv7_2
            print(block_7_2.name, block_7_2.get_shape())


            conv7_3 = slim.convolution2d(block_7_2, 960, [1, 1], stride=1, scope='conv7_3/expand')
            print(conv7_3.name, conv7_3.get_shape())
            conv7_3 = slim.separable_convolution2d(conv7_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv7_3/dwise')
            print(conv7_3.name, conv7_3.get_shape())
            conv7_3 = slim.convolution2d(conv7_3, 160, [1, 1], stride=1, activation_fn=None,
                                         scope='conv7_3/linear')
            print(conv7_3.name, conv7_3.get_shape())
            block_7_3 = block_7_2 + conv7_3
            print(block_7_3.name, block_7_3.get_shape())

            conv7_4 = slim.convolution2d(block_7_3, 960, [1, 1], stride=1, scope='conv7_4/expand')
            print(conv7_4.name, conv7_4.get_shape())
            conv7_4 = slim.separable_convolution2d(conv7_4, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv7_4/dwise')
            print(conv7_4.name, conv7_4.get_shape())
            conv7_4 = slim.convolution2d(conv7_4, 320, [1, 1], stride=1, activation_fn=None,
                                         scope='conv7_4/linear')
            print(conv7_4.name, conv7_4.get_shape())
            features['feature6'] = conv7_4
    return features


# -------------------------------------------------------------------------
def conv_bn_relu(input, out_channel, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'conv_bn_relu'):
        input = slim.convolution2d(input, out_channel, kernel_size, stride, rate=dilation, activation_fn=None)
        input = slim.batch_norm(input, activation_fn=tf.nn.relu, fused=False)
    return input


def depthwise_conv_bn(input, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'depthwise_conv_bn'):
        input = slim.separable_conv2d(input, None, kernel_size, depth_multiplier=1, stride=stride, rate=dilation,
                                      activation_fn=None)
        input = slim.batch_norm(input, activation_fn=None, fused=False)
    return input


def shuffle_unit(input, groups):
    with tf.variable_scope('shuffle_unit'):
        n, h, w, c = input.get_shape().as_list()
        input = tf.reshape(input, shape=tf.convert_to_tensor([tf.shape(input)[0], h, w, groups, c // groups]))
        input = tf.transpose(input, tf.convert_to_tensor([0, 1, 2, 4, 3]))
        input = tf.reshape(input, shape=tf.convert_to_tensor([tf.shape(input)[0], h, w, c]))
    return input


def shufflenet_v2_block(input, out_channel, stride=1, dilation=1, shuffle_group=2):
    with tf.variable_scope(None, 'shuffle_v2_block'):
        half_channel = out_channel // 2
        if stride == 1:
            top, bottom = tf.split(input, num_or_size_splits=2, axis=3)

            top = conv_bn_relu(top, half_channel, [1, 1], stride=1)
            top = depthwise_conv_bn(top, [3, 3], stride=stride, dilation=dilation)
            top = conv_bn_relu(top, half_channel, [1, 1], stride=1)

            out = tf.concat([top, bottom], axis=3)
            out = shuffle_unit(out, shuffle_group)
        else:
            # 网络右分支
            b0 = conv_bn_relu(input, half_channel, [1, 1], stride=1)
            b0 = depthwise_conv_bn(b0, [3, 3], stride=stride, dilation=dilation)
            b0 = conv_bn_relu(b0, half_channel, [1, 1], stride=1)
            # 网络左分支
            b1 = depthwise_conv_bn(input, [3, 3], stride=stride, dilation=dilation)
            b1 = conv_bn_relu(b1, half_channel, [1, 1], stride=1)

            out = tf.concat([b0, b1], axis=3)
            out = shuffle_unit(out, shuffle_group)
        return out


def pfld_inference_for_shuffleNetV2(input, weight_decay, shuffle_group=2):
    # [(out_channel, repeat_times), (out_channel, repeat_times), ...]
    # # model_scale = 0.5
    # channel_sizes = [(48, 4), (96, 8), (192, 4), (1024, 1)]

    # model_scale = 1.0
    channel_sizes = [(116, 4), (232, 8), (464, 4), (1024, 1)]

    # # model_scale = 1.5
    # channel_sizes = [(176, 4), (352, 8), (704, 4), (1024, 1)]
    #
    # # model_scale = 2.0
    # channel_sizes = [(244, 4), (488, 8), (976, 4), (2048, 1)]

    with tf.variable_scope('pfld_inference'):
        features = {}
        with slim.arg_scope([slim.convolution2d, slim.separable_conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            padding='SAME'):
            print('PFLD input shape({}): {}'.format(input.name, input.get_shape()))
            # 112*112*3=>56*56*24
            with tf.variable_scope('conv_1'):
                conv1 = conv_bn_relu(input, 24, [3, 3], stride=2)
            print(conv1.name, conv1.get_shape())

            # 实现stage2:56*56*24=>28*28*C
            with tf.variable_scope('shuffle_block_1'):
                out_channel, repeate_times = channel_sizes[0]
                shuffle_block_1 = shufflenet_v2_block(conv1, out_channel, stride=2, shuffle_group=shuffle_group)
                print(shuffle_block_1.name, shuffle_block_1.get_shape())
                for i in range(repeate_times - 1):
                    shuffle_block_1 = shufflenet_v2_block(shuffle_block_1, out_channel, stride=1,
                                                          shuffle_group=shuffle_group)
                    print(shuffle_block_1.name, shuffle_block_1.get_shape())

            features['auxiliary_input'] = shuffle_block_1

            # 实现stage3:28*28*C=>14*14*C
            with tf.variable_scope('shuffle_block_2'):
                out_channel, repeate_times = channel_sizes[1]
                shuffle_block_2 = shufflenet_v2_block(shuffle_block_1, out_channel, stride=2, shuffle_group=shuffle_group)
                print(shuffle_block_2.name, shuffle_block_2.get_shape())
                for i in range(repeate_times - 1):
                    shuffle_block_2 = shufflenet_v2_block(shuffle_block_2, out_channel, stride=1,
                                                          shuffle_group=shuffle_group)
                    print(shuffle_block_2.name, shuffle_block_2.get_shape())

            # 实现stage4:14*14*C=>7*7*C
            with tf.variable_scope('shuffle_block_3'):
                out_channel, repeate_times = channel_sizes[2]
                shuffle_block_3 = shufflenet_v2_block(shuffle_block_2, out_channel, stride=2,
                                                      shuffle_group=shuffle_group)
                print(shuffle_block_3.name, shuffle_block_3.get_shape())
                for i in range(repeate_times - 1):
                    shuffle_block_3 = shufflenet_v2_block(shuffle_block_3, out_channel, stride=1,
                                                          shuffle_group=shuffle_group)
                    print(shuffle_block_3.name, shuffle_block_3.get_shape())

            # 7*7*C=>1*1*C
            with tf.variable_scope('end_conv'):
                with slim.arg_scope([slim.convolution2d], padding='valid'):
                    out_channel = channel_sizes[-1][0]
                    end_conv = conv_bn_relu(shuffle_block_3, out_channel, [3, 3], stride=1)
                    print(end_conv.name, end_conv.get_shape())
                    end_conv = conv_bn_relu(end_conv, out_channel, [3, 3], stride=1)
                    print(end_conv.name, end_conv.get_shape())
                    end_conv = conv_bn_relu(end_conv, out_channel, [3, 3], stride=1,)
                    print(end_conv.name, end_conv.get_shape())

            group_pool1 = slim.avg_pool2d(shuffle_block_2, [shuffle_block_2.get_shape()[1],
                                                            shuffle_block_2.get_shape()[2]], stride=1)
            print(group_pool1.name, group_pool1.get_shape())
            group_pool2 = slim.avg_pool2d(shuffle_block_3, [shuffle_block_3.get_shape()[1],
                                                            shuffle_block_3.get_shape()[2]], stride=1)
            print(group_pool2.name, group_pool2.get_shape())
            group_pool3 = slim.avg_pool2d(end_conv, [end_conv.get_shape()[1], end_conv.get_shape()[2]], stride=1)
            print(group_pool3.name, group_pool3.get_shape())

            s1 = slim.flatten(group_pool1)
            s2 = slim.flatten(group_pool2)
            s3 = slim.flatten(group_pool3)

            multi_scale = tf.concat([s1, s2, s3], 1)
            landmarks = slim.fully_connected(multi_scale, num_outputs=196, activation_fn=None, scope='fc')
            print(landmarks.name, landmarks.get_shape())

        return features, landmarks
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
def hard_sigmoid(input, name='hard_sigmoid'):
    with tf.variable_scope(name):
        h_sigmoid = tf.nn.relu6(input + 3) / 6
    return h_sigmoid


def hard_swish(input, name='hard_swish'):
    with tf.variable_scope(name):
        h_swish = input * tf.nn.relu6(input + 3) / 6
    return h_swish


def relu6(input, name='relu6'):
    return tf.nn.relu6(input, name)


def squeeze_and_excite(input, out_dims, ratio, layer_name='squeeze_and_excite'):
    with tf.variable_scope(layer_name):
        squeeze = slim.avg_pool2d(input, [input.get_shape()[1], input.get_shape()[2]], stride=1,
                                  scope=layer_name+'group_pool2d')
        excitation = slim.fully_connected(squeeze, num_outputs=out_dims // ratio, activation_fn=relu6,
                                          scope=layer_name+'_excitation1')
        excitation = slim.fully_connected(excitation, num_outputs=out_dims, activation_fn=hard_sigmoid,
                                          scope=layer_name+'_excitation2')
        excitation = tf.reshape(excitation, [-1, 1, 1, out_dims])
        scale = input * excitation
    return scale


def mobileNetV3_block(input, layer_name, expand_dims, out_dims, kernel, stride, ratio, activation_fn='RE', se=False,
                      short_cut=True):
    with tf.variable_scope(layer_name):
        net = slim.convolution2d(input, expand_dims, [1, 1], stride=1, activation_fn=None,
                                 scope=layer_name+'_pw_expand')

        if activation_fn == 'RE':
            net = slim.separable_convolution2d(net, num_outputs=None, stride=stride, activation_fn=relu6,
                                               depth_multiplier=1, kernel_size=kernel, scope=layer_name+'_dwise')
        elif activation_fn == 'HS':
            net = slim.separable_convolution2d(net, num_outputs=None, stride=stride, activation_fn=hard_swish,
                                               depth_multiplier=1, kernel_size=kernel, scope=layer_name+'_dwise')
        else:
            raise NotImplementedError

        if se is True:
            channel = net.get_shape().as_list()[-1]
            net = squeeze_and_excite(net, out_dims=channel, ratio=ratio, layer_name=layer_name+'se')

        # if activation_fn == 'RE':
        #     net = slim.convolution2d(net, out_dims, [1, 1], stride=1, activation_fn=relu6,
        #                              scope=layer_name+'_pw_reduce')
        # elif activation_fn == 'HS':
        #     net = slim.convolution2d(net, out_dims, [1, 1], stride=1, activation_fn=hard_swish,
        #                              scope=layer_name+'_pw_reduce')
        # else:
        #     raise NotImplementedError
        net = slim.convolution2d(net, out_dims, [1, 1], stride=1, activation_fn=None,
                                 scope=layer_name + '_pw_reduce')
        if stride == 1 and short_cut is True:
            net = net + input
    return net


def pfld_inference_for_mobileNetV3_large(input, weight_decay, batch_norm_params):
    layers = [
        [16, 16, 3, 1, "RE", False, 16],
        [16, 24, 3, 2, "RE", False, 64],
        [24, 24, 3, 1, "RE", False, 72],
        [24, 40, 5, 2, "RE", True, 72],
        [40, 40, 5, 1, "RE", True, 120],

        [40, 40, 5, 1, "RE", True, 120],
        [40, 80, 3, 2, "HS", False, 240],
        [80, 80, 3, 1, "HS", False, 200],
        [80, 80, 3, 1, "HS", False, 184],
        [80, 80, 3, 1, "HS", False, 184],

        [80, 112, 3, 1, "HS", True, 480],
        [112, 112, 3, 1, "HS", True, 672],
        [112, 160, 5, 2, "HS", True, 672],
        [160, 160, 5, 1, "HS", True, 960],
        [160, 160, 5, 1, "HS", True, 960],
    ]
    reduction_ratio = 4
    multiplier = 1
    with tf.variable_scope('pfld_inference'):
        features = {}
        with slim.arg_scope([slim.convolution2d, slim.separable_conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            padding='SAME'):
            print('PFLD input shape({}): {}'.format(input.name, input.get_shape()))

            # 112*112*3
            out = slim.convolution2d(input, 16 * multiplier, [3, 3], stride=1, activation_fn=hard_swish, scope='conv_1')
            print(out.name, out.get_shape())

            with tf.variable_scope("MobilenetV3_large"):
                for index in range(6):
                    in_channels, out_channels, kernel_size, stride, activatation, se, expand_dims = layers[index]
                    out_channels *= multiplier
                    out = mobileNetV3_block(out, "bneck{}".format(index), expand_dims, out_channels, kernel_size,
                                            stride, ratio=reduction_ratio, activation_fn=activatation, se=se,
                                            short_cut=(in_channels == out_channels))
                    print(out.name, out.get_shape())

                # 28*28
                features['auxiliary_input'] = out

                # 14*14
                index = 6
                in_channels, out_channels, kernel_size, stride, activatation, se, expand_dims = layers[index]
                out_channels *= multiplier
                out1 = mobileNetV3_block(out, "bneck{}".format(index), expand_dims, out_channels, kernel_size,
                                         stride, ratio=reduction_ratio, activation_fn=activatation, se=se,
                                         short_cut=(in_channels == out_channels))
                print(out1.name, out1.get_shape())
                for index in range(7, 12):
                    in_channels, out_channels, kernel_size, stride, activatation, se, expand_dims = layers[index]
                    out_channels *= multiplier
                    out1 = mobileNetV3_block(out1, "bneck{}".format(index), expand_dims, out_channels, kernel_size,
                                             stride, ratio=reduction_ratio, activation_fn=activatation, se=se,
                                            short_cut=(in_channels == out_channels))
                    print(out1.name, out1.get_shape())

                # 7*7
                index = 12
                in_channels, out_channels, kernel_size, stride, activatation, se, expand_dims = layers[index]
                out_channels *= multiplier
                out2 = mobileNetV3_block(out1, "bneck{}".format(index), expand_dims, out_channels, kernel_size, stride,
                                         ratio=reduction_ratio, activation_fn=activatation, se=se,
                                         short_cut=(in_channels == out_channels))
                print(out2.name, out2.get_shape())
                for index in range(13, len(layers)):
                    in_channels, out_channels, kernel_size, stride, activatation, se, expand_dims = layers[index]
                    out_channels *= multiplier
                    out2 = mobileNetV3_block(out2, "bneck{}".format(index), expand_dims, out_channels, kernel_size,
                                             stride, ratio=reduction_ratio, activation_fn=activatation, se=se,
                                             short_cut=(in_channels == out_channels))
                    print(out2.name, out2.get_shape())

                out3 = slim.convolution2d(out2, 960, [7, 7], stride=1, activation_fn=hard_swish, padding='valid',
                                          scope='conv_2')
                print(out3.name, out3.get_shape())

                out3 = slim.convolution2d(out3, 1280, [1, 1], stride=1, normalizer_fn=None, activation_fn=hard_swish,
                                          scope='conv_3')
                print(out3.name, out3.get_shape())

            s1 = slim.flatten(out1)
            s2 = slim.flatten(out2)
            s3 = slim.flatten(out3)
            multi_scale = tf.concat([s1, s2, s3], 1)
            landmarks = slim.fully_connected(multi_scale, num_outputs=136, activation_fn=None, scope='fc')
            print(landmarks.name, landmarks.get_shape())
        return features, landmarks


def pfld_inference_for_mobileNetV3_small(input, weight_decay, batch_norm_params):
    layers = [
        [16, 16, 3, 2, "RE", True, 16],
        [16, 24, 3, 2, "RE", False, 72],
        [24, 24, 3, 1, "RE", False, 88],
        [24, 40, 5, 2, "HS", True, 96],
        [40, 40, 5, 1, "HS", True, 240],
        [40, 40, 5, 1, "HS", True, 240],
        [40, 48, 5, 1, "HS", True, 120],
        [48, 48, 5, 1, "HS", True, 144],
        [48, 96, 5, 2, "HS", True, 288],
        [96, 96, 5, 1, "HS", True, 576],
        [96, 96, 5, 1, "HS", True, 576],
    ]
    reduction_ratio = 4
    multiplier = 1
    with tf.variable_scope('pfld_inference'):
        features = {}
        with slim.arg_scope([slim.convolution2d, slim.separable_conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            padding='SAME'):
            print('PFLD input shape({}): {}'.format(input.name, input.get_shape()))

            # 112*112*3
            out = slim.convolution2d(input, 16 * multiplier, [3, 3], stride=1, activation_fn=hard_swish, scope='conv_1')
            print(out.name, out.get_shape())

            with tf.variable_scope("MobilenetV3_large"):
                for index in range(3):
                    in_channels, out_channels, kernel_size, stride, activatation, se, expand_dims = layers[index]
                    out_channels *= multiplier
                    out = mobileNetV3_block(out, "bneck{}".format(index), expand_dims, out_channels, kernel_size,
                                            stride, ratio=reduction_ratio, activation_fn=activatation, se=se,
                                            short_cut=(in_channels == out_channels))
                    print(out.name, out.get_shape())

                # 28*28
                features['auxiliary_input'] = out

                # 14*14
                index = 3
                in_channels, out_channels, kernel_size, stride, activatation, se, expand_dims = layers[index]
                out_channels *= multiplier
                out1 = mobileNetV3_block(out, "bneck{}".format(index), expand_dims, out_channels, kernel_size,
                                         stride, ratio=reduction_ratio, activation_fn=activatation, se=se,
                                         short_cut=(in_channels == out_channels))
                print(out1.name, out1.get_shape())
                for index in range(4, 8):
                    in_channels, out_channels, kernel_size, stride, activatation, se, expand_dims = layers[index]
                    out_channels *= multiplier
                    out1 = mobileNetV3_block(out1, "bneck{}".format(index), expand_dims, out_channels, kernel_size,
                                             stride, ratio=reduction_ratio, activation_fn=activatation, se=se,
                                             short_cut=(in_channels == out_channels))
                    print(out1.name, out1.get_shape())

                # 7*7
                index = 8
                in_channels, out_channels, kernel_size, stride, activatation, se, expand_dims = layers[index]
                out_channels *= multiplier
                out2 = mobileNetV3_block(out1, "bneck{}".format(index), expand_dims, out_channels, kernel_size, stride,
                                         ratio=reduction_ratio, activation_fn=activatation, se=se,
                                         short_cut=(in_channels == out_channels))
                print(out2.name, out2.get_shape())
                for index in range(9, len(layers)):
                    in_channels, out_channels, kernel_size, stride, activatation, se, expand_dims = layers[index]
                    out_channels *= multiplier
                    out2 = mobileNetV3_block(out2, "bneck{}".format(index), expand_dims, out_channels, kernel_size,
                                             stride, ratio=reduction_ratio, activation_fn=activatation, se=se,
                                             short_cut=(in_channels == out_channels))
                    print(out2.name, out2.get_shape())

                out3 = slim.convolution2d(out2, 576, [1, 1], stride=1, activation_fn=hard_swish, dscope='conv_2')
                print(out3.name, out3.get_shape())

                out3 = slim.avg_pool2d(out3, [out3.get_shape()[1], out3.get_shape()[2]], stride=1, scope='group_pool')
                print(out3.name, out3.get_shape())

                out3 = slim.convolution2d(out3, 1280, [1, 1], stride=1, normalizer_fn=None, activation_fn=hard_swish,
                                          scope='conv_3')
                print(out3.name, out3.get_shape())

            s1 = slim.flatten(out1)
            s2 = slim.flatten(out2)
            s3 = slim.flatten(out3)
            multi_scale = tf.concat([s1, s2, s3], 1)
            landmarks = slim.fully_connected(multi_scale, num_outputs=136, activation_fn=None, scope='fc')
            print(landmarks.name, landmarks.get_shape())
        return features, landmarks


# -------------------------------------------------------------------------
def pfld_inference(input, weight_decay, batch_norm_params):

    coefficient = 1
    with tf.variable_scope('pfld_inference'):
        features = {}
        with slim.arg_scope([slim.convolution2d, slim.separable_conv2d],
                            activation_fn=tf.nn.relu6,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            padding='SAME'):
            print('PFLD input shape({}): {}'.format(input.name, input.get_shape()))
            # 112*112*3
            conv1 = slim.convolution2d(input, 64*coefficient, [3, 3], stride=2, scope='conv_1')
            print(conv1.name, conv1.get_shape())

            # 56*56*64
            conv2 = slim.separable_convolution2d(conv1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv2/dwise')
            print(conv2.name, conv2.get_shape())

            # 56*56*64
            conv3_1 = slim.convolution2d(conv2, 128, [1, 1], stride=2, activation_fn=None, scope='conv3_1/expand')
            print(conv3_1.name, conv3_1.get_shape())
            conv3_1 = slim.separable_convolution2d(conv3_1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_1/dwise')
            print(conv3_1.name, conv3_1.get_shape())
            conv3_1 = slim.convolution2d(conv3_1, 64*coefficient, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_1/linear')
            print(conv3_1.name, conv3_1.get_shape())

            conv3_2 = slim.convolution2d(conv3_1, 128, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_2/expand')
            print(conv3_2.name, conv3_2.get_shape())
            conv3_2 = slim.separable_convolution2d(conv3_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_2/dwise')
            print(conv3_2.name, conv3_2.get_shape())
            conv3_2 = slim.convolution2d(conv3_1, 64*coefficient, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_2/linear')
            print(conv3_2.name, conv3_2.get_shape())

            block3_2 = conv3_1 + conv3_2
            print(block3_2.name, block3_2.get_shape())

            conv3_3 = slim.convolution2d(block3_2, 128, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_3/expand')
            print(conv3_3.name, conv3_3.get_shape())
            conv3_3 = slim.separable_convolution2d(conv3_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_3/dwise')
            print(conv3_3.name, conv3_3.get_shape())
            conv3_3 = slim.convolution2d(conv3_3, 64*coefficient, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_3linear')
            print(conv3_3.name, conv3_3.get_shape())

            block3_3 = block3_2 + conv3_3
            print(block3_3.name, block3_3.get_shape())

            conv3_4 = slim.convolution2d(block3_3, 128, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_4/expand')
            print(conv3_4.name, conv3_4.get_shape())
            conv3_4 = slim.separable_convolution2d(conv3_4, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_4/dwise')
            print(conv3_4.name, conv3_4.get_shape())
            conv3_4 = slim.convolution2d(conv3_4, 64*coefficient, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_4/linear')
            print(conv3_4.name, conv3_4.get_shape())

            block3_4 = block3_3 + conv3_4
            print(block3_4.name, block3_4.get_shape())

            conv3_5 = slim.convolution2d(block3_4, 128, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_5/expand')
            print(conv3_5.name, conv3_5.get_shape())
            conv3_5 = slim.separable_convolution2d(conv3_5, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_5/dwise')
            print(conv3_5.name, conv3_5.get_shape())
            conv3_5 = slim.convolution2d(conv3_5, 64*coefficient, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_5/linear')
            print(conv3_5.name, conv3_5.get_shape())

            block3_5 = block3_4 + conv3_5
            print(block3_5.name, block3_5.get_shape())

            features['auxiliary_input'] = block3_5

            #28*28*64
            conv4_1 = slim.convolution2d(block3_5, 128, [1, 1], stride=2, activation_fn=None,
                                         scope='conv4_1/expand')
            print(conv4_1.name, conv4_1.get_shape())
            conv4_1 = slim.separable_convolution2d(conv4_1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv4_1/dwise')
            print(conv4_1.name, conv4_1.get_shape())
            conv4_1 = slim.convolution2d(conv4_1, 128*coefficient, [1, 1], stride=1, activation_fn=None,
                                         scope='conv4_1/linear')
            print(conv4_1.name, conv4_1.get_shape())

            #14*14*128
            conv5_1 = slim.convolution2d(conv4_1, 512, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_1/expand')
            print(conv5_1.name, conv5_1.get_shape())
            conv5_1 = slim.separable_convolution2d(conv5_1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_1/dwise')
            print(conv5_1.name, conv5_1.get_shape())
            conv5_1 = slim.convolution2d(conv5_1, 128*coefficient, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_1/linear')
            print(conv5_1.name, conv5_1.get_shape())

            conv5_2 = slim.convolution2d(conv5_1, 512, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_2/expand')
            print(conv5_2.name, conv5_2.get_shape())
            conv5_2 = slim.separable_convolution2d(conv5_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_2/dwise')
            print(conv5_2.name, conv5_2.get_shape())
            conv5_2 = slim.convolution2d(conv5_2, 128*coefficient, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_2/linear')
            print(conv5_2.name, conv5_2.get_shape())

            block5_2 = conv5_1 + conv5_2
            print(block5_2.name, block5_2.get_shape())

            conv5_3 = slim.convolution2d(block5_2, 512, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_3/expand')
            print(conv5_3.name, conv5_3.get_shape())
            conv5_3 = slim.separable_convolution2d(conv5_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_3/dwise')
            print(conv5_3.name, conv5_3.get_shape())
            conv5_3 = slim.convolution2d(conv5_3, 128*coefficient, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_3/linear')
            print(conv5_3.name, conv5_3.get_shape())

            block5_3 = block5_2 + conv5_3
            print(block5_3.name, block5_3.get_shape())

            conv5_4 = slim.convolution2d(block5_3, 512, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_4/expand')
            print(conv5_4.name, conv5_4.get_shape())
            conv5_4 = slim.separable_convolution2d(conv5_4, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_4/dwise')
            print(conv5_4.name, conv5_4.get_shape())
            conv5_4 = slim.convolution2d(conv5_4, 128*coefficient, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_4/linear')
            print(conv5_4.name, conv5_4.get_shape())

            block5_4 = block5_3 + conv5_4
            print(block5_4.name, block5_4.get_shape())

            conv5_5 = slim.convolution2d(block5_4, 512, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_5/expand')
            print(conv5_5.name, conv5_5.get_shape())
            conv5_5 = slim.separable_convolution2d(conv5_5, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_5/dwise')
            print(conv5_5.name, conv5_5.get_shape())
            conv5_5 = slim.convolution2d(conv5_5, 128*coefficient, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_5/linear')
            print(conv5_5.name, conv5_5.get_shape())

            block5_5 = block5_4 + conv5_5
            print(block5_5.name, block5_5.get_shape())

            conv5_6 = slim.convolution2d(block5_5, 512, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_6/expand')
            print(conv5_6.name, conv5_6.get_shape())
            conv5_6 = slim.separable_convolution2d(conv5_6, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_6/dwise')
            print(conv5_6.name, conv5_6.get_shape())
            conv5_6 = slim.convolution2d(conv5_6, 128*coefficient, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_6/linear')
            print(conv5_6.name, conv5_6.get_shape())

            block5_6 = block5_5 + conv5_6
            print(block5_6.name, block5_6.get_shape())

            #14*14*128
            conv6_1 = slim.convolution2d(block5_6, 256, [1, 1], stride=1, activation_fn=None, scope='conv6_1/expand')
            print(conv6_1.name, conv6_1.get_shape())
            conv6_1 = slim.separable_convolution2d(conv6_1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv6_1/dwise')
            print(conv6_1.name, conv6_1.get_shape())
            conv6_1 = slim.convolution2d(conv6_1, 16*coefficient, [1, 1], stride=1, activation_fn=None,
                                         scope='conv6_1/linear')
            print(conv6_1.name, conv6_1.get_shape())

            #14*14*16
            conv7 = slim.convolution2d(conv6_1, 32*coefficient, [3, 3], stride=2, activation_fn=None, scope='conv7')
            print(conv7.name, conv7.get_shape())

            #7*7*32
            conv8 = slim.convolution2d(conv7, 128*coefficient, [7, 7], stride=1, activation_fn=None, padding='valid',
                                       scope='conv8')
            print(conv8.name, conv8.get_shape())

            # avg_pool1 = slim.avg_pool2d(conv6_1, [conv6_1.get_shape()[1], conv6_1.get_shape()[2]], stride=1)
            # print(avg_pool1.name, avg_pool1.get_shape())
            #
            # avg_pool2 = slim.avg_pool2d(conv7,[conv7.get_shape()[1],conv7.get_shape()[2]],stride=1)
            # print(avg_pool2.name,avg_pool2.get_shape())
            #
            # s1 = slim.flatten(avg_pool1)
            # s2 = slim.flatten(avg_pool2)
            s1 = slim.flatten(conv6_1)
            s2 = slim.flatten(conv7)
            #1*1*128
            s3 = slim.flatten(conv8)
            multi_scale = tf.concat([s1, s2, s3], 1)
            landmarks = slim.fully_connected(multi_scale, num_outputs=136, activation_fn=None, scope='fc')
            print(landmarks.name, landmarks.get_shape())
        return features, landmarks


# def create_model(input, landmark, phase_train, args):
#     batch_norm_params = {
#         'decay': 0.995,
#         'epsilon': 0.001,
#         'updates_collections':  None,  # tf.GraphKeys.UPDATE_OPS,
#         'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
#         'is_training': phase_train
#     }
#
#     landmark_dim = int(landmark.get_shape()[-1])
#     # features, landmarks_pre = pfld_inference(input, args.weight_decay, batch_norm_params)
#     # features, landmarks_pre = pfld_inference_for_shuffleNetV2(input, args.weight_decay)
#     # features, landmarks_pre = pfld_inference_for_mobileNetV3_large(input, args.weight_decay, batch_norm_params)
#     features, landmarks_pre = pfld_inference_for_mobileNetV3_small(input, args.weight_decay, batch_norm_params)
#     # loss
#     # landmarks_loss = tf.reduce_sum(tf.square(landmarks_pre - landmark), axis=1)
#     # landmarks_loss = tf.reduce_mean(landmarks_loss)
#
#     # add the auxiliary net
#     # : finish the loss function
#     print('\nauxiliary net')
#     with slim.arg_scope([slim.convolution2d, slim.fully_connected],
#                         activation_fn=tf.nn.relu,
#                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
#                         biases_initializer=tf.zeros_initializer(),
#                         weights_regularizer=slim.l2_regularizer(args.weight_decay),
#                         normalizer_fn=slim.batch_norm,
#                         normalizer_params=batch_norm_params):
#         pfld_input = features['auxiliary_input']
#         net_aux = slim.convolution2d(pfld_input, 128, [3, 3], stride=2, scope='pfld_conv1')
#         print(net_aux.name, net_aux.get_shape())
#         # net = slim.max_pool2d(net, kernel_size=[3, 3], stride=1, scope='pool1', padding='SAME')
#         net_aux = slim.convolution2d(net_aux, 128, [3, 3], stride=1, scope='pfld_conv2')
#         print(net_aux.name, net_aux.get_shape())
#         net_aux = slim.convolution2d(net_aux, 32, [3, 3], stride=2, scope='pfld_conv3')
#         print(net_aux.name, net_aux.get_shape())
#         net_aux = slim.convolution2d(net_aux, 128, [7, 7], stride=1, scope='pfld_conv4')
#         print(net_aux.name, net_aux.get_shape())
#         net_aux = slim.max_pool2d(net_aux, kernel_size=[3, 3], stride=1, scope='pool1', padding='SAME')
#         print(net_aux.name, net_aux.get_shape())
#         net_aux = slim.flatten(net_aux)
#         print(net_aux.name, net_aux.get_shape())
#         fc1 = slim.fully_connected(net_aux, num_outputs=32, activation_fn=None, scope='pfld_fc1')
#         print(fc1.name, fc1.get_shape())
#         euler_angles_pre = slim.fully_connected(fc1, num_outputs=3, activation_fn=None, scope='pfld_fc2')
#         print(euler_angles_pre.name, euler_angles_pre.get_shape())
#
#     # return landmarks_loss, landmarks, heatmap_loss, HeatMaps
#     # return landmarks_pre, landmarks_loss, euler_angles_pre
#     return landmarks_pre, euler_angles_pre

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

        #self.in_features = 48 + 96 + 192

        self.in_features = 192

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

        block_out3 = self.secondconv(self.doubleBlazeBlock2(block_out2))
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
        block_out3_ = functional.adaptive_avg_pool2d(block_out3, 1).squeeze(-1).squeeze(-1)
        #print(block_out3_.size())
        assert block_out3_.size(1) == self.in_features
        pre_landmarks = self.fc(block_out3_)
        
        return pre_landmarks, block_out1


class MBConvBlock(nn.Module):

    def __init__(self):
        super(MBConvBlock, self).__init__()


class EfficientNet(nn.Module):
    def __init__(self, nums_class=136):
        super(EfficientNet, self).__init__()

        self.firstconv = nn.Sequential(

        )

        self.in_features = 1280

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

    def extract_features(self, input):
        return input

    def forward(self, input):

        features = self.extract_features(input)

        features = functional.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)

        assert features.size(1) == self.in_features
        pre_landmarks = self.fc(features)

        return pre_landmarks




