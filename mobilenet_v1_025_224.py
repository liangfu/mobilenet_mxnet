import mxnet as mx
import numpy as np
import math
from pprint import pprint
import json

# mxnet-cpu only support channel first, default convert the model and weight as channel first

def RefactorModel():

    input           = mx.sym.var('input')
    Conv2d_0_Conv2D_pad = mx.sym.pad(data = input, mode = 'constant', pad_width=(0, 0, 0, 0, 0L, 1L, 0L, 1L), constant_value = 0.0, name = 'MobilenetV1/MobilenetV1/Conv2d_0/Conv2D_pad')
    Conv2d_0_Conv2D = mx.sym.Convolution(data=Conv2d_0_Conv2D_pad, kernel=(3L, 3L), stride=(2L, 2L), dilate = (), num_filter = 8, num_group = 1, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_0/Conv2D')
    Conv2d_0_BatchNorm = mx.sym.BatchNorm(data = Conv2d_0_Conv2D, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm')
    Conv2d_0_Relu6 = mx.sym.Activation(data = Conv2d_0_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_0/Relu6')
    Conv2d_1_depthwise_depthwise = mx.sym.Convolution(data=Conv2d_0_Relu6, kernel=(3L, 3L), stride=(1L, 1L), dilate = (), pad=(1L, 1L), num_filter = 8, num_group = 8, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise')
    Conv2d_1_depthwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_1_depthwise_depthwise, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNorm')
    Conv2d_1_depthwise_Relu6 = mx.sym.Activation(data = Conv2d_1_depthwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6')
    Conv2d_1_pointwise_Conv2D = mx.sym.Convolution(data=Conv2d_1_depthwise_Relu6, kernel=(1L, 1L), stride=(1L, 1L), dilate = (), pad=(0L, 0L), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D')
    Conv2d_1_pointwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_1_pointwise_Conv2D, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNorm')
    Conv2d_1_pointwise_Relu6 = mx.sym.Activation(data = Conv2d_1_pointwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6')
    Conv2d_2_depthwise_depthwise_pad = mx.sym.pad(data = Conv2d_1_pointwise_Relu6, mode = 'constant', pad_width=(0, 0, 0, 0, 0L, 1L, 0L, 1L), constant_value = 0.0, name = 'MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise_pad')
    Conv2d_2_depthwise_depthwise = mx.sym.Convolution(data=Conv2d_2_depthwise_depthwise_pad, kernel=(3L, 3L), stride=(2L, 2L), dilate = (), num_filter = 16, num_group = 16, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise')
    Conv2d_2_depthwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_2_depthwise_depthwise, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNorm')
    Conv2d_2_depthwise_Relu6 = mx.sym.Activation(data = Conv2d_2_depthwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6')
    Conv2d_2_pointwise_Conv2D = mx.sym.Convolution(data=Conv2d_2_depthwise_Relu6, kernel=(1L, 1L), stride=(1L, 1L), dilate = (), pad=(0L, 0L), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D')
    Conv2d_2_pointwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_2_pointwise_Conv2D, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNorm')
    Conv2d_2_pointwise_Relu6 = mx.sym.Activation(data = Conv2d_2_pointwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6')
    Conv2d_3_depthwise_depthwise = mx.sym.Convolution(data=Conv2d_2_pointwise_Relu6, kernel=(3L, 3L), stride=(1L, 1L), dilate = (), pad=(1L, 1L), num_filter = 32, num_group = 32, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise')
    Conv2d_3_depthwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_3_depthwise_depthwise, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNorm')
    Conv2d_3_depthwise_Relu6 = mx.sym.Activation(data = Conv2d_3_depthwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6')
    Conv2d_3_pointwise_Conv2D = mx.sym.Convolution(data=Conv2d_3_depthwise_Relu6, kernel=(1L, 1L), stride=(1L, 1L), dilate = (), pad=(0L, 0L), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D')
    Conv2d_3_pointwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_3_pointwise_Conv2D, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNorm')
    Conv2d_3_pointwise_Relu6 = mx.sym.Activation(data = Conv2d_3_pointwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6')
    Conv2d_4_depthwise_depthwise_pad = mx.sym.pad(data = Conv2d_3_pointwise_Relu6, mode = 'constant', pad_width=(0, 0, 0, 0, 0L, 1L, 0L, 1L), constant_value = 0.0, name = 'MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise_pad')
    Conv2d_4_depthwise_depthwise = mx.sym.Convolution(data=Conv2d_4_depthwise_depthwise_pad, kernel=(3L, 3L), stride=(2L, 2L), dilate = (), num_filter = 32, num_group = 32, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise')
    Conv2d_4_depthwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_4_depthwise_depthwise, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNorm')
    Conv2d_4_depthwise_Relu6 = mx.sym.Activation(data = Conv2d_4_depthwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6')
    Conv2d_4_pointwise_Conv2D = mx.sym.Convolution(data=Conv2d_4_depthwise_Relu6, kernel=(1L, 1L), stride=(1L, 1L), dilate = (), pad=(0L, 0L), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D')
    Conv2d_4_pointwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_4_pointwise_Conv2D, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNorm')
    Conv2d_4_pointwise_Relu6 = mx.sym.Activation(data = Conv2d_4_pointwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6')
    Conv2d_5_depthwise_depthwise = mx.sym.Convolution(data=Conv2d_4_pointwise_Relu6, kernel=(3L, 3L), stride=(1L, 1L), dilate = (), pad=(1L, 1L), num_filter = 64, num_group = 64, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise')
    Conv2d_5_depthwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_5_depthwise_depthwise, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNorm')
    Conv2d_5_depthwise_Relu6 = mx.sym.Activation(data = Conv2d_5_depthwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6')
    Conv2d_5_pointwise_Conv2D = mx.sym.Convolution(data=Conv2d_5_depthwise_Relu6, kernel=(1L, 1L), stride=(1L, 1L), dilate = (), pad=(0L, 0L), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D')
    Conv2d_5_pointwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_5_pointwise_Conv2D, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNorm')
    Conv2d_5_pointwise_Relu6 = mx.sym.Activation(data = Conv2d_5_pointwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6')
    Conv2d_6_depthwise_depthwise_pad = mx.sym.pad(data = Conv2d_5_pointwise_Relu6, mode = 'constant', pad_width=(0, 0, 0, 0, 0L, 1L, 0L, 1L), constant_value = 0.0, name = 'MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise_pad')
    Conv2d_6_depthwise_depthwise = mx.sym.Convolution(data=Conv2d_6_depthwise_depthwise_pad, kernel=(3L, 3L), stride=(2L, 2L), dilate = (), num_filter = 64, num_group = 64, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise')
    Conv2d_6_depthwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_6_depthwise_depthwise, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNorm')
    Conv2d_6_depthwise_Relu6 = mx.sym.Activation(data = Conv2d_6_depthwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6')
    Conv2d_6_pointwise_Conv2D = mx.sym.Convolution(data=Conv2d_6_depthwise_Relu6, kernel=(1L, 1L), stride=(1L, 1L), dilate = (), pad=(0L, 0L), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D')
    Conv2d_6_pointwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_6_pointwise_Conv2D, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNorm')
    Conv2d_6_pointwise_Relu6 = mx.sym.Activation(data = Conv2d_6_pointwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6')
    Conv2d_7_depthwise_depthwise = mx.sym.Convolution(data=Conv2d_6_pointwise_Relu6, kernel=(3L, 3L), stride=(1L, 1L), dilate = (), pad=(1L, 1L), num_filter = 128, num_group = 128, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise')
    Conv2d_7_depthwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_7_depthwise_depthwise, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNorm')
    Conv2d_7_depthwise_Relu6 = mx.sym.Activation(data = Conv2d_7_depthwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6')
    Conv2d_7_pointwise_Conv2D = mx.sym.Convolution(data=Conv2d_7_depthwise_Relu6, kernel=(1L, 1L), stride=(1L, 1L), dilate = (), pad=(0L, 0L), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D')
    Conv2d_7_pointwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_7_pointwise_Conv2D, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNorm')
    Conv2d_7_pointwise_Relu6 = mx.sym.Activation(data = Conv2d_7_pointwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6')
    Conv2d_8_depthwise_depthwise = mx.sym.Convolution(data=Conv2d_7_pointwise_Relu6, kernel=(3L, 3L), stride=(1L, 1L), dilate = (), pad=(1L, 1L), num_filter = 128, num_group = 128, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise')
    Conv2d_8_depthwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_8_depthwise_depthwise, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNorm')
    Conv2d_8_depthwise_Relu6 = mx.sym.Activation(data = Conv2d_8_depthwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6')
    Conv2d_8_pointwise_Conv2D = mx.sym.Convolution(data=Conv2d_8_depthwise_Relu6, kernel=(1L, 1L), stride=(1L, 1L), dilate = (), pad=(0L, 0L), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D')
    Conv2d_8_pointwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_8_pointwise_Conv2D, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm')
    Conv2d_8_pointwise_Relu6 = mx.sym.Activation(data = Conv2d_8_pointwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6')
    Conv2d_9_depthwise_depthwise = mx.sym.Convolution(data=Conv2d_8_pointwise_Relu6, kernel=(3L, 3L), stride=(1L, 1L), dilate = (), pad=(1L, 1L), num_filter = 128, num_group = 128, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise')
    Conv2d_9_depthwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_9_depthwise_depthwise, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNorm')
    Conv2d_9_depthwise_Relu6 = mx.sym.Activation(data = Conv2d_9_depthwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6')
    Conv2d_9_pointwise_Conv2D = mx.sym.Convolution(data=Conv2d_9_depthwise_Relu6, kernel=(1L, 1L), stride=(1L, 1L), dilate = (), pad=(0L, 0L), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D')
    Conv2d_9_pointwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_9_pointwise_Conv2D, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNorm')
    Conv2d_9_pointwise_Relu6 = mx.sym.Activation(data = Conv2d_9_pointwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6')
    Conv2d_10_depthwise_depthwise = mx.sym.Convolution(data=Conv2d_9_pointwise_Relu6, kernel=(3L, 3L), stride=(1L, 1L), dilate = (), pad=(1L, 1L), num_filter = 128, num_group = 128, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise')
    Conv2d_10_depthwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_10_depthwise_depthwise, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNorm')
    Conv2d_10_depthwise_Relu6 = mx.sym.Activation(data = Conv2d_10_depthwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6')
    Conv2d_10_pointwise_Conv2D = mx.sym.Convolution(data=Conv2d_10_depthwise_Relu6, kernel=(1L, 1L), stride=(1L, 1L), dilate = (), pad=(0L, 0L), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D')
    Conv2d_10_pointwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_10_pointwise_Conv2D, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNorm')
    Conv2d_10_pointwise_Relu6 = mx.sym.Activation(data = Conv2d_10_pointwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6')
    Conv2d_11_depthwise_depthwise = mx.sym.Convolution(data=Conv2d_10_pointwise_Relu6, kernel=(3L, 3L), stride=(1L, 1L), dilate = (), pad=(1L, 1L), num_filter = 128, num_group = 128, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise')
    Conv2d_11_depthwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_11_depthwise_depthwise, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNorm')
    Conv2d_11_depthwise_Relu6 = mx.sym.Activation(data = Conv2d_11_depthwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6')
    Conv2d_11_pointwise_Conv2D = mx.sym.Convolution(data=Conv2d_11_depthwise_Relu6, kernel=(1L, 1L), stride=(1L, 1L), dilate = (), pad=(0L, 0L), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D')
    Conv2d_11_pointwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_11_pointwise_Conv2D, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNorm')
    Conv2d_11_pointwise_Relu6 = mx.sym.Activation(data = Conv2d_11_pointwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6')
    Conv2d_12_depthwise_depthwise_pad = mx.sym.pad(data = Conv2d_11_pointwise_Relu6, mode = 'constant', pad_width=(0, 0, 0, 0, 0L, 1L, 0L, 1L), constant_value = 0.0, name = 'MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise_pad')
    Conv2d_12_depthwise_depthwise = mx.sym.Convolution(data=Conv2d_12_depthwise_depthwise_pad, kernel=(3L, 3L), stride=(2L, 2L), dilate = (), num_filter = 128, num_group = 128, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise')
    Conv2d_12_depthwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_12_depthwise_depthwise, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNorm')
    Conv2d_12_depthwise_Relu6 = mx.sym.Activation(data = Conv2d_12_depthwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6')
    Conv2d_12_pointwise_Conv2D = mx.sym.Convolution(data=Conv2d_12_depthwise_Relu6, kernel=(1L, 1L), stride=(1L, 1L), dilate = (), pad=(0L, 0L), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D')
    Conv2d_12_pointwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_12_pointwise_Conv2D, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNorm')
    Conv2d_12_pointwise_Relu6 = mx.sym.Activation(data = Conv2d_12_pointwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6')
    Conv2d_13_depthwise_depthwise = mx.sym.Convolution(data=Conv2d_12_pointwise_Relu6, kernel=(3L, 3L), stride=(1L, 1L), dilate = (), pad=(1L, 1L), num_filter = 256, num_group = 256, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise')
    Conv2d_13_depthwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_13_depthwise_depthwise, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNorm')
    Conv2d_13_depthwise_Relu6 = mx.sym.Activation(data = Conv2d_13_depthwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6')
    Conv2d_13_pointwise_Conv2D = mx.sym.Convolution(data=Conv2d_13_depthwise_Relu6, kernel=(1L, 1L), stride=(1L, 1L), dilate = (), pad=(0L, 0L), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D')
    Conv2d_13_pointwise_BatchNorm = mx.sym.BatchNorm(data = Conv2d_13_pointwise_Conv2D, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNorm')
    Conv2d_13_pointwise_Relu6 = mx.sym.Activation(data = Conv2d_13_pointwise_BatchNorm, act_type = 'relu', name = 'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6')
    Logits_AvgPool_1a_AvgPool = mx.sym.Pooling(data = Conv2d_13_pointwise_Relu6, global_pool = False, kernel=(7L, 7L), pool_type = 'avg', stride=(2L, 2L), pad=(0L, 0L), name = 'MobilenetV1/Logits/AvgPool_1a/AvgPool')
    Logits_Conv2d_1c_1x1_Conv2D = mx.sym.Convolution(data=Logits_AvgPool_1a_AvgPool, kernel=(1L, 1L), stride=(1L, 1L), dilate = (), pad=(0L, 0L), num_filter = 1001, num_group = 1, no_bias = False, layout = 'NCHW', name = 'MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D')
    Logits_SpatialSqueeze = mx.sym.flatten(data = Logits_Conv2d_1c_1x1_Conv2D, name = 'MobilenetV1/Logits/SpatialSqueeze')
    Predictions_Reshape = mx.sym.reshape(data = Logits_SpatialSqueeze, shape = (-1, 1001), reverse = False, name = 'MobilenetV1/Predictions/Reshape')
    Predictions_Softmax = mx.sym.SoftmaxOutput(data = Predictions_Reshape, name = 'softmax')
    
    # if a GPU is available, change mx.cpu() to mx.gpu()
    model           = mx.mod.Module(symbol = Predictions_Softmax, context = mx.cpu(), data_names = ['input'])
    return model

def deploy_weight(model, weight_file):

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    arg_params = dict()
    aux_params = dict()
    for weight_name, weight_data in weights_dict.items():
        weight_name = str(weight_name)
        if "moving" in weight_name:
            aux_params[weight_name] = mx.nd.array(weight_data)
        elif weight_name.endswith("depthwise_weight"):
            arg_params[weight_name] = mx.nd.array(np.swapaxes(weight_data,0,1))
        else:
            arg_params[weight_name] = mx.nd.array(weight_data)
            

    model.bind(for_training = False, data_shapes = [('input', (1, 3, 224, 224))])

    executor = model._exec_group.execs[0]
    pprint(dict(map(lambda item: (item[0],item[1].shape), executor.arg_dict.items())))
    
    model.set_params(arg_params = arg_params, aux_params = aux_params, allow_missing = True)

    return model


import matplotlib.pyplot as plt
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


def get_image(fname, show = False):
    import cv2
    # download and show the image
    # fname = mx.test_utils.download(url)
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    if show:
        plt.imshow(img)
        plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


def predict(model, labels, fname):
    # to show the image, change the argument show into True
    img = get_image(fname, show = False)
    # compute the predict probabilities
    model.forward(Batch([mx.nd.array(img)]))
    prob = model.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    with open("MobileNet/labels.json") as fp:
        label2name = json.load(fp)
    print(label2name)
    for i in a[0:5]:
        print('prbability = %f, class = %s, name = %s' %(prob[i], labels[i], label2name[str(i)]))


if __name__ == '__main__':
    model = RefactorModel()
    # remember to adjust params path
    model = deploy_weight(model, 'IR/mobilenet_v1_0.25_224-0000.params')

    # call function predict
    with open('imagenet_lsvrc_2015_synsets.txt', 'r') as f:
        labels = [l.rstrip() for l in f]
    predict(model, labels, 'cat_square.jpg')
