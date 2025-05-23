# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.registry import MODELS
# from mmcv.runner import force_fp32
from torch import distributed as dist
from torch import nn as nn
from torch.autograd.function import Function


class AllReduce(Function):

    @staticmethod
    def forward(ctx, input):
        input_list = [
            torch.zeros_like(input) for k in range(dist.get_world_size())
        ]
        # Use allgather instead of allreduce in-place operations is unreliable
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output



@MODELS.register_module('naiveSyncBN1d')
class NaiveSyncBatchNorm1d(nn.BatchNorm1d):
    """Syncronized Batch Normalization for 3D Tensors.

    Note:
        This implementation is modified from
        https://github.com/facebookresearch/detectron2/

        `torch.nn.SyncBatchNorm` has known unknown bugs.
        It produces significantly worse AP (and sometimes goes NaN)
        when the batch size on each worker is quite different
        (e.g., when scale augmentation is used).
        In 3D detection, different workers has points of different shapes,
        whish also cause instability.

        Use this implementation before `nn.SyncBatchNorm` is fixed.
        It is slower than `nn.SyncBatchNorm`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16_enabled = False

    # customized normalization layer still needs this decorator
    # to force the input to be fp32 and the output to be fp16
    # TODO: make mmcv fp16 utils handle customized norm layers
    # @force_fp32(out_fp16=True)
    def forward(self, input):
        assert input.dtype == torch.float32, \
            f'input should be in float32 type, got {input.dtype}'
        if not dist.is_initialized() or dist.get_world_size() == 1 or not self.training:
            return super().forward(input)
        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        dim_2 = input.dim() == 2
        if dim_2:
            input.unsqueeze_(2)

        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2])
        meansqr = torch.mean(input * input, dim=[0, 2])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (
            mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1)
        bias = bias.reshape(1, -1, 1)
        input = input * scale + bias
        if dim_2:
            input = input.squeeze(2)
        return input


# @NORM_LAYERS.register_module('naiveSyncBN2d')
class NaiveSyncBatchNorm2d(nn.BatchNorm2d):
    """Syncronized Batch Normalization for 4D Tensors.

    Note:
        This implementation is modified from
        https://github.com/facebookresearch/detectron2/

        `torch.nn.SyncBatchNorm` has known unknown bugs.
        It produces significantly worse AP (and sometimes goes NaN)
        when the batch size on each worker is quite different
        (e.g., when scale augmentation is used).
        This phenomenon also occurs when the multi-modality feature fusion
        modules of multi-modality detectors use SyncBN.

        Use this implementation before `nn.SyncBatchNorm` is fixed.
        It is slower than `nn.SyncBatchNorm`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16_enabled = False

    # customized normalization layer still needs this decorator
    # to force the input to be fp32 and the output to be fp16
    # TODO: make mmcv fp16 utils handle customized norm layers
    # @force_fp32(out_fp16=True)
    def forward(self, input):
        assert input.dtype == torch.float32, \
            f'input should be in float32 type, got {input.dtype}'
        if not dist.is_initialized() or dist.get_world_size() == 1 or not self.training:
            return super().forward(input)
        # if dist.get_world_size() == 1 or not self.training:
        #     return super().forward(input)

        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (
            mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias

# @NORM_LAYERS.register_module('naiveSyncBN3d')
class NaiveSyncBatchNorm3d(nn.BatchNorm3d):
    """Syncronized Batch Normalization for 5D Tensors.

    Note:
        This implementation is modified from
        https://github.com/facebookresearch/detectron2/

        `torch.nn.SyncBatchNorm` has known unknown bugs.
        It produces significantly worse AP (and sometimes goes NaN)
        when the batch size on each worker is quite different
        (e.g., when scale augmentation is used).
        This phenomenon also occurs when the multi-modality feature fusion
        modules of multi-modality detectors use SyncBN.

        Use this implementation before `nn.SyncBatchNorm` is fixed.
        It is slower than `nn.SyncBatchNorm`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16_enabled = False

    # customized normalization layer still needs this decorator
    # to force the input to be fp32 and the output to be fp16
    # TODO: make mmcv fp16 utils handle customized norm layers
    # @force_fp32(out_fp16=True)
    def forward(self, input):
        assert input.dtype == torch.float32, \
            f'input should be in float32 type, got {input.dtype}'
        if not dist.is_initialized() or dist.get_world_size() == 1 or not self.training:
            return super().forward(input)
        # if dist.get_world_size() == 1 or not self.training:
        #     return super().forward(input)

        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3, 4])
        meansqr = torch.mean(input * input, dim=[0, 2, 3, 4])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (
            mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1, 1)
        return input * scale + bias
