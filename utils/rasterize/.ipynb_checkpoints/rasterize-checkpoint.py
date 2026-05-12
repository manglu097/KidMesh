from torch.utils.cpp_extension import load
rasterize_cuda = load(
    'rasterize_cuda', ['./utils/rasterize/rasterize_cuda.cpp', './utils/rasterize/rasterize_cuda_kernel.cu'], verbose=True)
# rasterize_cuda = load(
#     'rasterize_cuda', ['./utils/rasterize/rasterize_cuda.cpp', './utils/rasterize/dst_tf_cuda_kernel.cu'], verbose=True)


import math
from torch import nn
from torch.autograd import Function
import torch

import rasterize_cuda
from IPython import embed


torch.manual_seed(42)


class RasterizeFunction(Function):
    @staticmethod
    def forward(ctx, vertices, faces, shape):
        # embed()

        N, _, _ = vertices.shape
        D, H, W = shape 
        shape = torch.tensor([D,H,W]).int().cuda()
        volume = []

        if not vertices.is_cuda:
            vertices = vertices.cuda()

        if not faces.is_cuda:
            faces = faces.cuda()    
                 
        for vertices_, faces_ in zip(vertices, faces):    # 就按照批次循环1次
            v = (shape[None].float() - 1) * (vertices_.clone() + 1)/2
            # 目的：这行代码的目的是将顶点坐标vertices_从原始的坐标空间转换到目标体积（或3D网格）的坐标空间。这个过程通常涉及到缩放和偏移操作。
            #
            # 操作解释：
            #
            # vertices_.clone()：首先克隆顶点张量，以避免修改原始数据。
            # (vertices_.clone() + 1) / 2：假设原始顶点坐标在[-1, 1]的范围内（这是一种常见的归一化表示），此操作将顶点坐标变换到[0, 1]的范围内。这是通过先将坐标偏移+1然后缩放/2来实现的。
            # shape[None].float() - 1：计算目标体积的每个维度的大小，并从每个维度减去1，以适应从0开始的索引。使用[None]来增加一个新的轴，以便进行后续的广播操作。
            # 乘以shape[None].float() - 1：将缩放后的顶点坐标进一步缩放到目标体积的大小。因为顶点坐标现在在[0, 1]范围内，乘以(D-1, H-1, W-1)可以将顶点坐标缩放到体积空间内。
            # v = vertices_.clone()
            v = torch.round(v).float()
            # 这一步将调整后的顶点坐标v进行四舍五入，以确保坐标是整数。在栅格化过程中，通常需要顶点坐标为整数值，因为它们将被用作体积数据（一个离散的3D网格）中的索引。取整操作确保了每个顶点都精确地映射到体积网格的一个体素上。
            f = faces_.int() 
 
            volume_ = rasterize_cuda.forward(v, f, shape)[0].float()[None]  
            volume += [volume_]
        
        volume = torch.cat(volume, dim=0)
        volume.requires_grad = True
        # ctx.save_for_backward(*variables) 
        ctx.vertices = vertices
        ctx.faces = faces
        ctx.shape = shape
        ctx.volume = volume

        return volume

    @staticmethod
    def backward(ctx, grad_output):
        vertices = ctx.vertices
        faces = ctx.faces
        shape = ctx.shape
        volume = ctx.volume

        grad_volume = grad_output.contiguous()
        D, H, W = shape 
        shape = torch.tensor([D,H,W]).int().cuda()
        grad_vertices = []
        # embed()
                 
        for output_, grad_volume_, vertices_, faces_ in zip(volume, grad_volume, vertices, faces):
            v = (shape[None].float() - 1) * (vertices_.clone() + 1)/2 
            # v = vertices_.clone()
            v = torch.round(v).float()
            f = faces_.int() 
 
            grad_vertices_ = rasterize_cuda.backward(output_, grad_volume_, v, f, shape)[0].float()[None]  
            grad_vertices += [grad_vertices_]
        
        grad_vertices = torch.cat(grad_vertices, dim=0)
        # grad_vertices = vertices
        grad_faces = grad_shape = None
        return grad_vertices, grad_faces, grad_shape


class Rasterize(nn.Module):
    def __init__(self, shape):
        super(Rasterize, self).__init__() 
        self.shape = shape
 

    def forward(self, vertices, faces): 
        return RasterizeFunction.apply(vertices, faces, self.shape)


class RasterizeCPU(nn.Module):
    def __init__(self, shape):
        super(Rasterize, self).__init__() 
        self.shape = shape
 

    def forward(self, vertices, faces): 
        return RasterizeFunction.apply(vertices, faces, self.shape)
# embed()
