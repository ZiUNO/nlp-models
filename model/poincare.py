# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 10:01
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : poincare.py
# @Software: PyCharm
import torch
from torch.autograd import Function

eps = 1e-6
boundary = 1 - eps


# class Arcosh(Function):
#     @staticmethod
#     def forward(ctx, x):
#         z = torch.sqrt(x ** 2 - 1)
#         ctx.save_for_backward(z)
#         return torch.log(x + z)
#
#     @staticmethod
#     def backward(ctx, grad_outputs):
#         z, = ctx.saved_tensors
#         z = torch.clamp(z, min=eps)
#         grad = grad_outputs / z
#         return grad


class PoincareDistance(Function):
    @staticmethod
    def norm_square(x):
        return torch.clamp(torch.norm(x, p=2, dim=-1), min=0, max=boundary) ** 2

    @staticmethod
    def forward(ctx, x, y):
        x_norm_square = PoincareDistance.norm_square(x)
        y_norm_square = PoincareDistance.norm_square(y)
        alpha = 1 + 2 * PoincareDistance.norm_square(x - y) / ((1 - x_norm_square) * (1 - y_norm_square))

        # arcosh
        arc_grad = torch.sqrt(alpha ** 2 - 1)
        ctx.save_for_backward(x, y, arc_grad)
        return torch.log(alpha + arc_grad)

    @staticmethod
    def backward(ctx, grad_outputs):
        x, y, arc_grad = ctx.saved_tensors

        # arcosh'
        arc_grad = torch.clamp(arc_grad, min=eps)

        x_y = x - y
        x_y_norm_2 = PoincareDistance.norm_square(x_y)
        x_y_norm_2 = x_y_norm_2.view(x.shape[0], x.shape[1], 1).expand(x.shape)
        x_norm_2_y_norm_2 = (1 - torch.norm(x, p=2, dim=-1) ** 2) * (1 - torch.norm(y, p=2, dim=-1) ** 2)
        x_norm_2_y_norm_2 = x_norm_2_y_norm_2.view(x.shape[0], x.shape[1], 1).expand(x.shape)
        x_grad = 4 * x_y / x_norm_2_y_norm_2 - 4 * x * x_y_norm_2 / x_norm_2_y_norm_2 ** 2
        y_grad = -4 * x_y / x_norm_2_y_norm_2 - 4 * y * x_y_norm_2 / x_norm_2_y_norm_2 ** 2

        grad_inputs = grad_outputs.detach().view(x.shape[0], x.shape[1], 1).expand_as(x)
        arc_grad = arc_grad.view(x.shape[0], x.shape[1], 1).expand_as(x)
        return grad_inputs * arc_grad * x_grad, grad_inputs * arc_grad * y_grad


def poincare_distance(x, y):
    return PoincareDistance.apply(x, y)
