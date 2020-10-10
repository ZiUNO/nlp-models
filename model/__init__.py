# -*- coding: utf-8 -*-
# @Time    : 2020/10/4 20:09
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : __init__.py
# @Software: PyCharm
import torch


class Model:
    @staticmethod
    def from_pretrained(model_path):
        return torch.load(model_path)