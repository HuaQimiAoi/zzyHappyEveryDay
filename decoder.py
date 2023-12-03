# Modeling Irregular Time Series with Continuous Recurrent Units (CRUs)
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This source code is derived from Pytorch RKN Implementation (https://github.com/ALRhub/rkn_share)
# Copyright (c) 2021 Philipp Becker (Autonomous Learning Robots Lab @ KIT)
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import torch
from typing import Tuple, Iterable

nn = torch.nn


# 定义 ELU 激活函数的扩展形式 elup1
def elup1(x: torch.Tensor) -> torch.Tensor:
    """
    ELU 激活函数的扩展形式，其中 x 大于等于 0 时，计算 exp(x)；小于 0 时，计算 x + 1。
    :param x: 输入张量
    :return: 激活后的张量
    """
    return torch.exp(x).where(x < 0.0, x + 1.0)


# 定义方差激活函数 var_activation
def var_activation(x: torch.Tensor) -> torch.Tensor:
    """
    方差激活函数，计算输入张量 x 的指数形式。
    :param x: 输入张量
    :return: 激活后的张量
    """
    return torch.exp(x)


# 定义 SplitDiagGaussianDecoder 类
class SplitDiagGaussianDecoder(nn.Module):
    def __init__(self, lod: int, out_dim: int, dec_var_activation: str):
        """
        低维输出的解码器，本实现是“分裂”的，即有完全独立的网络从潜在均值映射到输出均值，
        从潜在协方差映射到输出方差。
        :param lod: 潜在观测维度（用于计算输入大小）
        :param out_dim: 目标数据的维度（假定为一个向量，不支持图像）
        :param dec_var_activation: 用于输出方差的激活函数的名称
        """
        self.dec_var_activation = dec_var_activation
        super(SplitDiagGaussianDecoder, self).__init__()
        self._latent_obs_dim = lod
        self._out_dim = out_dim
        # 构建均值映射的隐藏层
        self._hidden_layers_mean, num_last_hidden_mean = self._build_hidden_layers_mean()
        assert isinstance(self._hidden_layers_mean, nn.ModuleList), "_build_hidden_layers_means needs to return a " \
                                                                    "torch.nn.ModuleList or else the hidden weights " \
                                                                    "are not found by the optimizer"
        # 构建方差映射的隐藏层
        self._hidden_layers_var, num_last_hidden_var = self._build_hidden_layers_var()
        assert isinstance(self._hidden_layers_var, nn.ModuleList), "_build_hidden_layers_var needs to return a " \
                                                                   "torch.nn.ModuleList or else the hidden weights " \
                                                                   "are not found by the optimizer"
        # 均值映射的输出层
        self._out_layer_mean = nn.Linear(
            in_features=num_last_hidden_mean, out_features=out_dim)
        # 方差映射的输出层
        self._out_layer_var = nn.Linear(
            in_features=num_last_hidden_var, out_features=out_dim)

    def _build_hidden_layers_mean(self) -> Tuple[nn.ModuleList, int]:
        """
        为均值解码器构建隐藏层
        :return: nn.ModuleList 隐藏层的列表, 最后一层输出的大小
        """
        raise NotImplementedError

    def _build_hidden_layers_var(self) -> Tuple[nn.ModuleList, int]:
        """
        为方差解码器构建隐藏层
        :return: nn.ModuleList 隐藏层的列表, 最后一层输出的大小
        """
        raise NotImplementedError

    def forward(self, latent_mean: torch.Tensor, latent_cov: Iterable[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数，将输入 latent_mean 和 latent_cov 映射到输出均值和输出方差。
        :param latent_mean: 潜在均值张量
        :param latent_cov: 潜在协方差张量的可迭代集合
        :return: 输出均值和输出方差的元组
        """
        # 对均值进行隐藏层的前向传播
        h_mean = latent_mean
        for layer in self._hidden_layers_mean:
            h_mean = layer(h_mean)
        mean = self._out_layer_mean(h_mean)
        # 对方差进行隐藏层的前向传播
        h_var = latent_cov
        for layer in self._hidden_layers_var:
            h_var = layer(h_var)
        log_var = self._out_layer_var(h_var)
        # 根据方差激活函数计算方差
        if self.dec_var_activation == 'exp':
            var = torch.exp(log_var)
        elif self.dec_var_activation == 'relu':
            var = torch.maximum(log_var, torch.zeros_like(log_var))
        elif self.dec_var_activation == 'square':
            var = torch.square(log_var)
        elif self.dec_var_activation == 'abs':
            var = torch.abs(log_var)
        elif self.dec_var_activation == 'elup1':
            var = elup1(log_var)
        else:
            raise Exception('Variance activation function unknown.')
        return mean, var


class BernoulliDecoder(nn.Module):
    def __init__(self, lod: int, out_dim: int, args):
        """ 用于图像输出的解码器
        :param lod: 潜在观测维度（用于计算输入大小）
        :param out_dim: 目标数据的维度（假定为图像）
        :param args: 解析后的参数
        """
        super(BernoulliDecoder, self).__init__()
        self._latent_obs_dim = lod
        self._out_dim = out_dim
        # 构建隐藏层
        self._hidden_layers, num_last_hidden = self._build_hidden_layers()
        assert isinstance(self._hidden_layers, nn.ModuleList), "_build_hidden_layers_means needs to return a " \
            "torch.nn.ModuleList or else the hidden weights " \
            "are not found by the optimizer"
        # 输出层，使用ConvTranspose2d进行反卷积，并通过Sigmoid激活函数输出
        self._out_layer = nn.Sequential(nn.ConvTranspose2d(in_channels=num_last_hidden, out_channels=1, kernel_size=2, stride=2, padding=5),
                                        nn.Sigmoid())

    def _build_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        为解码器构建隐藏层
        :return: nn.ModuleList 隐藏层的列表, 最后一层输出的大小
        """
        raise NotImplementedError

    def forward(self, latent_mean: torch.Tensor) \
            -> torch.Tensor:
        """ 解码器的前向传播
        :param latent_mean: 潜在均值张量
        :return: 输出均值
        """
        # 对潜在均值进行隐藏层的前向传播
        h_mean = latent_mean
        for layer in self._hidden_layers:
            h_mean = layer(h_mean)
            #print(f'decoder: {h_mean.shape}')
        mean = self._out_layer(h_mean)
        #print(f'decoder mean {mean.shape}')
        return mean
