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
from typing import Tuple

nn = torch.nn

# 定义 ELU 激活函数的扩展形式 elup1


def elup1(x: torch.Tensor) -> torch.Tensor:
    """ELU+1激活函数
    :param x: 输入张量
    :return: 经过ELU+1激活后的张量
    """
    return torch.exp(x).where(x < 0.0, x + 1.0)


# 定义方差激活函数
def var_activation(x: torch.Tensor) -> torch.Tensor:
    """方差激活函数
    :param x: 输入张量
    :return: 经过方差激活后的张量
    """
    return torch.exp(x)


class Encoder(nn.Module):

    def __init__(self, lod: int, enc_var_activation: str,
                 output_normalization: str = "post"):
        """高斯编码器，如 RKN ICML 论文中描述的（如果 output_normalization=post）
        :param lod: 潜在观测维度，即编码器均值和方差的输出维度
        :param enc_var_activation: 潜在观测噪声的激活函数
        :param output_normalization: 输出标准化的时机:
                - post: 在输出层之后
                - pre: 在最后一个隐藏层之后，这在大多数情况下同样有效，但更加原则性
                - none: (或其他任何字符串) 不标准化
        """
        super(Encoder, self).__init__()
        # 构建隐藏层
        self._hidden_layers, size_last_hidden = self._build_hidden_layers()
        assert isinstance(self._hidden_layers, nn.ModuleList), "_build_hidden_layers needs to return a " \
                                                               "torch.nn.ModuleList or else the hidden weights are " \
                                                               "not found by the optimizer"
        # 输出均值层
        self._mean_layer = nn.Linear(
            in_features=size_last_hidden, out_features=lod)
        # 输出对数方差层
        self._log_var_layer = nn.Linear(
            in_features=size_last_hidden, out_features=lod)
        # 编码器方差激活函数
        self.enc_var_activation = enc_var_activation
        # 输出标准化的时机
        self._output_normalization = output_normalization

    def _build_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        为编码器构建隐藏层
        :return: nn.ModuleList 隐藏层的列表, 最后一层输出的大小
        """
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 编码器的前向传播
        :param obs: 观测数据张量
        :return: 输出均值和输出方差的元组
        """
        # 初始隐藏层输入为观测数据
        h = obs
        # 通过隐藏层进行前向传播
        for layer in self._hidden_layers:
            h = layer(h)
        # 如果输出标准化时机为 pre，对隐藏层的输出进行 L2 标准化
        if self._output_normalization.lower() == "pre":
            h = nn.functional.normalize(h, p=2, dim=-1, eps=1e-8)
        # 计算均值
        mean = self._mean_layer(h)
        # 如果输出标准化时机为 post，对均值进行 L2 标准化
        if self._output_normalization.lower() == "post":
            mean = nn.functional.normalize(mean, p=2, dim=-1, eps=1e-8)
        # 计算对数方差
        log_var = self._log_var_layer(h)
        # 应用编码器方差激活函数
        if self.enc_var_activation == 'exp':
            var = torch.exp(log_var)
        elif self.enc_var_activation == 'relu':
            var = torch.maximum(log_var, torch.zeros_like(log_var))
        elif self.enc_var_activation == 'square':
            var = torch.square(log_var)
        elif self.enc_var_activation == 'abs':
            var = torch.abs(log_var)
        elif self.enc_var_activation == 'elup1':
            var = torch.exp(log_var).where(log_var < 0.0, log_var + 1.0)
        else:
            raise Exception('Variance activation function unknown.')
        return mean, var
