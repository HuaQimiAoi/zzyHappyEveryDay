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
import numpy as np
import time as t
from datetime import datetime
import os
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
from lib.utils import TimeDistributed, log_to_tensorboard, make_dir
from lib.encoder import Encoder
from lib.decoder import SplitDiagGaussianDecoder, BernoulliDecoder
from lib.CRULayer import CRULayer
from lib.CRUCell import var_activation, var_activation_inverse
from lib.losses import rmse, mse, GaussianNegLogLik, bernoulli_nll
from lib.data_utils import align_output_and_target, adjust_obs_for_extrapolation

optim = torch.optim
nn = torch.nn


class CRU(nn.Module):

    def __init__(self, target_dim: int, lsd: int, args,
                 use_cuda_if_available: bool = True, bernoulli_output: bool = False):
        """
                CRU模型（Constrained Recurrent Units）的定义

                :param target_dim: 输出的维度
                :param lsd: 潜在状态维度
                :param args: 解析后的参数
                :param use_cuda_if_available: 是否使用CUDA，如果可用的话
                :param bernoulli_output: 是否使用伯努利分布的输出（适用于图像数据）
                """
        super().__init__()
        # 指定模型运行设备
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        # 潜在状态维度和输出维度计算
        self._lsd = lsd
        if self._lsd % 2 == 0:
            self._lod = int(self._lsd / 2)
        else:
            raise Exception('Latent state dimension must be even number.')
        # 保存解析后的参数
        self.args = args

        # 参数设置（TODO: 使之可配置）
        self._enc_out_normalization = "pre"
        self._initial_state_variance = 10.0
        self._learning_rate = self.args.lr
        self.bernoulli_output = bernoulli_output

        # 主要模型组件
        self._cru_layer = CRULayer(
            latent_obs_dim=self._lod, args=args).to(self._device)

        # 初始化编码器
        Encoder._build_hidden_layers = self._build_enc_hidden_layers
        enc = Encoder(self._lod, output_normalization=self._enc_out_normalization,
                      enc_var_activation=args.enc_var_activation).to(dtype=torch.float64)
        # 初始化解码器
        if bernoulli_output:
            BernoulliDecoder._build_hidden_layers = self._build_dec_hidden_layers
            self._dec = TimeDistributed(BernoulliDecoder(self._lod, out_dim=target_dim, args=args).to(
                self._device, dtype=torch.float64), num_outputs=1, low_mem=True)
            self._enc = TimeDistributed(
                enc, num_outputs=2, low_mem=True).to(self._device)

        else:
            SplitDiagGaussianDecoder._build_hidden_layers_mean = self._build_dec_hidden_layers_mean
            SplitDiagGaussianDecoder._build_hidden_layers_var = self._build_dec_hidden_layers_var
            self._dec = TimeDistributed(SplitDiagGaussianDecoder(self._lod, out_dim=target_dim, dec_var_activation=args.dec_var_activation).to(
                dtype=torch.float64), num_outputs=2).to(self._device)
            self._enc = TimeDistributed(enc, num_outputs=2).to(self._device)

        # 构建（默认）初始状态
        self._initial_mean = torch.zeros(1, self._lsd).to(
            self._device, dtype=torch.float64)
        log_ic_init = var_activation_inverse(self._initial_state_variance)
        self._log_icu = torch.nn.Parameter(
            log_ic_init * torch.ones(1, self._lod).to(self._device, dtype=torch.float64))
        self._log_icl = torch.nn.Parameter(
            log_ic_init * torch.ones(1, self._lod).to(self._device, dtype=torch.float64))
        self._ics = torch.zeros(1, self._lod).to(
            self._device, dtype=torch.float64)

        # 模型参数和优化器
        self._params = list(self._enc.parameters())
        self._params += list(self._cru_layer.parameters())
        self._params += list(self._dec.parameters())
        self._params += [self._log_icu, self._log_icl]

        self._optimizer = optim.Adam(self._params, lr=self.args.lr)
        self._shuffle_rng = np.random.RandomState(
            42)  # 用于打乱批次的随机数生成器

    def _build_enc_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
                为编码器构建隐藏层
                :return: torch.nn.ModuleList形式的隐藏层列表和最后一层的输出大小
                """
        raise NotImplementedError

    def _build_dec_hidden_layers_mean(self) -> Tuple[nn.ModuleList, int]:
        """
                为解码器均值部分构建隐藏层
                :return: torch.nn.ModuleList形式的隐藏层列表和最后一层的输出大小
                """
        raise NotImplementedError

    def _build_dec_hidden_layers_var(self) -> Tuple[nn.ModuleList, int]:
        """
                为解码器方差部分构建隐藏层
                :return: torch.nn.ModuleList形式的隐藏层列表和最后一层的输出大小
                """
        raise NotImplementedError

    def forward(self, obs_batch: torch.Tensor, time_points: torch.Tensor = None,
                obs_valid: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                对一个批次进行单次前向传播
                :param obs_batch: 观测序列的批次
                :param time_points: 观测的时间戳
                :param obs_valid: 时间戳是否包含有效的观测
                :return: 输出均值和输出方差的元组
                """
        # 编码器计算潜在变量的均值和方差
        y, y_var = self._enc(obs_batch)
        # 使用CRU层进行状态预测
        post_mean, post_cov, prior_mean, prior_cov, kalman_gain = self._cru_layer(y, y_var, self._initial_mean,
                                                                                  [var_activation(self._log_icu), var_activation(
                                                                                      self._log_icl), self._ics],
                                                                                  obs_valid=obs_valid, time_points=time_points)
        # 对于伯努利分布的输出（图像），输出图像的均值
        if self.bernoulli_output:
            out_mean = self._dec(post_mean)
            out_var = None

        # 对于单步预测任务，输出下一时刻的均值和方差
        elif self.args.task == 'one_step_ahead_prediction':
            out_mean, out_var = self._dec(
                prior_mean, torch.cat(prior_cov, dim=-1))

        # 对于其他任务，输出滤波后的观测均值和方差
        else:
            out_mean, out_var = self._dec(
                post_mean, torch.cat(post_cov, dim=-1))

        # 返回中间结果，用于调试和分析
        intermediates = {
            'post_mean': post_mean,
            'post_cov': post_cov,
            'prior_mean': prior_mean,
            'prior_cov': prior_cov,
            'kalman_gain': kalman_gain,
            'y': y,
            'y_var': y_var
        }

        return out_mean, out_var, intermediates

    # new code component
    def interpolation(self, data, track_gradient=True):
        """计算插值任务的损失

                :param data: 数据批次
                :param track_gradient: 是否跟踪渐变以进行反向传播
                :return: 损失，输出，输入，中间变量，插值点上的度量
                """
        # 将数据移动到设备上
        if self.bernoulli_output:
            obs, truth, obs_valid, obs_times, mask_truth = [
                j.to(self._device) for j in data]
            mask_obs = None
        else:
            obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
                j.to(self._device) for j in data]

        # 调整时间戳
        obs_times = self.args.ts * obs_times

        with torch.set_grad_enabled(track_gradient):
            # 调用前向传播函数获取输出
            output_mean, output_var, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)
            # 计算伯努利分布输出的损失
            if self.bernoulli_output:
                loss = bernoulli_nll(truth, output_mean, uint8_targets=False)
                mask_imput = (~obs_valid[..., None, None, None]) * mask_truth
                imput_loss = np.nan  # TODO: 计算插值点上的伯努利损失
                imput_mse = mse(
                    truth.flatten(
                        start_dim=2), output_mean.flatten(
                        start_dim=2), mask=mask_imput.flatten(
                        start_dim=2))
            # 计算高斯分布输出的损失
            else:
                loss = GaussianNegLogLik(
                    output_mean, truth, output_var, mask=mask_truth)
                # 仅在插值点上计算度量
                mask_imput = (~obs_valid[..., None]) * mask_truth
                imput_loss = GaussianNegLogLik(
                    output_mean, truth, output_var, mask=mask_imput)
                imput_mse = mse(truth, output_mean, mask=mask_imput)

        return loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse

    def extrapolation(self, data, track_gradient=True):
        """计算外推任务的损失

         :param data: 数据批次
         :param track_gradient: 是否跟踪渐变以进行反向传播
         :return: 损失，输出，输入，中间变量，插值点上的度量
         """
        # 将数据移动到设备上
        obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
            j.to(self._device) for j in data]
        # 调整观测值和时间戳以适应外推
        obs, obs_valid = adjust_obs_for_extrapolation(
            obs, obs_valid, obs_times, self.args.cut_time)
        obs_times = self.args.ts * obs_times

        with torch.set_grad_enabled(track_gradient):
            # 调用前向传播函数获取输出
            output_mean, output_var, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)
            # 计算高斯分布输出的损失
            loss = GaussianNegLogLik(
                output_mean, truth, output_var, mask=mask_truth)

            # 仅在插值点上计算度量
            mask_imput = (~obs_valid[..., None]) * mask_truth
            imput_loss = GaussianNegLogLik(
                output_mean, truth, output_var, mask=mask_imput)
            imput_mse = mse(truth, output_mean, mask=mask_imput)

        return loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse

    def regression(self, data, track_gradient=True):
        """计算回归任务的损失

                :param data: 数据批次
                :param track_gradient: 是否跟踪渐变以进行反向传播
                :return: 损失，输入，中间变量和计算得到的输出
                """
        # 将数据移动到设备上
        obs, truth, obs_times, obs_valid = [j.to(self._device) for j in data]
        mask_truth = None
        mask_obs = None
        with torch.set_grad_enabled(track_gradient):
            # 调用前向传播函数获取输出
            output_mean, output_var, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)
            # 计算高斯分布输出的损失
            loss = GaussianNegLogLik(
                output_mean, truth, output_var, mask=mask_truth)

        return loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates

    def one_step_ahead_prediction(self, data, track_gradient=True):
        """计算一步预测任务的损失

                :param data: 数据批次
                :param track_gradient: 是否跟踪渐变以进行反向传播
                :return: 损失，输入，中间变量和计算得到的输出
                """
        # 将数据移动到设备上
        obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
            j.to(self._device) for j in data]
        # 调整时间戳
        obs_times = self.args.ts * obs_times
        with torch.set_grad_enabled(track_gradient):
            # 调用前向传播函数获取输出
            output_mean, output_var, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)
            # 调整输出和目标以对齐
            output_mean, output_var, truth, mask_truth = align_output_and_target(
                output_mean, output_var, truth, mask_truth)
            # 计算高斯分布输出的损失
            loss = GaussianNegLogLik(
                output_mean, truth, output_var, mask=mask_truth)

        return loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates

    def train_epoch(self, dl, optimizer):
        """训练模型一个epoch

                :param dl: 包含训练数据的数据加载器
                :param optimizer: 用于训练的优化器
                :return: 评估指标，计算得到的输出，输入，中间变量
                """
        # 初始化各个指标
        epoch_ll = 0  # 用于记录epoch的总损失
        epoch_rmse = 0  # 用于记录epoch的均方根误差
        epoch_mse = 0  # 用于记录epoch的均方误差

        if self.args.save_intermediates is not None:
            mask_obs_epoch = []  # 记录epoch每个batch的遮掩观测数据
            intermediates_epoch = []  # 记录epoch每个batch的中间变量

        if self.args.task == 'extrapolation' or self.args.task == 'interpolation':
            epoch_imput_ll = 0  # 用于记录epoch的插值损失
            epoch_imput_mse = 0  # 用于记录epoch的插值均方误差

        for i, data in enumerate(dl):
            # 根据任务类型调用对应的任务处理函数
            if self.args.task == 'interpolation':
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.interpolation(
                    data)

            elif self.args.task == 'extrapolation':
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.extrapolation(
                    data)

            elif self.args.task == 'regression':
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates = self.regression(
                    data)

            elif self.args.task == 'one_step_ahead_prediction':
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates = self.one_step_ahead_prediction(
                    data)

            else:
                raise Exception('Unknown task')

            # 检查是否存在NaN
            if torch.any(torch.isnan(loss)):
                print('--NAN in loss')
            for name, par in self.named_parameters():
                if torch.any(torch.isnan(par)):
                    print('--NAN before optimiser step in parameter ', name)
            torch.autograd.set_detect_anomaly(
                self.args.anomaly_detection)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            if self.args.grad_clip:
                nn.utils.clip_grad_norm_(self.parameters(), 1)
            optimizer.step()

            # 检查梯度中是否存在NaN
            for name, par in self.named_parameters():
                if torch.any(torch.isnan(par.grad)):
                    print('--NAN in gradient ', name)
                if torch.any(torch.isnan(par)):
                    print('--NAN after optimiser step in parameter ', name)

            # 在整个epoch中汇总指标和中间变量
            epoch_ll += loss
            epoch_rmse += rmse(truth, output_mean, mask_truth).item()
            epoch_mse += mse(truth, output_mean, mask_truth).item()

            if self.args.task == 'extrapolation' or self.args.task == 'interpolation':
                epoch_imput_ll += imput_loss
                epoch_imput_mse += imput_mse
                imput_metrics = [epoch_imput_ll /
                                 (i + 1), epoch_imput_mse / (i + 1)]
            else:
                imput_metrics = None

            if self.args.save_intermediates is not None:
                intermediates_epoch.append(intermediates)
                mask_obs_epoch.append(mask_obs)

        # 保存用于绘图的数据
        if self.args.save_intermediates is not None:
            torch.save(mask_obs_epoch, os.path.join(
                self.args.save_intermediates, 'train_mask_obs.pt'))
            torch.save(intermediates_epoch, os.path.join(
                self.args.save_intermediates, 'train_intermediates.pt'))

        return epoch_ll / (i + 1), epoch_rmse / (i + 1), epoch_mse / (i + 1), [
            output_mean, output_var], intermediates, [obs, truth, mask_obs], imput_metrics

    def eval_epoch(self, dl):
        """在整个数据集上评估模型

            :param dl: 包含验证或测试数据的数据加载器
            :return: 评估指标，计算得到的输出，输入，中间变量
            """
        epoch_ll = 0  # 用于记录epoch的总损失
        epoch_rmse = 0  # 用于记录epoch的均方根误差
        epoch_mse = 0  # 用于记录epoch的均方误差

        if self.args.task == 'extrapolation' or self.args.task == 'interpolation':
            epoch_imput_ll = 0  # 用于记录epoch的插值损失
            epoch_imput_mse = 0  # 用于记录epoch的插值均方误差

        if self.args.save_intermediates is not None:
            mask_obs_epoch = []  # 记录epoch每个batch的遮掩观测数据
            intermediates_epoch = []  # 记录epoch每个batch的中间变量

        for i, data in enumerate(dl):
            if self.args.task == 'interpolation':
                # 调用插值任务处理函数
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.interpolation(
                    data, track_gradient=False)

            elif self.args.task == 'extrapolation':
                # 调用外推任务处理函数
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.extrapolation(
                    data, track_gradient=False)

            elif self.args.task == 'regression':
                # 调用回归任务处理函数
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates = self.regression(
                    data, track_gradient=False)

            elif self.args.task == 'one_step_ahead_prediction':
                # 调用一步预测任务处理函数
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates = self.one_step_ahead_prediction(
                    data, track_gradient=False)

            # 汇总指标
            epoch_ll += loss
            epoch_rmse += rmse(truth, output_mean, mask_truth).item()
            epoch_mse += mse(truth, output_mean, mask_truth).item()

            if self.args.task == 'extrapolation' or self.args.task == 'interpolation':
                epoch_imput_ll += imput_loss
                epoch_imput_mse += imput_mse
                imput_metrics = [epoch_imput_ll /
                                 (i + 1), epoch_imput_mse / (i + 1)]
            else:
                imput_metrics = None

            if self.args.save_intermediates is not None:
                intermediates_epoch.append(intermediates)
                mask_obs_epoch.append(mask_obs)

        # 保存用于绘图的数据
        if self.args.save_intermediates is not None:
            torch.save(output_mean, os.path.join(
                self.args.save_intermediates, 'valid_output_mean.pt'))
            torch.save(obs, os.path.join(
                self.args.save_intermediates, 'valid_obs.pt'))
            torch.save(output_var, os.path.join(
                self.args.save_intermediates, 'valid_output_var.pt'))
            torch.save(truth, os.path.join(
                self.args.save_intermediates, 'valid_truth.pt'))
            torch.save(intermediates_epoch, os.path.join(
                self.args.save_intermediates, 'valid_intermediates.pt'))
            torch.save(mask_obs_epoch, os.path.join(
                self.args.save_intermediates, 'valid_mask_obs.pt'))

        return epoch_ll / (i + 1), epoch_rmse / (i + 1), epoch_mse / (i + 1), [
            output_mean, output_var], intermediates, [obs, truth, mask_obs], imput_metrics

    def train(self, train_dl, valid_dl, identifier, logger, epoch_start=0):
        """在训练集上训练模型并在测试数据上评估。记录结果并保存已训练的模型。

            :param train_dl: 训练数据加载器
            :param valid_dl: 验证数据加载器
            :param identifier: 记录器标识
            :param logger: 记录器对象
            :param epoch_start: 起始epoch
            """

        # 使用Adam优化器和学习率衰减LambdaLR初始化训练过程
        optimizer = optim.Adam(self.parameters(), self.args.lr)
        def lr_update(epoch): return self.args.lr_decay ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_update)
        # 创建TensorBoard的SummaryWriter用于可视化
        make_dir(f'../results/tensorboard/{self.args.dataset}')
        writer = SummaryWriter(
            f'../results/tensorboard/{self.args.dataset}/{identifier}')

        for epoch in range(epoch_start, self.args.epochs):
            start = datetime.now()
            logger.info(f'Epoch {epoch} starts: {start.strftime("%H:%M:%S")}')

            # 训练
            train_ll, train_rmse, train_mse, train_output, intermediates, train_input, train_imput_metrics = self.train_epoch(
                train_dl, optimizer)
            end_training = datetime.now()
            if self.args.tensorboard:
                # 将训练结果记录到TensorBoard
                log_to_tensorboard(self, writer=writer,
                                   mode='train',
                                   metrics=[train_ll, train_rmse, train_mse],
                                   output=train_output,
                                   input=train_input,
                                   intermediates=intermediates,
                                   epoch=epoch,
                                   imput_metrics=train_imput_metrics,
                                   log_rythm=self.args.log_rythm)

            # 评估
            valid_ll, valid_rmse, valid_mse, valid_output, intermediates, valid_input, valid_imput_metrics = self.eval_epoch(
                valid_dl)
            if self.args.tensorboard:
                # 将验证结果记录到TensorBoard
                log_to_tensorboard(self, writer=writer,
                                   mode='valid',
                                   metrics=[valid_ll, valid_rmse, valid_mse],
                                   output=valid_output,
                                   input=valid_input,
                                   intermediates=intermediates,
                                   epoch=epoch,
                                   imput_metrics=valid_imput_metrics,
                                   log_rythm=self.args.log_rythm)

            end = datetime.now()
            logger.info(
                f'Training epoch {epoch} took: {(end_training - start).total_seconds()}')
            logger.info(f'Epoch {epoch} took: {(end - start).total_seconds()}')
            logger.info(
                f' train_nll: {train_ll:3f}, train_mse: {train_mse:3f}')
            logger.info(
                f' valid_nll: {valid_ll:3f}, valid_mse: {valid_mse:3f}')
            # 如果任务是外推或有缺失率，记录插值和外推的额外指标
            if self.args.task == 'extrapolation' or self.args.impute_rate is not None:
                if self.bernoulli_output:
                    logger.info(
                        f' train_mse_imput: {train_imput_metrics[1]:3f}')
                    logger.info(
                        f' valid_mse_imput: {valid_imput_metrics[1]:3f}')
                else:
                    logger.info(
                        f' train_nll_imput: {train_imput_metrics[0]:3f}, train_mse_imput: {train_imput_metrics[1]:3f}')
                    logger.info(
                        f' valid_nll_imput: {valid_imput_metrics[0]:3f}, valid_mse_imput: {valid_imput_metrics[1]:3f}')

            # 更新学习率
            scheduler.step()

        # 保存已训练模型
        make_dir(f'../results/models/{self.args.dataset}')
        torch.save({'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_ll,
                    }, f'../results/models/{self.args.dataset}/{identifier}.tar')
