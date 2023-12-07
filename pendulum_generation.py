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

import numpy as np
from PIL import Image
from PIL import ImageDraw
import os


def add_img_noise(imgs, first_n_clean, random, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0):
    """
    :param imgs: 要添加噪声的图像
    :param first_n_clean: 保持前 n 张图像清晰，以便滤波器适应
    :param random: 用于采样的 np.random.RandomState
    :param r: "时间上的相关性因子"，值越小噪声越相关
    :param t_ll: 每个序列的下界从中采样的区间的下界
    :param t_lu: 每个序列的下界从中采样的区间的上界
    :param t_ul: 每个序列的上界从中采样的区间的下界
    :param t_uu: 每个序列的上界从中采样的区间的上界
    :return: 带噪声的图像，用于创建它们的因子
    """

    # 检查边界是否有效
    assert t_ll <= t_lu <= t_ul <= t_uu, "Invalid bounds for noise generation"
    # 如果图像的维度少于5，则添加一个维度
    if len(imgs.shape) < 5:
        imgs = np.expand_dims(imgs, -1)
    # 获取图像批次大小和序列长度
    batch_size, seq_len = imgs.shape[:2]
    # 初始化因子数组
    factors = np.zeros([batch_size, seq_len])
    # 随机初始化第一个因子
    factors[:, 0] = random.uniform(low=0.0, high=1.0, size=batch_size)
    # 循环生成序列中的因子
    for i in range(seq_len - 1):
        factors[:, i + 1] = np.clip(factors[:, i] + random.uniform(
            low=-r, high=r, size=batch_size), a_min=0.0, a_max=1.0)
    # 从给定区间内随机生成两个矩阵，用于缩放因子
    t1 = random.uniform(low=t_ll, high=t_lu, size=(batch_size, 1))
    t2 = random.uniform(low=t_ul, high=t_uu, size=(batch_size, 1))

    # 缩放因子以在指定区间内
    factors = (factors - t1) / (t2 - t1)
    factors = np.clip(factors, a_min=0.0, a_max=1.0)
    factors = np.reshape(factors, list(factors.shape) + [1, 1, 1])
    # 将前 n 张图像的因子设置为1.0，保持清晰度
    factors[:, :first_n_clean] = 1.0
    # 初始化带噪声的图像列表
    noisy_imgs = []
    # 生成带噪声的图像
    for i in range(batch_size):
        if imgs.dtype == np.uint8:
            # 对于 uint8 类型的图像，噪声范围在 0 到 255 之间
            noise = random.uniform(low=0.0, high=255, size=imgs.shape[1:])
            noisy_imgs.append(
                (factors[i] * imgs[i] + (1 - factors[i]) * noise).astype(np.uint8))
        else:
            # 对于其他类型的图像，噪声范围在 0 到 1.1 之间
            noise = random.uniform(low=0.0, high=1.1, size=imgs.shape[1:])
            noisy_imgs.append(factors[i] * imgs[i] + (1 - factors[i]) * noise)
    # 将生成的带噪声图像合并并返回
    return np.squeeze(np.concatenate([np.expand_dims(n, 0) for n in noisy_imgs], 0)), factors


def add_img_noise4(imgs, first_n_clean, random, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0):
    """
    :param imgs: 要添加噪声的图像
    :param first_n_clean: 保持前 n 张图像清晰，以便滤波器适应
    :param random: 用于采样的 np.random.RandomState
    :param r: "时间上的相关性因子"，值越小噪声越相关
    :param t_ll: 每个序列的下界从中采样的区间的下界
    :param t_lu: 每个序列的下界从中采样的区间的上界
    :param t_ul: 每个序列的上界从中采样的区间的下界
    :param t_uu: 每个序列的上界从中采样的区间的上界
    :return: 带噪声的图像，用于创建它们的因子
    """
    # 计算图像的一半尺寸
    half_x = int(imgs.shape[2] / 2)
    half_y = int(imgs.shape[3] / 2)
    # 检查边界是否有效
    assert t_ll <= t_lu <= t_ul <= t_uu, "Invalid bounds for noise generation"
    # 如果图像的维度少于5，则添加一个维度
    if len(imgs.shape) < 5:
        imgs = np.expand_dims(imgs, -1)
    # 获取图像批次大小和序列长度
    batch_size, seq_len = imgs.shape[:2]
    # 初始化因子数组
    factors = np.zeros([batch_size, seq_len, 4])
    # 随机初始化第一个因子
    factors[:, 0] = random.uniform(low=0.0, high=1.0, size=(batch_size, 4))
    # 循环生成序列中的因子
    for i in range(seq_len - 1):
        factors[:, i + 1] = np.clip(factors[:, i] + random.uniform(
            low=-r, high=r, size=(batch_size, 4)), a_min=0.0, a_max=1.0)
    # 从给定区间内随机生成两个矩阵，用于缩放因子
    t1 = random.uniform(low=t_ll, high=t_lu, size=(batch_size, 1, 4))
    t2 = random.uniform(low=t_ul, high=t_uu, size=(batch_size, 1, 4))
    # 缩放因子以在指定区间内
    factors = (factors - t1) / (t2 - t1)
    factors = np.clip(factors, a_min=0.0, a_max=1.0)
    factors = np.reshape(factors, list(factors.shape) + [1, 1, 1])
    # 将前 n 张图像的因子设置为1.0，保持清晰度
    factors[:, :first_n_clean] = 1.0
    # 初始化带噪声的图像列表和 qs 列表
    noisy_imgs = []
    qs = []
    # 循环处理每个批次的图像
    for i in range(batch_size):
        if imgs.dtype == np.uint8:
            # 对于 uint8 类型的图像，噪声范围在 0 到 255 之间
            qs.append(detect_pendulums(imgs[i], half_x, half_y))
            noise = random.uniform(low=0.0, high=255, size=[
                                   4, seq_len, half_x, half_y, imgs.shape[-1]]).astype(np.uint8)
            # 生成带噪声的图像
            curr = np.zeros(imgs.shape[1:], dtype=np.uint8)
            curr[:, :half_x, :half_y] = (factors[i, :, 0] * imgs[i, :, :half_x, :half_y] + (
                1 - factors[i, :, 0]) * noise[0]).astype(np.uint8)
            curr[:, :half_x, half_y:] = (factors[i, :, 1] * imgs[i, :, :half_x, half_y:] + (
                1 - factors[i, :, 1]) * noise[1]).astype(np.uint8)
            curr[:, half_x:, :half_y] = (factors[i, :, 2] * imgs[i, :, half_x:, :half_y] + (
                1 - factors[i, :, 2]) * noise[2]).astype(np.uint8)
            curr[:, half_x:, half_y:] = (factors[i, :, 3] * imgs[i, :, half_x:, half_y:] + (
                1 - factors[i, :, 3]) * noise[3]).astype(np.uint8)
        else:
            # 对于其他类型的图像，噪声范围在 0 到 1.0 之间
            noise = random.uniform(low=0.0, high=1.0, size=[
                                   4, seq_len, half_x, half_y, imgs.shape[-1]])
            curr = np.zeros(imgs.shape[1:])
            curr[:, :half_x, :half_y] = factors[i, :, 0] * imgs[i, :,
                                                                :half_x, :half_y] + (1 - factors[i, :, 0]) * noise[0]
            curr[:, :half_x, half_y:] = factors[i, :, 1] * imgs[i, :,
                                                                :half_x, half_y:] + (1 - factors[i, :, 1]) * noise[1]
            curr[:, half_x:, :half_y] = factors[i, :, 2] * imgs[i, :,
                                                                half_x:, :half_y] + (1 - factors[i, :, 2]) * noise[2]
            curr[:, half_x:, half_y:] = factors[i, :, 3] * imgs[i, :,
                                                                half_x:, half_y:] + (1 - factors[i, :, 3]) * noise[3]
        noisy_imgs.append(curr)
    # 将因子数组扩展一个额外的维度并合并
    factors_ext = np.concatenate([np.squeeze(factors), np.zeros(
        [factors.shape[0], factors.shape[1], 1])], -1)
    # 将 qs 列表合并为数组
    q = np.concatenate([np.expand_dims(q, 0) for q in qs], 0)
    # 初始化新的因子数组
    f = np.zeros(q.shape)
    # 填充新的因子数组
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            for k in range(3):
                f[i, j, k] = factors_ext[i, j, q[i, j, k]]
    # 将生成的带噪声图像合并并返回
    return np.squeeze(np.concatenate([np.expand_dims(n, 0) for n in noisy_imgs], 0)), f


def detect_pendulums(imgs, half_x, half_y):
    """
        :param imgs: 图像数据
        :param half_x: 图像水平方向一半的尺寸
        :param half_y: 图像垂直方向一半的尺寸
        :return: 检测到的 pendulums 类别
    """
    # 将图像划分成四个区域
    qs = [imgs[:, :half_x, :half_y], imgs[:, :half_x, half_y:],
          imgs[:, half_x:, :half_y], imgs[:, half_x:, half_y:]]
    # 统计每个区域中红色通道大于5的像素数量
    r_cts = np.array(
        [np.count_nonzero(q[:, :, :, 0] > 5, axis=(-1, -2)) for q in qs]).T
    # 统计每个区域中绿色通道大于5的像素数量
    g_cts = np.array(
        [np.count_nonzero(q[:, :, :, 1] > 5, axis=(-1, -2)) for q in qs]).T
    # 统计每个区域中蓝色通道大于5的像素数量
    b_cts = np.array(
        [np.count_nonzero(q[:, :, :, 2] > 5, axis=(-1, -2)) for q in qs]).T
    # 合并三个通道的像素数量信息
    cts = np.concatenate([np.expand_dims(c, 1)
                         for c in [r_cts, g_cts, b_cts]], 1)
    # 获取每个像素数量最大的通道索引
    q_max = np.max(cts, -1)
    q = np.argmax(cts, -1)
    # 将像素数量较少的区域标记为类别4
    q[q_max < 10] = 4
    return q


class Pendulum:

    MAX_VELO_KEY = 'max_velo'
    MAX_TORQUE_KEY = 'max_torque'
    MASS_KEY = 'mass'
    LENGTH_KEY = 'length'
    GRAVITY_KEY = 'g'
    FRICTION_KEY = 'friction'
    DT_KEY = 'dt'
    SIM_DT_KEY = 'sim_dt'
    TRANSITION_NOISE_TRAIN_KEY = 'transition_noise_train'
    TRANSITION_NOISE_TEST_KEY = 'transition_noise_test'

    OBSERVATION_MODE_LINE = "line"
    OBSERVATION_MODE_BALL = "ball"


    # 构造函数，初始化 Pendulum 实例
    def __init__(self,
                 img_size,
                 observation_mode,
                 generate_actions=False,
                 transition_noise_std=0.0,
                 observation_noise_std=0.0,
                 pendulum_params=None,
                 seed=0):
        """
                :param img_size: 图像大小
                :param observation_mode: 观测模式，可选 "line" 或 "ball"
                :param generate_actions: 是否生成动作
                :param transition_noise_std: 状态转移噪声标准差
                :param observation_noise_std: 观测噪声标准差
                :param pendulum_params: 摆的参数
                :param seed: 随机数种子
                """
        # 确保观测模式只能是 "ball" 或 "line"
        assert observation_mode == Pendulum.OBSERVATION_MODE_BALL or observation_mode == Pendulum.OBSERVATION_MODE_LINE
        # 全局参数
        self.state_dim = 2
        self.action_dim = 1
        self.img_size = img_size
        self.observation_dim = img_size ** 2
        self.observation_mode = observation_mode

        self.random = np.random.RandomState(seed)

        # 图像参数
        self.img_size_internal = 128
        self.x0 = self.y0 = 64
        self.plt_length = 55 if self.observation_mode == Pendulum.OBSERVATION_MODE_LINE else 50
        self.plt_width = 8

        self.generate_actions = generate_actions

        # 模拟参数
        if pendulum_params is None:
            pendulum_params = self.pendulum_default_params()
        self.max_velo = pendulum_params[Pendulum.MAX_VELO_KEY]
        self.max_torque = pendulum_params[Pendulum.MAX_TORQUE_KEY]
        self.dt = pendulum_params[Pendulum.DT_KEY]
        self.mass = pendulum_params[Pendulum.MASS_KEY]
        self.length = pendulum_params[Pendulum.LENGTH_KEY]
        self.inertia = self.mass * self.length**2 / 3
        self.g = pendulum_params[Pendulum.GRAVITY_KEY]
        self.friction = pendulum_params[Pendulum.FRICTION_KEY]
        self.sim_dt = pendulum_params[Pendulum.SIM_DT_KEY]

        self.observation_noise_std = observation_noise_std
        self.transition_noise_std = transition_noise_std
        # 过渡协方差矩阵和观测协方差矩阵
        self.tranisition_covar_mat = np.diag(
            np.array([1e-8, self.transition_noise_std**2, 1e-8, 1e-8]))
        self.observation_covar_mat = np.diag(
            [self.observation_noise_std**2, self.observation_noise_std**2])

    def sample_data_set(self, num_episodes, episode_length, full_targets):
        """
                采样数据集的方法
                :param num_episodes: 数据集中的轨迹数量
                :param episode_length: 每个轨迹的长度
                :param full_targets: 是否包含完整的目标信息
                :return: 图像数据、目标状态、当前状态、带噪声的目标状态、时间步长
                """
        # 初始化状态和动作数组
        states = np.zeros((num_episodes, episode_length, self.state_dim))
        actions = self._sample_action(
            (num_episodes, episode_length, self.action_dim))
        # 设置初始状态和时间步长数组
        states[:, 0, :] = self._sample_init_state(num_episodes)
        t = np.zeros((num_episodes, episode_length))
        # 生成状态和时间步长
        for i in range(1, episode_length):
            states[:, i, :], dt = self._get_next_states(
                states[:, i - 1, :], actions[:, i - 1, :])
            t[:, i:] += dt
        # 调整角度范围为 [-π, π)
        states[..., 0] -= np.pi
        # 添加观测噪声
        if self.observation_noise_std > 0.0:
            observation_noise = self.random.normal(loc=0.0,
                                                   scale=self.observation_noise_std,
                                                   size=states.shape)
        else:
            observation_noise = np.zeros(states.shape)
        # 计算目标状态
        targets = self.pendulum_kinematic(states)
        # 根据观测模式选择性添加噪声
        if self.observation_mode == Pendulum.OBSERVATION_MODE_LINE:
            noisy_states = states + observation_noise
            noisy_targets = self.pendulum_kinematic(noisy_states)
        elif self.observation_mode == Pendulum.OBSERVATION_MODE_BALL:
            noisy_targets = targets + observation_noise
        # 生成带噪声的图像
        imgs = self._generate_images(noisy_targets[..., :2])
        # 返回生成的数据集
        return imgs, targets[..., :(4 if full_targets else 2)], states, noisy_targets[..., :(4 if full_targets else 2)], t/self.dt

    @staticmethod
    def pendulum_default_params():
        """
                获取默认的摆参数
                :return: 摆参数字典
                """
        # 返回默认的摆参数字典
        return {
            Pendulum.MAX_VELO_KEY: 8,# 最大速度
            Pendulum.MAX_TORQUE_KEY: 10,# 最大扭矩
            Pendulum.MASS_KEY: 1,# 质量
            Pendulum.LENGTH_KEY: 1,# 摆杆长度
            Pendulum.GRAVITY_KEY: 9.81,# 重力加速度
            Pendulum.FRICTION_KEY: 0,# 摩擦系数
            Pendulum.DT_KEY: 0.05,# 时间步长
            Pendulum.SIM_DT_KEY: 1e-4# 模拟时间步长
        }

    def _sample_action(self, shape):
        """
                采样动作的方法
                :param shape: 动作数组的形状
                :return: 随机生成的动作数组
                """
        # 如果允许生成动作，则在指定范围内随机生成动作；否则返回零数组
        if self.generate_actions:
            return self.random.uniform(-self.max_torque, self.max_torque, shape)
        else:
            return np.zeros(shape=shape)

    def _transition_function(self, states, actions):
        """
               进行状态转移的方法
               :param states: 当前状态
               :param actions: 当前动作
               :return: 下一个状态，时间步长
               """
        # 计算状态变化，使用欧拉方法进行模拟
        dt = self.dt
        n_steps = dt / self.sim_dt

        if n_steps != np.round(n_steps):
            # 如果步长不是整数，进行警告
            n_steps = np.round(n_steps)

        c = self.g * self.length * self.mass / self.inertia
        for i in range(0, int(n_steps)):
            # 计算新速度
            velNew = states[..., 1:2] + self.sim_dt * (c * np.sin(states[..., 0:1])
                                                       + actions / self.inertia
                                                       - states[..., 1:2] * self.friction)
            # 更新状态
            states = np.concatenate(
                (states[..., 0:1] + self.sim_dt * velNew, velNew), axis=1)
        return states, dt

    def _get_next_states(self, states, actions):
        """
                获取下一个状态的方法
                :param states: 当前状态
                :param actions: 当前动作
                :return: 下一个状态，时间步长
                """
        # 对动作进行截断，确保在允许范围内
        actions = np.maximum(-self.max_torque,
                             np.minimum(actions, self.max_torque))
        # 获取下一状态，并考虑过渡噪声
        states, dt = self._transition_function(states, actions)
        if self.transition_noise_std > 0.0:
            # 添加过渡噪声
            states[:, 1] += self.random.normal(loc=0.0,
                                               scale=self.transition_noise_std,
                                               size=[len(states)])
        # 将角度限制在 [0, 2π) 范围内
        states[:, 0] = ((states[:, 0]) % (2 * np.pi))
        return states, dt

    def get_ukf_smothing(self, obs):
        """
                获取 UKF 平滑结果的方法
                :param obs: 观测数据
                :return: 平滑后的均值、协方差矩阵、成功标志
                """
        # 获取批量大小和序列长度
        batch_size, seq_length = obs.shape[:2]
        # 初始化成功标志、均值和协方差数组
        succ = np.zeros(batch_size, dtype=np.bool)
        means = np.zeros([batch_size, seq_length, 4])
        covars = np.zeros([batch_size, seq_length, 4, 4])
        fail_ct = 0
        # 遍历每个样本
        for i in range(batch_size):
            # 每隔10个样本输出一次进度
            if i % 10 == 0:
                print(i)
            try:
                # 使用 UKF 进行状态平滑
                means[i], covars[i] = self.ukf.filter(obs[i])
                # 设置成功标志
                succ[i] = True
            except:
                # 如果发生异常，增加失败计数
                fail_ct += 1
        # 输出失败率，并返回成功的样本的均值和协方差
        print(fail_ct / batch_size, "failed")

        return means[succ], covars[succ], succ

    def _sample_init_state(self, nr_epochs):
        """
                采样初始状态的方法
                :param nr_epochs: 数据集中的轨迹数量
                :return: 随机生成的初始状态数组
                """
        # 生成随机的初始状态，包括角度和角速度
        return np.concatenate((self.random.uniform(0, 2 * np.pi, (nr_epochs, 1)), np.zeros((nr_epochs, 1))), 1)

    def add_observation_noise(self, imgs, first_n_clean, r=0.2, t_ll=0.1, t_lu=0.4, t_ul=0.6, t_uu=0.9):
        """
                给图像添加观测噪声的方法
                :param imgs: 输入图像数据
                :param first_n_clean: 保持前 n 张图像不添加噪声
                :param r: "correlation (over time) factor"
                :param t_ll: 时间步长下界的下界
                :param t_lu: 时间步长下界的上界
                :param t_ul: 时间步长上界的下界
                :param t_uu: 时间步长上界的上界
                :return: 添加噪声后的图像数据
                """
        return add_img_noise(imgs, first_n_clean, self.random, r, t_ll, t_lu, t_ul, t_uu)

    def _get_task_space_pos(self, joint_states):
        """
                获取任务空间位置的方法
                :param joint_states: 关节状态数据
                :return: 任务空间位置数组
                """
        task_space_pos = np.zeros(list(joint_states.shape[:-1]) + [2])
        task_space_pos[..., 0] = np.sin(joint_states[..., 0]) * self.length
        task_space_pos[..., 1] = np.cos(joint_states[..., 0]) * self.length
        return task_space_pos

    def _generate_images(self, ts_pos):
        """
         生成图像的方法
         :param ts_pos: 任务空间位置数据
         :return: 生成的图像数据
         """
        # 创建一个全零数组，用于存储生成的图像
        imgs = np.zeros(shape=list(ts_pos.shape)[
                        :-1] + [self.img_size, self.img_size], dtype=np.uint8)
        # 遍历任务空间位置数组的每个序列和时间步
        for seq_idx in range(ts_pos.shape[0]):
            for idx in range(ts_pos.shape[1]):
                # 使用内部方法生成单个图像，并将其存储在imgs数组中的相应位置
                imgs[seq_idx, idx] = self._generate_single_image(
                    ts_pos[seq_idx, idx])

        return imgs

    def _generate_single_image(self, pos):
        """
                生成单个图像的方法
                :param pos: 单个任务空间位置
                :return: 生成的单个图像
                """
        # 计算图像上的坐标，将位置信息映射到图像坐标空间
        x1 = pos[0] * (self.plt_length / self.length) + self.x0
        y1 = pos[1] * (self.plt_length / self.length) + self.y0
        # 创建灰度图像
        img = Image.new('F', (self.img_size_internal,
                        self.img_size_internal), 0.0)
        draw = ImageDraw.Draw(img)
        # 根据观察模式绘制图像
        if self.observation_mode == Pendulum.OBSERVATION_MODE_LINE:
            draw.line([(self.x0, self.y0), (x1, y1)],
                      fill=1.0, width=self.plt_width)
        elif self.observation_mode == Pendulum.OBSERVATION_MODE_BALL:
            x_l = x1 - self.plt_width
            x_u = x1 + self.plt_width
            y_l = y1 - self.plt_width
            y_u = y1 + self.plt_width
            draw.ellipse((x_l, y_l, x_u, y_u), fill=1.0)
        # 调整图像大小
        img = img.resize((self.img_size, self.img_size),
                         resample=Image.ANTIALIAS)
        # 将图像转换为数组并进行剪裁
        img_as_array = np.asarray(img)
        img_as_array = np.clip(img_as_array, 0, 1)
        # 将图像值缩放到[0, 255]
        return 255.0 * img_as_array

    def _kf_transition_function(self, state, noise):
        """
                KF 状态转移函数
                :param state: 当前状态
                :param noise: 过程噪声
                :return: 下一个状态
                """
        # 计算状态转移的步数
        nSteps = self.dt / self.sim_dt
        # 检查步数是否为整数
        if nSteps != np.round(nSteps):
            print('Warning from Pendulum: dt does not match up')
            nSteps = np.round(nSteps)
        # 计算控制系数
        c = self.g * self.length * self.mass / self.inertia
        # 执行状态转移的迭代过程
        for i in range(0, int(nSteps)):
            velNew = state[1] + self.sim_dt * \
                (c * np.sin(state[0]) - state[1] * self.friction)
            state = np.array([state[0] + self.sim_dt * velNew, velNew])
        # 对角度取模，确保在 [0, 2*pi) 范围内
        state[0] = state[0] % (2 * np.pi)
        # 添加噪声到角速度
        state[1] = state[1] + noise[1]
        return state

    def pendulum_kinematic_single(self, js):
        """
                单个摆的运动学方法
                :param js: 单个摆的关节状态
                :return: 摆的任务空间位置
                """
        # 解析输入数组
        theta, theat_dot = js
        # 计算位置和速度
        x = np.sin(theta)
        y = np.cos(theta)
        x_dot = theat_dot * y
        y_dot = theat_dot * -x
        # 构造包含运动学信息的数组
        return np.array([x, y, x_dot, y_dot]) * self.length

    def pendulum_kinematic(self, js_batch):
        """
                摆的运动学方法
                :param js_batch: 摆的关节状态数据
                :return: 摆的任务空间位置数据
                """
        # 解析输入数组
        theta = js_batch[..., :1]
        theta_dot = js_batch[..., 1:]
        # 计算位置和速度
        x = np.sin(theta)
        y = np.cos(theta)
        x_dot = theta_dot * y
        y_dot = theta_dot * -x
        # 合并结果数组
        return np.concatenate([x, y, x_dot, y_dot], axis=-1)

    def inverse_pendulum_kinematics(self, ts_batch):
        """
            计算逆向摆的运动学，从位置和速度信息中推导出角度和角速度。
            Args:
                ts_batch (numpy.ndarray): 包含多个摆的位置和速度信息的数组。
            Returns:
                numpy.ndarray: 包含多个摆的角度和角速度信息的数组。
            """
        # 解析输入数组
        x = ts_batch[..., :1]
        y = ts_batch[..., 1:2]
        x_dot = ts_batch[..., 2:3]
        y_dot = ts_batch[..., 3:]
        # 计算相关参数
        val = x / y
        theta = np.arctan2(x, y)
        theta_dot_outer = 1 / (1 + val**2)
        theta_dot_inner = (x_dot * y - y_dot * x) / y**2
        # 合并结果数组
        return np.concatenate([theta, theta_dot_outer * theta_dot_inner], axis=-1)


def generate_pendulums(file_path, task, impute_rate=0.5):
    """
        生成摆的数据集，并保存为压缩文件。

        Args:
            file_path (str): 保存数据集的文件路径。
            task (str): 任务类型，可以是 'interpolation' 或 'regression'。
            impute_rate (float, optional): 缺失率，用于生成缺失数据。默认为 0.5。

        Returns:
            None
        """
    if task == 'interpolation':
        # 设置摆的参数
        pend_params = Pendulum.pendulum_default_params()
        pend_params[Pendulum.FRICTION_KEY] = 0.1
        n = 100
        # 创建 Pendulum 实例
        pendulum = Pendulum(24, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                            transition_noise_std=0.1, observation_noise_std=1e-5,
                            seed=42, pendulum_params=pend_params)
        rng = pendulum.random
        # 生成训练数据集
        train_obs, _, _, _, train_ts = pendulum.sample_data_set(
            2000, n, full_targets=False)
        train_obs = np.expand_dims(train_obs, -1)
        train_targets = train_obs.copy()
        train_obs_valid = rng.rand(
            train_obs.shape[0], train_obs.shape[1], 1) > impute_rate
        train_obs_valid[:, :5] = True
        train_obs[np.logical_not(np.squeeze(train_obs_valid))] = 0
        # 生成测试数据集
        test_obs, _, _, _, test_ts = pendulum.sample_data_set(
            1000, n, full_targets=False)
        test_obs = np.expand_dims(test_obs, -1)
        test_targets = test_obs.copy()
        test_obs_valid = rng.rand(
            test_obs.shape[0], test_obs.shape[1], 1) > impute_rate
        test_obs_valid[:, :5] = True
        test_obs[np.logical_not(np.squeeze(test_obs_valid))] = 0
        # 创建保存文件夹
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # 保存数据集为压缩文件
        np.savez_compressed(os.path.join(file_path, f"pend_interpolation_ir{impute_rate}"),
                            train_obs=train_obs, train_targets=train_targets, train_obs_valid=train_obs_valid, train_ts=train_ts,
                            test_obs=test_obs, test_targets=test_targets, test_obs_valid=test_obs_valid, test_ts=test_ts)
    
    elif task == 'regression':
        # 设置摆的参数
        pend_params = Pendulum.pendulum_default_params()
        pend_params[Pendulum.FRICTION_KEY] = 0.1
        pend_params[Pendulum.DT_KEY] = 0.01
        n = 100
        # 创建 Pendulum 实例
        pendulum = Pendulum(24, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                            transition_noise_std=0.1, observation_noise_std=1e-5,
                            seed=42, pendulum_params=pend_params)
        # 生成训练数据集
        train_obs, train_targets, _, _, train_ts = pendulum.sample_data_set(
            2000, n, full_targets=False)
        train_obs, _ = pendulum.add_observation_noise(train_obs, first_n_clean=5, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75,
                                                      t_uu=1.0)
        train_obs = np.expand_dims(train_obs, -1)
        # 生成测试数据集
        test_obs, test_targets, _, _, test_ts = pendulum.sample_data_set(
            1000, n, full_targets=False)
        test_obs, _ = pendulum.add_observation_noise(test_obs, first_n_clean=5, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75,
                                                     t_uu=1.0)
        test_obs = np.expand_dims(test_obs, -1)
        # 创建保存文件夹
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # 保存数据集为压缩文件
        np.savez_compressed(os.path.join(file_path, f"pend_regression.npz"),
                            train_obs=train_obs, train_targets=train_targets, train_ts=train_ts,
                            test_obs=test_obs, test_targets=test_targets, test_ts=test_ts)
