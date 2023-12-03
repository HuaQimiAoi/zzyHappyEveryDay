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
# This source code is derived from Latent ODEs for Irregularly-Sampled Time Series (https://github.com/YuliaRubanova/latent_ode)
# Copyright (c) 2019 Yulia Rubanova
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import torch
from sklearn.model_selection import train_test_split
import os
import numpy as np
import tarfile
from torchvision.datasets.utils import download_url


# taken from https://github.com/YuliaRubanova/latent_ode and not modified
class PhysioNet(object):
    # 数据集文件的下载链接
    urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
    ]

    # 结果数据的下载链接
    outcome_urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt']

    # 41个生理参数的列表
    params = [
        'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
        'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
        'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
    ]

    # 将参数名映射到索引
    params_dict = {k: i for i, k in enumerate(params)}

    # 结果标签的列表
    labels = [
        "SAPS-I",
        "SOFA",
        "Length_of_stay",
        "Survival",
        "In-hospital_death"]
    # 将结果标签映射到索引
    labels_dict = {k: i for i, k in enumerate(labels)}

    def __init__(self, root, train=True, download=False,
                 quantization=0.1, n_samples=None, device=torch.device("cpu")):
        """
        初始化函数，用于创建 PhysioNet 类的实例。

        参数：
        - root: 数据集的根目录。
        - train: 如果为 True，则加载训练集，否则加载测试集。
        - download: 如果为 True，则下载数据集。
        - quantization: 时间量化参数，将时间量化为固定的步长。
        - n_samples: 加载数据集的样本数目。
        - device: 数据集加载到的设备，可以是 'cpu' 或 'cuda:0'，分别代表不用GPU和使用GPU

        返回：
        无
        """
        self.root = root
        self.train = train
        self.reduce = "average" # 数据集的降维方式
        self.quantization = quantization # 时间量化的步长

        # 如果指定下载数据集，则调用 download 方法
        if download:
            self.download()

        # 检查数据集是否已经存在，否则抛出运行时错误
        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')

        # 根据训练集或测试集选择数据文件
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        # 根据设备加载数据和标签
        if device == torch.device("cpu"):
            self.data = torch.load(
                os.path.join(
                    self.processed_folder,
                    data_file),
                map_location='cpu')
            self.labels = torch.load(
                os.path.join(
                    self.processed_folder,
                    self.label_file),
                map_location='cpu')
        else:
            self.data = torch.load(
                os.path.join(
                    self.processed_folder,
                    data_file))
            self.labels = torch.load(
                os.path.join(
                    self.processed_folder,
                    self.label_file))

        # 如果指定了样本数目，则截取相应数量的样本
        if n_samples is not None:
            self.data = self.data[:n_samples]
            self.labels = self.labels[:n_samples]

    def download(self):
        """
        下载数据集的方法。首先检查数据集是否已存在，如果存在则直接返回。

        如果数据集不存在，创建文件夹，并根据 GPU 可用性选择设备。然后下载 outcome 数据，
        解析并保存为 PyTorch Tensor 格式。

        参数：
        无

        返回：
        无
        """

        # 如果数据集已存在，则直接返回
        if self._check_exists():
            return

        # 根据 GPU 可用性选择设备
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # 创建 raw_folder 和 processed_folder 文件夹
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # 下载 outcome 数据
        for url in self.outcome_urls:
            # 从 URL 中提取文件名
            filename = url.rpartition('/')[2]
            # 使用 download_url 函数下载数据
            download_url(url, self.raw_folder, filename, None)
            # 构建下载的文本文件路径
            txtfile = os.path.join(self.raw_folder, filename)
            # 读取文本文件内容
            with open(txtfile) as f:
                lines = f.readlines()
                outcomes = {}
                for l in lines[1:]:
                    # 解析每行数据，提取记录 ID 和标签信息
                    l = l.rstrip().split(',')
                    record_id, labels = l[0], np.array(l[1:]).astype(float)
                    outcomes[record_id] = torch.Tensor(labels).to(self.device)
                # 保存 outcome 数据为 PyTorch Tensor 格式
                torch.save(
                    labels,
                    os.path.join(
                        self.processed_folder,
                        filename.split('.')[0] + '.pt')
                )
        # 下载数据集数据
        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename, None)
            # 解压 tar.gz 文件
            tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
            tar.extractall(self.raw_folder)
            tar.close()
            # 打印处理的文件名
            print('Processing {}...'.format(filename))
            # 构建数据目录路径
            dirname = os.path.join(self.raw_folder, filename.split('.')[0])
            patients = []  # 存储患者数据的列表
            total = 0  # 记录总数据数
            for txtfile in os.listdir(dirname):  # 提取记录 ID
                record_id = txtfile.split('.')[0]
                # 打开文件并逐行读取数据
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = 0
                    tt = [0.]  # 存储时间点
                    vals = [torch.zeros(len(self.params)).to(self.device)] # 存储观测值
                    mask = [torch.zeros(len(self.params)).to(self.device)] # 存储掩码
                    nobs = [torch.zeros(len(self.params))] # 存储观测次数
                    for l in lines[1:]:
                        total += 1
                        # 解析每行数据
                        time, param, val = l.split(',')
                        # 将时间转换为小时
                        time = float(time.split(':')[
                                     0]) + float(time.split(':')[1]) / 60.
                        # 四舍五入时间戳（默认为6分钟）
                        time = round(
                            time / self.quantization) * self.quantization

                        if time != prev_time:
                            # 如果时间点发生变化，添加新的时间点和相应的变量
                            tt.append(time)
                            vals.append(torch.zeros(
                                len(self.params)).to(self.device))
                            mask.append(torch.zeros(
                                len(self.params)).to(self.device))
                            nobs.append(torch.zeros(
                                len(self.params)).to(self.device))
                            prev_time = time

                        if param in self.params_dict:
                            # 如果参数在定义的参数字典中
                            n_observations = nobs[-1][self.params_dict[param]]
                            if self.reduce == 'average' and n_observations > 0:
                                # 如果采用平均值策略且已有观测值
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations +
                                           float(val)) / (n_observations + 1)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                # 直接将新的观测值添加到列表中
                                vals[-1][self.params_dict[param]] = float(val)
                            # 更新掩码和观测次数
                            mask[-1][self.params_dict[param]] = 1
                            nobs[-1][self.params_dict[param]] += 1
                        else:
                            # 参数为 'RecordID'，用于检查是否有意外的参数
                            assert param == 'RecordID', 'Read unexpected param {}'.format(
                                param)

                # 转换为 PyTorch Tensor
                tt = torch.tensor(tt).to(self.device)
                vals = torch.stack(vals)
                mask = torch.stack(mask)

                labels = None
                if record_id in outcomes:
                    # 只有训练集有标签
                    labels = outcomes[record_id]
                    # 在 Physionet 提供的5种标签中，仅取最后一种--死亡率
                    labels = labels[4]
                # 将数据保存为 PyTorch Tensor 格式
                patients.append((record_id, tt, vals, mask, labels))
            # 保存处理后的数据
            torch.save(
                patients,
                os.path.join(self.processed_folder,
                             filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            )

        print('Done!')

    # 检查数据集是否已存在
    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition('/')[2]
            # 构建数据文件的路径（包括量化值作为后缀）
            # 检查文件是否存在
            # 如果文件不存在，返回False表示数据集不完整
            if not os.path.exists(
                    os.path.join(self.processed_folder,
                                 filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            ):
                return False
        # 如果所有文件都存在，返回True表示数据集已准备好
        return True

    # 使用@property创建只读属性
    @property
    def raw_folder(self):
        # 返回原始数据存储路径，该路径是根路径和子文件夹'raw'的组合
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        # 返回处理后数据存储路径，该路径是根路径和子文件夹'processed'的组合
        return os.path.join(self.root, 'processed')

    @property
    def training_file(self):
        # 返回训练集数据文件的文件名，包括量化值作为后缀
        return 'set-a_{}.pt'.format(self.quantization)

    @property
    def test_file(self):
        # 返回测试集数据文件的文件名，包括量化值作为后缀
        return 'set-b_{}.pt'.format(self.quantization)

    @property
    def label_file(self):
        # 返回标签文件的文件名
        return 'Outcomes-a.pt'

    def __getitem__(self, index):
        # 通过索引获取数据集中特定位置的数据
        return self.data[index]

    def __len__(self):
        # 返回数据集中数据点的总数
        return len(self.data)

    def get_label(self, record_id):
        # 根据记录 ID 获取对应的标签
        return self.labels[record_id]

    def __repr__(self):
        # 返回数据集的描述字符串，包括数据点数量、数据集划分、根路径、量化值和降采样策略
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(
            'train' if self.train is True else 'test')
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Quantization: {}\n'.format(self.quantization)
        fmt_str += '    Reduce: {}\n'.format(self.reduce)
        return fmt_str


# 划分训练集、测试集、验证集
def train_test_valid_split(input_path):
    # 从文件加载set-a和set-b数据，合并为一个列表
    a = torch.load(os.path.join(input_path, 'set-a_0.1.pt'))
    b = torch.load(os.path.join(input_path, 'set-b_0.1.pt'))
    data = a + b
    # 使用train_test_split函数划分数据集为训练集、验证集和测试集
    train_valid, test = train_test_split(data, test_size=0.2, random_state=0)
    train, valid = train_test_split(
        train_valid, test_size=0.25, random_state=0)
    return train, train_valid, valid, test


# 提取时变特征
def remove_timeinvariant_features(input_path, name):
    # 从文件加载数据
    data = torch.load(
        os.path.join(
            input_path,
            name + '.pt'),
        map_location='cpu')
    data_timevariant = []
    for sample in data:
        obs = sample[2]
        mask = sample[3]
        obs_timevariant = obs[:, 4:] # 从第5列开始截取时变特征
        mask_timevariant = mask[:, 4:]
        data_timevariant.append(
            (sample[0], sample[1], obs_timevariant, mask_timevariant, sample[4]))
    return data_timevariant


# 对数据进行归一化和保存
def normalize_data_and_save(input_path, name):
    data = torch.load(
        os.path.join(
            input_path,
            name + '.pt'),
        map_location='cpu')
    # 获取数据集中时变特征的最小和最大值
    min_value, max_value = get_min_max_physionet(data)

    data_normalized = []
    # 遍历数据集的每个样本，对时变特征进行归一化
    for sample in data:
        obs = sample[2]
        mask = sample[3]
        obs_normalized = normalize_obs(obs, mask, min_value, max_value)
        data_normalized.append(
            (sample[0], sample[1], obs_normalized, sample[3], sample[4]))
    return data_normalized


# 用于将观测数据进行归一化
def normalize_obs(obs, mask, min_value, max_value):
    # 确保数据的最后一个维度相同
    assert obs.shape[-1] == min_value.shape[-1] == max_value.shape[-1], 'Dimension missmatch'
    # 将max_value中值为0的元素设为1，避免除零错误
    max_value[max_value == 0] = 1
    # 归一化计算
    obs_norm = (obs - min_value) / (max_value - min_value)
    # 将mask为0的位置设为0
    obs_norm[mask == 0] = 0
    return obs_norm


# 定义新组件，用于获取Physionet数据集中时变特征的最小和最大值
def get_min_max_physionet(data):
    # 将所有样本的时变特征拼接为一个张量
    obs = torch.cat([sample[2] for sample in data])
    # 计算时变特征的最小值和最大值
    min_value, _ = torch.min(obs, dim=0)
    max_value, _ = torch.max(obs, dim=0)
    return min_value, max_value

# 定义函数，用于下载和处理Physionet数据集
def download_and_process_physionet(file_path):
    # 初始化Physionet类实例以下载数据集
    dataset = PhysioNet(file_path, train=False, download=True)
    processed_path = os.path.join(file_path, 'processed')

    # 定义数据集划分的标签
    sets = ['train', 'train_valid', 'test', 'valid']
    # 划分并保存训练集、验证集和测试集
    train, train_valid, valid, test = train_test_valid_split(processed_path)
    torch.save(train, os.path.join(processed_path, 'train.pt'))
    torch.save(train_valid, os.path.join(processed_path, 'train_valid.pt'))
    torch.save(valid, os.path.join(processed_path, 'valid.pt'))
    torch.save(test, os.path.join(processed_path, 'test.pt'))

    # 对每个数据集划分进行处理
    for set in sets:
        # 移除时不变特征
        data_timevariant = remove_timeinvariant_features(
            processed_path, name=set)
        torch.save(
            data_timevariant,
            os.path.join(
                processed_path,
                f'f37_{set}.pt'))

        # 归一化数据并保存
        data_normalized = normalize_data_and_save(
            processed_path, name=f'f37_{set}')
        torch.save(data_normalized, os.path.join(file_path, f'norm_{set}.pt'))
