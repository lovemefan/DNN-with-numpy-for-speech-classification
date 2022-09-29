#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @FileName  :dataloader.py
# @Time      :2022/9/29 15:03
# @Author    :lovemefan
# @email     :lovemefan@outlook.com
import os
import zipfile
import requests
from utils.AudioReader import AudioReader
import numpy as np
from matplotlib import pyplot as plot


class DataLoader:
    def __init__(self, data_dir='data', train_radio=0.9):
        self.train_radio = train_radio
        self.absolute_path_dir = os.path.join(os.path.dirname(os.getcwd()), data_dir)
        self.download_dataset()
        self.load_data()

    def download_dataset(self):
        """下载数据集
        如果发现文件损坏，删掉文件重新下载
        数据集来源网站 https://github.com/karoldvl/ESC-50/archive/master.zip
        该数据集一共有50个类别，其中包括2000个样本
        """
        # 以下是数据集各个文件的下载地址
        dataset_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
        dataset_path = self.absolute_path_dir
        if not os.path.exists(os.path.join(dataset_path, 'master.zip')):
            self.download_file(dataset_url, dataset_path)

    def download_file(self, url, file_dir, file_name='master.zip'):
        """从网络上下载文件，如果文件已存在则跳过下载
        :param url 文件网络路径
        :param file_dir 保存文件夹
        :param file_name:  保存文件名
        """
        # 如果不存在就创建文件夹
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

        # 如果下载并解压完成则直接退出
        if os.path.join(file_dir, 'ESC-50-master'):
            return

        # 如果文件不存在
        absolute_path_file = os.path.join(file_dir, file_name)
        if not os.path.exists(absolute_path_file):
            print(f"正在下载{file_name}")
            response = requests.get(url)
            with open(absolute_path_file, 'wb') as f:
                f.write(response.content)

            print(f"{file_name}下载完毕")
            print(f"路径为：{absolute_path_file}")
            # 将文件夹解压开
            print(f"{file_name}正在解压")
            self.unzip(absolute_path_file)
            print(f"{file_name}解压完成")
            print('\n')

    def unzip(self, file_name):
        """ungz zip file"""
        with zipfile.ZipFile(file_name) as zf:
            zf.extractall()

    def load_data(self):
        """从文件中加载数据集"""
        # 加载训练集数据
        print('\033[34m')
        print("正在加载数据中")

        with open(os.path.join(self.absolute_path_dir, 'ESC-50-master', 'meta', 'esc50.csv'), 'r', encoding='utf-8') as f:
            f.readline()
            lines = f.readlines()

        labels = set([item.split(',')[3].strip() for item in lines])
        self.labels2id = {label: index for index, label in enumerate(labels)}
        self.id2labels  = {v: k for k, v in self.labels2id.items()}

        self.dataset = [(os.path.join(self.absolute_path_dir, "ESC-50-master", "audio_16k", item.split(',')[0]),
                         self.labels2id[item.split(',')[3]]) for item in lines]

        self.train_set = self.dataset[:int(len(self.dataset)*self.train_radio)]
        self.test_set = self.dataset[int(len(self.dataset)*self.train_radio):]

        print("加载数据完成")
        print('\033[0m')

    def get_batch(self, dataset, batch_size=None, shuffle=True):
        """返回训练集的一个batch
        :returns (data, label) 返回一个元组，分别是数据矩阵与标签矩阵
        """
        order = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(order)

        order = [i for i in order]

        # 如果没有指定batch_size 就返回所有数据
        if not batch_size:
            batch_size = len(dataset)

        batch = len(dataset) // batch_size
        # 如果训练集个数不是batch_size的整数倍，则batch数加一
        if len(dataset) % batch_size:
            batch += 1

        for i in range(batch):
            data_batch = []
            label_batch = []
            for _ in range(batch_size):
                if len(order) == 0:
                    break
                data_path, label = dataset[order.pop()]
                data_batch.append(AudioReader.read_pcm16(data_path)[0])
                label_batch.append(label)
            yield data_batch, label_batch

    def get_train_batch(self, batch_size=None):
        """返回测试集
        """
        return self.get_batch(self.train_set, batch_size)

    def get_test_data_batch(self):
        """返回测试集
        """
        return self.get_batch(self.test_set, batch_size=len(self.test_set))


    def console_out(self, text, color='b'):
        """向控制台输出有颜色的文字
        :param text 输出的文字
        :param color 输出的颜色，r 红色，b蓝色，g 绿色，y黄色
        """
        if color == 'r':
            print(f"\033[31m{text}\033[0m")
        elif color == 'b':
            print(f"\033[34m{text}\033[0m")
        elif color == 'g':
            print(f"\033[32m{text}\033[0m")
        elif color == 'y':
            print(f"\033[33m{text}\033[0m")
        else:
            print(f"\033[34m{text}\033[0m")


if __name__ == '__main__':
    # 下载数据集
    dataLoader = DataLoader("/home/zlf/dataset")

    for _ in dataLoader.get_train_batch(batch_size=2005):
        pass