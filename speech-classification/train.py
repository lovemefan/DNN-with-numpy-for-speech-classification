#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @FileName  :train.py
# @Time      :2022/9/29 15:27
# @Author    :lovemefan
# @email     :lovemefan@outlook.com
import os.path
import pickle
import argparse
from tqdm import trange, tqdm
from dataloader import DataLoader
from model import Model
from matplotlib import pyplot as plot

def init_args():
    """初始化输入参数"""
    parser = argparse.ArgumentParser(description="输入参数")
    parser.add_argument('--data', help='data path', required=True)
    parser.add_argument('--input_dim', help='dimension of input', default=80017)
    parser.add_argument('--hidden_dim', help='dimension of hidden layer', default=1024)
    parser.add_argument('--output_dim', help='dimension of output', default=50)
    parser.add_argument('--batch_size', help='batch_size', default=256)
    parser.add_argument('--learning_rate', help='learning_rate', default=2e-4)
    parser.add_argument('--parameters', help='parameters file', default="parameters.pkl")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # 初始化模型中所有的参数
    args = init_args()

    parameters = None
    # 读取参数
    if args.parameters:
        if os.path.exists('parameters.pkl'):
            with open('parameters.pkl', 'rb') as file_to_read:
                # 通过pickle的load函数读取data1.pkl中的对象，并赋值给data2
                parameters = pickle.load(file_to_read)

    # 初始化模型参数
    model = Model(args, parameters)
    dataLoader = DataLoader(args.data)
    batch_generator = dataLoader.get_train_batch(int(args.batch_size))
    epoch_size = 1

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    for img_batch, label_batch in batch_generator:
        for index in trange(len(img_batch), desc="batch"):
            model.train_batch(img_batch, label_batch, args.learning_rate)
            # 训练集平均loss
            train_loss_temp = model.loss_function(img_batch, label_batch) / len(label_batch)
            train_loss.append(train_loss_temp)
            # 测试集平均loss
            test_dataset = dataLoader.get_test_data_batch().__next__()
            test_loss_temp = model.loss_function(test_dataset[0],
                                                 test_dataset[1]) / len(label_batch)
            test_loss.append(test_loss_temp)

            # 用训练集检验准确率
            train_data = (img_batch, label_batch)
            train_data_size = len(train_data[1])
            tmp = [model.predict_label(train_data[0][i]).argmax() == train_data[1][i] for i in range(train_data_size)]
            train_accuracy_tmp = tmp.count(True) / len(tmp)
            train_accuracy.append(train_accuracy_tmp)

            # 利用测试集检验准确率
            test_data = (test_dataset[0], test_dataset[1])
            test_data_size = len(test_data[1])
            tmp = [model.predict_label(test_data[0][i]).argmax() == test_data[1][i] for i in range(test_data_size)]
            test_accuracy_temp = tmp.count(True)/len(tmp)
            test_accuracy.append(test_accuracy_temp)

            print(f"train_loss:{train_loss_temp}")
            print(f"test_loss:{test_loss_temp}")
            print(f"train_accuracy:{train_accuracy_tmp}")
            print(f"test_accuracy:{test_accuracy_temp}")


        # 保存参数
        # 打开(或创建)一个名为parameters.pkl的文件，打开方式为二进制写入(参数‘wb’)
        with open("parameters.pkl", "wb") as  file_to_save:
            # 通过pickle模块中的dump函数将data1保存到pkl文件中。
            # 第三个参数为序列化使用的协议版本，0：ASCII协议，所序列化的对象使用可打印的ASCII码表示；1：老式的二进制协议；2：2.3版本引入的新二进制协议，较以前的更高效；-1：使用当前版本支持的最高协议。其中协议0和1兼容老版本的python。protocol默认值为0。
            pickle.dump(model.parameters, file_to_save, -1)
        with open("result.pkl", "wb") as  file_to_save:
            # 通过pickle模块中的dump函数将data1保存到pkl文件中。
            # 第三个参数为序列化使用的协议版本，0：ASCII协议，所序列化的对象使用可打印的ASCII码表示；1：老式的二进制协议；2：2.3版本引入的新二进制协议，较以前的更高效；-1：使用当前版本支持的最高协议。其中协议0和1兼容老版本的python。protocol默认值为0。
            pickle.dump({'train_loss': train_loss, "test_loss": test_loss,
                         "train_accuracy":train_accuracy, "test_accuracy": test_accuracy}, file_to_save, -1)

    # loss
    plot.plot(range(len(train_loss)), train_loss, label='train_loss')
    plot.plot(range(len(test_loss)), test_loss, label='test_loss')
    plot.legend()  # 让图例生效
    plot.xlabel('batch')  # X轴标签
    plot.ylabel("loss")  # Y轴标签
    plot.savefig('loss.png')
    # accuracy
    plot.plot(range(len(train_accuracy)), train_accuracy, label='train_accuracy')
    plot.plot(range(len(test_accuracy)), test_accuracy, label='test_accuracy')
    plot.legend()  # 让图例生效
    plot.xlabel('batch')  # X轴标签
    plot.ylabel("accuracy")  # Y轴标签
    plot.savefig('accuracy.png')