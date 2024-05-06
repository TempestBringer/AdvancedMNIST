import os
import cv2
import csv
import random
import numpy as np
from random import Random
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image


def read_image(path: str, resize_x: int, resize_y: int):
    """
    读取RGB图像转化为灰度并预处理
    :param path:
    :param resize_x:
    :param resize_y:
    :return:
    """
    raw_image = cv2.imread(path)
    grayed_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    resized_image = process_image(grayed_image, resize_x, resize_y)
    return resized_image


def process_image(image: np.ndarray, resize_x: int, resize_y: int):
    """
    处理单通道的图像，进行压缩以及归一、在最前面添加维度
    :param image:
    :param resize_x:
    :param resize_y:
    :return:
    """
    resized_image: np.ndarray = cv2.resize(image, (resize_x, resize_y))
    resized_image = resized_image.astype(dtype=np.float32) / 255 + 1E-5
    resized_image = resized_image[np.newaxis, :]
    return resized_image


class BaseDataSet(object):
    def __init__(self, dataset_name: str, base_path: str, output_class: int, resize_x: int, resize_y: int,
                 train_test_split_ratio: float):
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.output_class = output_class
        self.resize_x = resize_x
        self.resize_y = resize_y
        self.train_test_split_ratio = train_test_split_ratio
        self.train_datasets = []
        self.test_datasets = []
        self.class_to_channel_mapping = {}
        self.read_dataset()

    def _read_dataset(self):
        pass

    def read_dataset(self):
        print("正在读取数据集：" + self.dataset_name)
        self._read_dataset()
        print("数据集" + self.dataset_name + "读取完毕")


class HandWrittenMathSymbols(BaseDataSet):
    def __init__(self, dataset_name: str, base_path: str, output_class: int, resize_x: int, resize_y: int,
                 train_test_split_ratio: float, random_seed=114514):
        """
        :param base_path: 指向分类文件夹的上一级即可
        """
        super().__init__(dataset_name, base_path, output_class, resize_x, resize_y, train_test_split_ratio)
        random.seed(random_seed)

    def _read_dataset(self):
        class_folders = os.listdir(self.base_path)
        for i in range(len(class_folders)):
            label = class_folders[i]
            self.class_to_channel_mapping.setdefault(i, label)
            label_folder_path = self.base_path + "/" + label
            labeled_images = os.listdir(label_folder_path)
            per_class_object_counter = 0
            for j in range(len(labeled_images)):
                labeled_image = labeled_images[j]
                # for labeled_image in labeled_images:
                labeled_image_path = label_folder_path + "/" + labeled_image
                labeled_image = read_image(labeled_image_path, self.resize_x, self.resize_y)
                # label_vec = np.zeros(self.output_class, dtype=np.float32)[np.newaxis, :]
                label_vec = np.zeros(self.output_class, dtype=np.float32)

                # label_vec[0][i] = 1
                label_vec[i] = 1
                if random.random() < self.train_test_split_ratio:
                    self.train_datasets.append((labeled_image, label_vec))
                else:
                    self.test_datasets.append((labeled_image, label_vec))
                # print(label_vec.shape)
                # print(labeled_image.shape)
                # input()
                per_class_object_counter += 1
            print("标签 " + label + " 读取完毕, 共" + str(per_class_object_counter) + "个项目")


if __name__ == "__main__":
    # dataset2 = HandWrittenMathSymbols("HandWrittenMathSymbols", "../datasets/handwrittenmathsymbols/extracted_images", 17, 22, 22)
    dataset2 = HandWrittenMathSymbols("HandWrittenMathSymbols", "../datasets/HASYv2/my", 17, 22, 22, 0.7)
    print(dataset2.class_to_channel_mapping)
    # dataset1 = HASY("Hasy-v2", "../datasets/HASYv2", 17, 22, 22, 0.7)
    # res = dataset2.datasets[100000]
