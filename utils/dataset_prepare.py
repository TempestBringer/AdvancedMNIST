import os
import cv2
import csv
import random
import numpy as np
from random import Random
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image


def read_image(path: str, resize_x: int, resize_y: int, do_resize_and_padding=False, padding_from_ratio = 0.75,
               reverse=False, enwiden_image_line=False):
    """
    读取RGB图像转化为灰度并预处理
    :param path:
    :param resize_x:
    :param resize_y:
    :return:
    """
    raw_image = cv2.imread(path)
    grayed_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(grayed_image)
    # plt.show()
    if do_resize_and_padding:
        resized_grayed_image = cv2.resize(grayed_image, (int(grayed_image.shape[0] * padding_from_ratio), int(grayed_image.shape[1] * padding_from_ratio)))
        top = bottom = int((grayed_image.shape[0] - resized_grayed_image.shape[0]) / 2)
        left = right = int((grayed_image.shape[1] - resized_grayed_image.shape[1]) / 2)
        border_color = (255, 255, 255)  # 边框颜色，这里是黑色
        grayed_image = cv2.copyMakeBorder(resized_grayed_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                            value=border_color)
        # plt.imshow(grayed_image)
        # plt.show()
    # print(grayed_image.shape)
    # input()
    resized_image = process_image(grayed_image, resize_x, resize_y, reverse=reverse)
    if enwiden_image_line:
        enwiden_image = np.zeros(shape=(resized_image.shape[0], resized_image.shape[1], resized_image.shape[2]))
        for pixel_row in range(1, resized_image.shape[1]):
            for pixel_col in range(1, resized_image.shape[2]):
                # print(pixel_row, pixel_col)
                # print(resized_image[0, pixel_row-1:pixel_row+1, pixel_col-1:pixel_col+1])
                pixel_value = np.max(resized_image[0, pixel_row-1:pixel_row+1, pixel_col-1:pixel_col+1])
                enwiden_image[:, pixel_row, pixel_col] = min(pixel_value*1.5, 1)

        resized_image = enwiden_image

    return resized_image


def process_image(image: np.ndarray, resize_x: int, resize_y: int, reverse=False):
    """
    处理单通道的图像，进行压缩以及归一、在最前面添加维度
    :param reverse: 是否反转颜色
    :param image:
    :param resize_x:
    :param resize_y:
    :return:
    """
    resized_image: np.ndarray = cv2.resize(image, (resize_x, resize_y))
    resized_image = resized_image.astype(dtype=np.float32) / 255
    resized_image = resized_image[np.newaxis, :]
    if reverse:
        resized_image = 1 - resized_image
    resized_image = resized_image + 1E-5
    return resized_image


class BaseDataSet(object):
    def __init__(self, dataset_name: str, base_path: str, output_class: int, resize_x: int, resize_y: int,
                 train_test_split_ratio: float, log_read_file=True, reverse=False, do_resize_and_padding=False,
                 padding_from_ratio=0.75, enwiden_image_line=False):
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.output_class = output_class
        self.resize_x = resize_x
        self.resize_y = resize_y
        self.train_test_split_ratio = train_test_split_ratio
        self.train_datasets = []
        self.test_datasets = []
        self.class_to_channel_mapping = {}
        self.log_read_file = log_read_file
        self.reverse = reverse
        self.do_resize_and_padding = do_resize_and_padding
        self.padding_from_ratio = padding_from_ratio
        self.enwiden_image_line = enwiden_image_line
        self.read_dataset()

    def _read_dataset(self):
        pass

    def read_dataset(self):
        if self.log_read_file:
            print("正在读取数据集：" + self.dataset_name)
        self._read_dataset()
        if self.log_read_file:
            print("数据集" + self.dataset_name + "读取完毕")


class HandWrittenMathSymbols(BaseDataSet):
    def __init__(self, dataset_name: str, base_path: str, output_class: int, resize_x: int, resize_y: int,
                 train_test_split_ratio: float, random_seed=114514, log_read_file=True, reverse=False,
                 do_resize_and_padding=False, padding_from_ratio=0.75, enwiden_image_line=False):
        """
        :param base_path: 指向分类文件夹的上一级即可
        """
        super().__init__(dataset_name, base_path, output_class, resize_x, resize_y, train_test_split_ratio,
                         log_read_file, reverse, do_resize_and_padding, padding_from_ratio, enwiden_image_line)
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
                # labeled_image = read_image(labeled_image_path, self.resize_x, self.resize_y, do_resize_and_padding=True)
                labeled_image = read_image(labeled_image_path, self.resize_x, self.resize_y, reverse=self.reverse,
                                           do_resize_and_padding=self.do_resize_and_padding,
                                           padding_from_ratio=self.padding_from_ratio)
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
            if self.log_read_file:
                print("标签 " + label + " 读取完毕, 共" + str(per_class_object_counter) + "个项目")


if __name__ == "__main__":
    # dataset2 = HandWrittenMathSymbols("HandWrittenMathSymbols", "../datasets/handwrittenmathsymbols/extracted_images", 17, 22, 22)
    dataset2 = HandWrittenMathSymbols("HandWrittenMathSymbols", "../datasets/HASYv2/my", 17, 22, 22, 0.7)
    print(dataset2.class_to_channel_mapping)
    # dataset1 = HASY("Hasy-v2", "../datasets/HASYv2", 17, 22, 22, 0.7)
    # res = dataset2.datasets[100000]
