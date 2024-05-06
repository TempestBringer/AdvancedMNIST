import os
import random
import shutil
import cv2
import csv
import numpy as np

def read_image(path: str, resize_x: int, resize_y: int):
    raw_image = cv2.imread(path)
    grayed_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    resized_image: np.ndarray = cv2.resize(grayed_image, (resize_x, resize_y))
    resized_image = resized_image.astype(dtype=np.float32)/255 + 1E-5
    resized_image = resized_image[np.newaxis, :]
    return resized_image

if __name__ == "__main__":
    base_path = "../datasets/HASYv2"
    label_file_path = base_path + "/hasy-data-labels.csv"
    symbol_file_path = base_path + "/symbol.csv"
    image_folder_path = base_path + "/hasy-data/"
    resize_x = 28
    resize_y = 28
    class_to_channel_mapping = {}
    output_class = 16
    train_test_split_ratio = 0.7
    train_datasets = []
    test_datasets = []
    # 打开csv文件
    with open(label_file_path, 'r') as file:
        # 创建csv读取器
        reader = csv.reader(file)
        # 逐行读取csv文件
        for row in reader:
            if row[2] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "[", "]"]:
                img_from_path = base_path + "/" + row[0]
                img_to_path = base_path + "/my/" + row[2] + "/" + row[0].split("/")[1]
                shutil.copy(img_from_path, img_to_path)
                # cur_image_path = base_path + "/" + row[0]
                # labeled_image = read_image(cur_image_path, resize_x, resize_y)
                # cur_image_label = row[2]
                # if not class_to_channel_mapping.__contains__(cur_image_label):
                #     class_to_channel_mapping.setdefault(cur_image_label, len(class_to_channel_mapping.keys()))
                # # print(row)
                # # 打标
                # label_vec = np.zeros(output_class, dtype=np.float32)
                # label_vec[class_to_channel_mapping[cur_image_label]] = 1
                # # 归类
                # if random.random() < train_test_split_ratio:
                #     train_datasets.append((labeled_image, label_vec))
                # else:
                #     test_datasets.append((labeled_image, label_vec))
            elif row[2] == "\\div":
                img_from_path = base_path + "/" + row[0]
                img_to_path = base_path + "/my/div/" + row[0].split("/")[1]
                shutil.copy(img_from_path, img_to_path)
            elif row[2] == "\\times":
                img_from_path = base_path + "/" + row[0]
                img_to_path = base_path + "/my/times/" + row[0].split("/")[1]
                shutil.copy(img_from_path, img_to_path)
    # print(class_to_channel_mapping)