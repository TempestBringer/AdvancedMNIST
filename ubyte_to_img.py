import os
import struct
from array import array
from os.path import join

import cv2
import numpy as np
from PIL import Image


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "Invalid magic number"
        image_data = np.fromfile(f, dtype=np.uint8).reshape(-1, rows, cols)
        # 黑底变白底
        image_data = -image_data + 255
    return image_data

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049, "Invalid magic number"
        label_data = np.fromfile(f, dtype=np.uint8)
    return label_data

def save_images_by_label(images, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for label in range(10):  # There are 10 labels (0 to 9)
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        label_indices = np.where(labels == label)[0]
        for i, idx in enumerate(label_indices):
            image = Image.fromarray(images[idx])
            image_path = os.path.join(label_dir, f"{label}_{i}.png")
            # Save image as PNG file
            image.save(image_path)

class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


def dataset_to_file(base_dir: str, x_train_datas, y_train_datas, x_test_datas, y_test_datas):
    assert len(x_train_datas) == len(y_train_datas)
    assert len(x_test_datas) == len(y_test_datas)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(base_dir + "/train"):
        os.mkdir(base_dir + "/train")
    if not os.path.exists(base_dir + "/test"):
        os.mkdir(base_dir + "/test")
    for i in range(10):
        if not os.path.exists(base_dir + "/train/" + str(i)):
            os.mkdir(base_dir + "/train/" + str(i))
        if not os.path.exists(base_dir + "/test/" + str(i)):
            os.mkdir(base_dir + "/test/" + str(i))

    for train_index in range(len(x_train_datas)):
        img = np.asarray(x_train_datas[train_index])
        label = y_train_datas[train_index]
        save_path = base_dir + "/train/" + str(label) + "/" + str(train_index) + ".png"
        cv2.imwrite(save_path, img)

    for test_index in range(len(x_test_datas)):
        img = np.asarray(x_test_datas[test_index])
        label = y_test_datas[test_index]
        save_path = base_dir + "/test/" + str(label) + "/" + str(test_index) + ".png"
        cv2.imwrite(save_path, img)


if __name__ == "__main__":
    input_path = './mnist'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    dataset_to_file("./img_dataset", x_train, y_train, x_test, y_test)

# if __name__ == "__main__":
#     # Load MNIST dataset
#     images = load_mnist_images('train-images.idx3-ubyte')
#     labels = load_mnist_labels('train-labels.idx1-ubyte')
#
#     # Save images by label
#     save_images_by_label(images, labels, 'mnist_images')