import os
import struct
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

if __name__ == "__main__":
    # Load MNIST dataset
    images = load_mnist_images('train-images.idx3-ubyte')
    labels = load_mnist_labels('train-labels.idx1-ubyte')

    # Save images by label
    save_images_by_label(images, labels, 'mnist_images')