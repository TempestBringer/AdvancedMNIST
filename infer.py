import argparse

import numpy as np
import torch

from nets import *
from utils.dataset_prepare import read_image, process_image


def parse_args():
    parser = argparse.ArgumentParser("手写数学符号数据集的网络训练")
    parser.add_argument("--read_ckpt", type=str, help="权重文件路径，用于继续训练，无则留空")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备，cuda:0等或cpu")
    parser.add_argument("--image_compress_x", type=int, default=32, help="训练集图片放缩到的大小")
    parser.add_argument("--image_compress_y", type=int, default=32, help="训练集图片放缩到的大小")
    parser.add_argument("--output_class", type=int, default=16, help="最后一层输出向量长度")
    return parser.parse_args()


class AdvancedMNISTInfer(object):
    def __init__(self, ckpt_path: str, symbol_mapping_path: str, output_class: int,
                 compress_x: int, compress_y: int):
        self.ckpt_path = ckpt_path
        self.symbol_mapping_path = symbol_mapping_path
        self.output_class = output_class
        self.compress_x = compress_x
        self.compress_y = compress_y

        # 实例化网络=============================================================================================【】
        # self.net = SampleNetA(self.output_class, False)
        self.net = SampleNetB(self.output_class, False)
        # self.net = AModel(28 * 28, 16)
        # self.net = AModel(32 * 32, 16)

        # 加载权重
        self.net.load_state_dict(torch.load(self.ckpt_path))
        # 加载mapping
        self.symbol_mapping = np.load(symbol_mapping_path, allow_pickle=True).item()
        self.symbol_mapping[12] = "/"
        self.symbol_mapping[13] = "*"
        self.symbol_mapping[14] = "("
        self.symbol_mapping[15] = ")"
    def infer_from_raw_image(self, data: np.ndarray):
        data = process_image(data, self.compress_x, self.compress_y)
        pos = self.infer_from_processed_image(data)
        return self.symbol_mapping[pos]

    def infer_from_processed_image(self, data: np.ndarray):
        infer_result: torch.FloatTensor = self.net(torch.tensor(data[np.newaxis, :]))
        infer_result = infer_result.detach().numpy()
        infer_pos = np.argmax(infer_result)
        return infer_pos


if __name__ == "__main__":
    # 解析参数
    global_args = parse_args()
    global_args.read_ckpt = "./ckpt/test.ckpt"
    global_args.symbol_mapping_path = "./ckpt/symbol_mapping.npy"

    infer_module = AdvancedMNISTInfer(global_args.read_ckpt, global_args.symbol_mapping_path,
                                      global_args.output_class,
                                      global_args.image_compress_x,
                                      global_args.image_compress_y)

    # 读取图片
    img = read_image("E:/Projects/PyCharm/AdvancedMnist/datasets/HASYv2/my/2/v2-00370.png",
                     global_args.image_compress_x,
                     global_args.image_compress_y)

    infer_module.infer_from_processed_image(img)
