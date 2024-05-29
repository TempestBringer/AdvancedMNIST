import argparse

import numpy as np
import torch

from nets import *
from utils.dataset_prepare import read_image, process_image
from utils.yaml_util import read_yaml_file


def parse_args():
    parser = argparse.ArgumentParser("手写数学符号数据集的网络训练")
    parser.add_argument("--config", default="./config.yaml", type=str, help="配置文件路径")
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
        # self.symbol_mapping[12] = "/"
        # self.symbol_mapping[13] = "*"
        # self.symbol_mapping[14] = "("
        # self.symbol_mapping[15] = ")"
    def infer_from_raw_image(self, data: np.ndarray, reverse=False):
        data = process_image(data, self.compress_x, self.compress_y, reverse=reverse)
        pos = self.infer_from_processed_image(data)
        return self.symbol_mapping[pos]

    def infer_from_processed_image(self, data: np.ndarray):
        infer_result: torch.FloatTensor = self.net(torch.tensor(data[np.newaxis, :]))
        infer_result = infer_result.detach().numpy()
        infer_pos = np.argmax(infer_result)
        return infer_pos


if __name__ == "__main__":
    # 解析参数
    parsed_args = parse_args()
    config = read_yaml_file(parsed_args.config)

    infer_module = AdvancedMNISTInfer(config['read_ckpt'], config['symbol_mapping_path'],
                                      config['output_class'],
                                      config['image_compress_x'],
                                      config['image_compress_y'])

    # 读取图片
    img = read_image("E:/Projects/PyCharm/AdvancedMnist/datasets/MNIST/test/6/66.png",
    # img = read_image("E:/Projects/PyCharm/AdvancedMnist/datasets/HASYv2/my/2/v2-00370.png",
                     config['image_compress_x'],
                     config['image_compress_y'],
                     reverse=False)

    pos = infer_module.infer_from_processed_image(img)
    print(pos)
