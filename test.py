import argparse

import numpy as np
import torch
from tqdm import tqdm

from nets import SampleNetB
from train import get_correct_test_count
from utils.dataset_prepare import HandWrittenMathSymbols, process_image
from utils.yaml_util import read_yaml_file


def parse_args():
    parser = argparse.ArgumentParser("手写数学符号数据集的网络训练")
    parser.add_argument("--config", default="./config.yaml", type=str, help="配置文件路径")
    return parser.parse_args()


if __name__ == "__main__":
    # 解析参数
    parsed_args = parse_args()
    config = read_yaml_file(parsed_args.config)
    # 设备
    device = config['device']
    device = torch.device(device)
    # 实例化网络
    net = SampleNetB(output_class=config['output_class'], is_training=False)
    net.load_state_dict(torch.load(config['read_ckpt']))
    # 读取测试数
    dataset = HandWrittenMathSymbols("手写数学符号数据集",
                                     config['dataset_test'],
                                     config['output_class'],
                                     config["image_compress_x"],
                                     config["image_compress_y"],
                                     train_test_split_ratio=0.0)
    # 打包进dataloader
    test_data_provider = torch.utils.data.DataLoader(dataset.test_datasets,
                                                      batch_size=config['batch_size'],
                                                      shuffle=True,
                                                      num_workers=config['data_load_workers'])
    # 发往显卡
    net = net.to(device)
    # 统计数据
    total_try = 0
    success_try = 0
    for j, cur_data in tqdm(enumerate(test_data_provider, 0)):
        input_image, label = cur_data
        input_tensor = torch.tensor(input_image).to(device)
        infer_result: torch.FloatTensor = net(input_tensor)
        infer_result = infer_result.detach().cpu().numpy()
        temp_test_report, success_sum, fail_sum = get_correct_test_count(label, infer_result, config)
        total_try += success_sum + fail_sum
        success_try += success_sum

    print("success on test set: ", success_try / total_try)


