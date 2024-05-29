import argparse
import os.path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from nets import SampleNetB
from utils.analyse_util import get_correct_test_count
from utils.dataset_prepare import HandWrittenMathSymbols
from utils.yaml_util import read_yaml_file


def parse_args():
    parser = argparse.ArgumentParser("手写数学符号数据集的网络训练")
    parser.add_argument("--config", default="./config.yaml", type=str, help="配置文件路径")
    return parser.parse_args()


def run_test(global_config: dict, net=None, state_dict=None, log_read_file=True, test_dataset_provider=None):
    # 设备
    device = global_config['device']
    device = torch.device(device)
    # 标签集映射
    label_mapping = np.load(global_config['save_ckpt_folder'] + "/symbol_mapping.npy", allow_pickle=True).item()
    # 实例化网络
    if net is None:
        net = SampleNetB(output_class=global_config['output_class'], is_training=False)
    if state_dict is None:
        net.load_state_dict(torch.load(global_config['read_ckpt']))
    else:
        net.load_state_dict(state_dict)
    # 读取测试数
    if test_dataset_provider is None:
        read_dataset = HandWrittenMathSymbols("MNIST-测试集",
                                              global_config['dataset_test'],
                                              global_config['output_class'],
                                              global_config["image_compress_x"],
                                              global_config["image_compress_y"],
                                              train_test_split_ratio=0.0,
                                              log_read_file=log_read_file)
        # 打包进dataloader
        test_dataset_provider = torch.utils.data.DataLoader(read_dataset.test_datasets,
                                                         batch_size=global_config['batch_size'],
                                                         shuffle=True,
                                                         num_workers=global_config['data_load_workers'])
    # 发往显卡
    net = net.to(device)
    # 统计数据
    total_try = 0
    success_try = 0
    test_report = np.zeros(shape=(global_config['output_class'], 2))
    for j, cur_data in tqdm(enumerate(test_dataset_provider, 0)):
        input_image, label = cur_data
        input_tensor = torch.tensor(input_image).to(device)
        infer_result: torch.FloatTensor = net(input_tensor)
        infer_result = infer_result.detach().cpu().numpy()
        temp_test_report, success_sum, fail_sum = get_correct_test_count(label, infer_result, global_config)
        total_try += success_sum + fail_sum
        success_try += success_sum
        test_report = test_report + temp_test_report

    print(f"测试总正确率{success_try / total_try}，其中: ", )

    # 保存测试数据
    test_report_save_path = global_config['test_save_path']
    if not os.path.exists(test_report_save_path):
        os.mkdir(test_report_save_path)
    np.save(test_report_save_path + "/test_report.npy", test_report)
    for test_report_row_index in range(global_config['output_class']):
        classify_correct_count = test_report[test_report_row_index][0]
        classify_wrong_count = test_report[test_report_row_index][1]
        print(f"在标签{str(label_mapping[test_report_row_index])}上，分类正确数{classify_correct_count}，"
              f"错误数{classify_wrong_count}，正确率{classify_correct_count / (classify_correct_count + classify_wrong_count)}")

    return total_try, success_try, test_report


if __name__ == "__main__":
    # 解析参数
    parsed_args = parse_args()
    config = read_yaml_file(parsed_args.config)
    epoch_record = []
    success_rate_test = []
    save_ckpt_folder = config['save_ckpt_folder']
    for i in range(config['epoch']):
        config['read_ckpt'] = "./ckpt/final/" + str(i) + "_test.ckpt"
        total_try, success_try, test_report = run_test(global_config=config)
        epoch_record.append(i)
        success_rate_test.append(success_try/total_try)


    plt.plot(epoch_record, success_rate_test, label="success rate on train set")
    plt.xlabel("epoch")
    plt.ylabel("success rate(%)")
    plt.legend()
    plt.savefig(save_ckpt_folder + "/test_success_rate.png")
    plt.clf()
