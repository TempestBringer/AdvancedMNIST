import os
import yaml
import torch
import argparse
import numpy as np

from tqdm import tqdm

from nets import *
from utils.yaml_util import read_yaml_file
from utils.dataset_prepare import HandWrittenMathSymbols


def parse_args():
    parser = argparse.ArgumentParser("手写数学符号数据集的网络训练")
    parser.add_argument("--config", default="./config.yaml", type=str, help="配置文件路径")
    return parser.parse_args()


def find_max_index_in_tensor(x):
    x = x.flatten()
    cur_max = x[0]
    cur_max_index = 0
    for m in range(x.shape[0]):
        if x[m] > cur_max:
            cur_max_index = m
    return cur_max_index


def get_correct_test_count(label, output, global_config: dict):
    temp_test_report = np.zeros(shape=(global_config['epoch'], global_config['output_class'] * 2))
    success_sum = 0
    fail_sum = 0
    row_count = label.shape[0]
    for g in range(row_count):
        x_row = label[g]
        y_row = output[g]
        x_index = find_max_index_in_tensor(x_row)
        y_index = find_max_index_in_tensor(y_row)
        temp_test_report[g][x_index * 2] += 1
        if x_index == y_index:
            success_sum += 1
            temp_test_report[g][x_index * 2 + 1] += 1
        else:
            fail_sum += 1
    return (temp_test_report, success_sum, fail_sum)


def get_new_lr(current_epoch: int, total_epoch: int, decay_epoch: int, initial_lr: float, target_lr: float) -> float:
    if current_epoch < decay_epoch:
        return initial_lr
    else:
        k = (target_lr - initial_lr) / (total_epoch - decay_epoch)
        return initial_lr + (current_epoch - decay_epoch) * k


if __name__ == "__main__":
    # 解析参数
    parsed_args = parse_args()
    config = read_yaml_file(parsed_args.config)
    # 设备
    device = config['device']
    device = torch.device(device)
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
        print("seed: " + str(config['seed']))
    else:
        seed = torch.seed()
        torch.manual_seed(seed)
        config['seed'] = seed
        print("seed: " + str(seed))

    epoch = config['epoch']
    lr = config['lr']
    save_ckpt_folder = config['save_ckpt_folder']
    save_ckpt_name = config['save_ckpt_name']

    # 存储ckpt文件夹
    if not os.path.exists(config['save_ckpt_folder']):
        os.mkdir(config['save_ckpt_folder'])
    # 实例化网络==============================================================================================【】
    net = SampleNetB(output_class=config['output_class'], is_training=True)

    # 是否继续训练
    if os.path.exists(config['read_ckpt']):
        net.load_state_dict(torch.load(config['read_ckpt']))
    # 读取训练集
    dataset = HandWrittenMathSymbols("手写数学符号数据集",
                                     config['dataset_train'],
                                     config['output_class'],
                                     config["image_compress_x"],
                                     config["image_compress_y"],
                                     train_test_split_ratio=1.0)
    # 打包进dataloader
    train_data_provider = torch.utils.data.DataLoader(dataset.train_datasets,
                                                      batch_size=config['batch_size'],
                                                      shuffle=True,
                                                      num_workers=config['data_load_workers'])

    # 损失函数
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    # 优化器==============================================================================================================【】
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=config['momentum'])
    # optimizer = torch.optim.Adam(net.parameters(), lr=global_args.lr,amsgrad=True)
    # 发往显卡
    net = net.to(device)
    criterion = criterion.to(device)

    # 训练循环
    loss = []
    success_rate_train = []
    success_rate_test = []
    # 测试报告
    test_report = np.zeros(shape=(epoch, config['output_class'] * 2))
    for i in range(epoch):
        loss_count = 0
        loss_sum = 0.0
        for j, cur_data in tqdm(enumerate(train_data_provider, 0)):
            input_image, label = cur_data
            input_image = input_image.to(device)
            label = label.to(device)
            cur_output = net(input_image)
            cur_loss = criterion(cur_output, label)
            net.zero_grad()
            cur_loss.backward()
            optimizer.step()

            loss_count += 1
            loss_sum += cur_loss.item()

        print("epoch: " + str(i) + " loss: " + str(loss_sum / loss_count) + " lr: " + str(optimizer.param_groups[0]['lr']))
        # lr_scheduler.step(i)
        optimizer.param_groups[0]['lr'] = get_new_lr(i, epoch, config['lr_decay_after_epoch'], lr, 1E-5)

        if i % config['test_on_train_set_interval'] == 0:
            total_try = 0
            success_try = 0
            print("testing on train set:")
            for j, cur_test_data in enumerate(train_data_provider, 0):
                test_image, label = cur_test_data
                test_image = test_image.to(device)
                label = label.to(device)
                output = net(test_image)
                temp_test_report, success_sum, fail_sum = get_correct_test_count(label, output, config)
                total_try += success_sum + fail_sum
                success_try += success_sum
                test_report = test_report + temp_test_report

            print("success on train set: ", success_try / total_try)

        if i % config['save_ckpt_interval'] == 0:
            torch.save(net.state_dict(), save_ckpt_folder + "/" + save_ckpt_name)

    # 保存权重
    torch.save(net.state_dict(), save_ckpt_folder + "/" + save_ckpt_name)
    np.save(save_ckpt_folder + "/current_loss.npy", loss)
    np.save(save_ckpt_folder + "/symbol_mapping.npy", dataset.class_to_channel_mapping)
