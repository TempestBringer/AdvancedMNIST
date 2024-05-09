import os

import torch
import argparse
import numpy as np

from tqdm import tqdm

from nets import *
from utils.dataset_prepare import HandWrittenMathSymbols


def parse_args():
    parser = argparse.ArgumentParser("手写数学符号数据集的网络训练")
    parser.add_argument("--read_ckpt", type=str, help="权重文件路径，用于继续训练，无则留空")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--save_ckpt_folder", type=str, help="权重文件路径，用于保存")
    parser.add_argument("--save_ckpt_name", type=str, help="权重文件名，用于保存")
    parser.add_argument("--epoch", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr_decay_after_epoch", type=int, default=10, help="lr在该训练轮数后线性衰减")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")

    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备，cuda:0等或cpu")
    parser.add_argument("--image_compress_x", type=int, default=32, help="训练集图片放缩到的大小")
    parser.add_argument("--image_compress_y", type=int, default=32, help="训练集图片放缩到的大小")
    parser.add_argument("--output_class", type=int, default=16, help="最后一层输出向量长度")
    parser.add_argument("--data_load_workers", type=int, default=1, help="数据集加载线程数")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--momentum", type=float, default=0.9, help="动量优化器参数")
    return parser.parse_args()


def find_max_index_in_tensor(x):
    x = x.flatten()
    cur_max = x[0]
    cur_max_index = 0
    for m in range(x.shape[0]):
        if x[m] > cur_max:
            cur_max_index = m
    return cur_max_index


def get_correct_test_count(label, output):
    temp_test_report = np.zeros(shape=(global_args.epoch, global_args.output_class * 2))
    success_sum = 0
    fail_sum = 0
    row_count = label.shape[0]
    for g in range(row_count):
        x_row = label[g]
        y_row = output[g]
        x_index = find_max_index_in_tensor(x_row)
        y_index = find_max_index_in_tensor(y_row)
        temp_test_report[i][x_index * 2] += 1
        if x_index == y_index:
            success_sum += 1
            temp_test_report[i][x_index * 2 + 1] += 1
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
    global_args = parse_args()
    # 设备
    device = global_args.device
    device = torch.device(device)
    if global_args.seed is not None:
        torch.manual_seed(global_args.seed)
        print("seed: " + str(global_args.seed))
    else:
        seed = torch.seed()
        torch.manual_seed(seed)
        print("seed: " + str(seed))
    # 存储ckpt文件夹
    if not os.path.exists(global_args.save_ckpt_folder):
        os.mkdir(global_args.save_ckpt_folder)
    # 实例化网络==============================================================================================【】
    net = SampleNetA(output_class=global_args.output_class, is_training=True)
    # net = SampleNetB(output_class=global_args.output_class, is_training=True)
    # net = AModel(28 * 28, 16)
    # 是否继续训练
    if global_args.read_ckpt is not None:
        net.load_state_dict(torch.load(global_args.read_ckpt))
    # 读取训练集
    dataset = HandWrittenMathSymbols("手写数学符号数据集",
                                     # "./datasets/handwrittenmathsymbols/extracted_images",
                                     "./datasets/HASYv2/my",
                                     global_args.output_class,
                                     global_args.image_compress_x,
                                     global_args.image_compress_y,
                                     train_test_split_ratio=0.7)
    # 打包进dataloader
    train_data_provider = torch.utils.data.DataLoader(dataset.train_datasets,
                                                      batch_size=global_args.batch_size,
                                                      shuffle=True,
                                                      num_workers=global_args.data_load_workers)
    test_data_provider = torch.utils.data.DataLoader(dataset.test_datasets,
                                                     batch_size=1,
                                                     shuffle=True,
                                                     num_workers=global_args.data_load_workers)

    # 损失函数
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    # 优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=global_args.lr, momentum=global_args.momentum)
    # print(optimizer.param_groups)
    # print(type(optimizer.param_groups))
    # lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: get_new_lr(epoch, global_args.epoch, global_args.lr_decay_after_epoch, global_args.lr, 1E-5))
    # 发往显卡
    net = net.to(device)
    criterion = criterion.to(device)

    # 训练循环
    loss = []
    # 测试报告
    test_report = np.zeros(shape=(global_args.epoch, global_args.output_class * 2))
    for i in range(global_args.epoch):
        loss_count = 0
        loss_sum = 0.0
        for j, cur_data in tqdm(enumerate(train_data_provider, 0)):
            input_image, label = cur_data
            input_image = input_image.to(device)
            label = label.to(device)
            cur_output = net(input_image)
            # print(cur_output)
            # print(label)
            # input()
            cur_loss = criterion(cur_output, label)
            net.zero_grad()
            cur_loss.backward()
            optimizer.step()

            loss_count += 1
            loss_sum += cur_loss.item()

        # print("epoch: " + str(i) + " loss: " + str(loss_sum / loss_count))
        print("epoch: " + str(i) + " loss: " + str(loss_sum / loss_count) + " lr: " + str(optimizer.param_groups[0]['lr']))
        # lr_scheduler.step(i)
        optimizer.param_groups[0]['lr'] = get_new_lr(i, global_args.epoch, global_args.lr_decay_after_epoch, global_args.lr, 1E-5)

        if i % 5==0:
            total_try = 0
            success_try = 0
            print("testing")
            for j, cur_test_data in enumerate(test_data_provider, 0):
                test_image, label = cur_test_data
                test_image = test_image.to(device)
                label = label.to(device)
                output = net(test_image)
                # test_output = find_max_index_in_tensor(output.cpu())
                # test_label = find_max_index_in_tensor(label)
                temp_test_report, success_sum, fail_sum = get_correct_test_count(label, output)
                # print(output)
                # print(label)
                # print(test_output)
                # print(test_label)
                # input()
                total_try += success_sum + fail_sum
                success_try += success_sum
                test_report = test_report + temp_test_report

            # print("avg loss: ", cur_loss / loss_count)
            print("success on test set: ", success_try / total_try)
        if i % 20 == 0:
            torch.save(net.state_dict(), global_args.save_ckpt_folder + "/" + global_args.save_ckpt_name)

    # 保存权重
    torch.save(net.state_dict(), global_args.save_ckpt_folder + "/" + global_args.save_ckpt_name)
    # np.save(global_args.save_ckpt_folder + "/current_test_result.npy", test_report)
    np.save(global_args.save_ckpt_folder + "/current_loss.npy", loss)
    np.save(global_args.save_ckpt_folder + "/symbol_mapping.npy", dataset.class_to_channel_mapping)
