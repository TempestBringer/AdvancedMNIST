import os
import argparse
import numpy as np

from tqdm import tqdm

from nets import *
from test import run_test
from utils.yaml_util import read_yaml_file
from utils.dataset_prepare import HandWrittenMathSymbols


def parse_args():
    parser = argparse.ArgumentParser("手写数学符号数据集的网络训练")
    parser.add_argument("--config", default="./config.yaml", type=str, help="配置文件路径")
    return parser.parse_args()


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
    try:
        if os.path.exists(config['read_ckpt']):
            net.load_state_dict(torch.load(config['read_ckpt']))
    except:
        pass
    # 读取训练集
    train_dataset = HandWrittenMathSymbols("MNIST-训练集",
                                           config['dataset_train'],
                                           config['output_class'],
                                           config["image_compress_x"],
                                           config["image_compress_y"],
                                           train_test_split_ratio=1.0)
    # 打包进dataloader
    train_data_provider = torch.utils.data.DataLoader(train_dataset.train_datasets,
                                                      batch_size=config['batch_size'],
                                                      shuffle=True,
                                                      num_workers=config['data_load_workers'])

    # 损失函数
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    # 优化器==============================================================================================================【】
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=config['momentum'])
    # 发往显卡
    net = net.to(device)
    criterion = criterion.to(device)

    # 训练循环
    loss = []
    success_rate_train = []
    success_rate_test = []

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

        if i % config['test_on_train_set_interval'] == config['test_on_train_set_interval'] - 1:
            print("在训练集上测试")
            run_test(config, net.state_dict(), log_read_file=False, test_dataset_provider=train_data_provider)


        if i % config['save_ckpt_interval'] == config['save_ckpt_interval'] - 1:
            ckpt_final_save_path = save_ckpt_folder + "/" + save_ckpt_name
            print(f"saving ckpt to {ckpt_final_save_path}")
            torch.save(net.state_dict(), ckpt_final_save_path)

    # 保存权重
    torch.save(net.state_dict(), save_ckpt_folder + "/" + save_ckpt_name)
    np.save(save_ckpt_folder + "/current_loss.npy", loss)
    np.save(save_ckpt_folder + "/symbol_mapping.npy", train_dataset.class_to_channel_mapping)

