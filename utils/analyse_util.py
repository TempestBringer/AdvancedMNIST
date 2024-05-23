import numpy as np


def get_correct_test_count(label, output, global_config: dict):
    """

    :param label:
    :param output:
    :param global_config:
    :return: t_test_report中每一对数都是(标签, 推理结果)
    """
    # 汇报多分类任务下的混淆矩阵，行=标签index，列0-3为tp/fp/tn/fn
    # 汇报多分类任务下的混淆矩阵，行=标签index，列0-1为分类正确/分类错误
    t_test_report = np.zeros(shape=(global_config['output_class'], 2))
    t_success_sum = 0
    t_fail_sum = 0
    row_count = label.shape[0]
    # 结果比对  tp：预测为真实际为真 fp：预测为真实际为假 tn：预测为假实际为假 fn：预测为假实际为真
    # true_positive = 0
    # false_positive = 0
    # true_negative = 0
    # false_negative = 0
    for g in range(row_count):
        x_row = label[g]
        y_row = output[g]
        # 标签值
        x_index = find_max_index_in_tensor(x_row)
        # 预测值
        y_index = find_max_index_in_tensor(y_row)
        if x_index == y_index:
            t_success_sum += 1
            t_test_report[x_index][0] += 1
        else:
            t_fail_sum += 1
            t_test_report[x_index][1] += 1
    return t_test_report, t_success_sum, t_fail_sum

def find_max_index_in_tensor(x):
    x = x.flatten()
    cur_max = x[0]
    cur_max_index = 0
    for m in range(x.shape[0]):
        if x[m] > cur_max:
            cur_max_index = m
    return cur_max_index
