import yaml


def read_yaml_file(yaml_file_path: str, print_argument=True):
    # 打开yaml文件
    file = open(yaml_file_path, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()

    # 将字符串转化为字典或列表
    yml_data = yaml.load(file_data, Loader=yaml.FullLoader)
    if print_argument:
        print(yml_data)
    return yml_data


if __name__ == "__main__":
    data = read_yaml_file("../config.yaml")
    print(data['device'])
    print(type(data['dataset_do_split']))
