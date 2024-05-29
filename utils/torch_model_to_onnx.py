import torch

from nets import SampleNetB

if __name__ == "__main__":
    pytorch_net_path = '../ckpt/final/49_test.ckpt'  # 原来模型保存的权重路径
    onnx_net_path = './net.onnx'  # 设置onnx模型保存的权重路径

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 权重导入模型
    net = SampleNetB(10, False).to(device)
    net.load_state_dict(torch.load(pytorch_net_path, map_location=device))
    net.eval()

    input = torch.randn(1, 1, 28, 28).to(device)  # (B,C,H,W)  其中Batch必须为1，因为test时一般为1，尺寸 H,W 必须和训练时的尺寸一致
    torch.onnx.export(net, input, onnx_net_path, verbose=False)
