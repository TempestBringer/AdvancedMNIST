import torch.nn as nn


class SampleNet(nn.Module):
    def __init__(self, output_class: int, is_training: bool):
        super().__init__()
        self.layers = nn.Sequential()
        self.output_class = output_class
        self.build_layers()

    def build_layers(self):
        # in_channel, out_channel, kernel_size, stride, padding, padding_mode, dilation, groups, bias
        # 28x28x1 -> 24*24*4
        # self.layers.append(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0, padding_mode="zeros", dilation=1, groups=1, bias=True))
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5))
        self.layers.append(nn.Dropout2d(0.2))
        # 24*24*4 -> 12*12*4
        self.layers.append(nn.MaxPool2d(2))
        # 12*12*4 -> 8*8*16
        # self.layers.append(nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding=0, padding_mode="zeros", dilation=1, groups=1, bias=True))
        self.layers.append(nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5))
        self.layers.append(nn.Dropout2d(0.2))
        # 8*8*16 -> 4*4*16
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(nn.Flatten(1, 3))
        # 4*4*16=256 -> 128
        self.layers.append(nn.Linear(256, 128))
        self.layers.append(nn.ReLU())
        # 128 -> 64
        self.layers.append(nn.Linear(128, 64))
        self.layers.append(nn.ReLU())
        # 64 -> output_class
        self.layers.append(nn.Linear(64, self.output_class))

    def forward(self, x):
        return self.layers.forward(x)
        # for layer in self.layers:
        #     x = layer(x)
        # return x


class SampleNetA(nn.Module):
    def __init__(self, output_class: int, is_training: bool):
        super().__init__()
        self.layers = nn.Sequential()
        self.output_class = output_class
        self.is_training = is_training
        self.build_layers()

    def build_layers(self):
        # in_channel, out_channel, kernel_size, stride, padding, padding_mode, dilation, groups, bias
        # 28x28x1 -> 24*24*8
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5))
        # 24*24*8 -> 12*12*8
        self.layers.append(nn.MaxPool2d(2))
        # 12*12*8 -> 8*8*32
        self.layers.append(nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5))
        # 8*8*32 -> 2048
        self.layers.append(nn.Flatten(1, 3))
        # 2048 -> 256
        self.layers.append(nn.Linear(2048, 256))
        self.layers.append(nn.ReLU())
        # 256 -> 64
        self.layers.append(nn.Linear(256, 64))
        self.layers.append(nn.ReLU())
        # 64 -> output_class
        self.layers.append(nn.Linear(64, self.output_class))

    def forward(self, x):
        return self.layers.forward(x)
        # for layer in self.layers:
        #     x = layer(x)
        # return x