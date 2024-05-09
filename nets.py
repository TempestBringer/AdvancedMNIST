import torch
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
        self.layers.append(nn.ReLU())
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
        self.layers.append(nn.ReLU())

    def forward(self, x):
        return self.layers.forward(x)
        # for layer in self.layers:
        #     x = layer(x)
        # return x

class AModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_full_layer_unit=200, hidden_compress_layer_unit=40):
        super().__init__()
        # input =
        self.img2linear = nn.Flatten(1, 3)
        self.level_0_adapt = nn.Linear(input_size, hidden_full_layer_unit)
        self.level_0_relu = nn.ReLU()
        self.level_1_fc = nn.Linear(hidden_full_layer_unit, hidden_full_layer_unit)
        self.level_1_relu = nn.ReLU()
        self.level_1_compress = nn.Linear(hidden_full_layer_unit, hidden_compress_layer_unit)
        self.level_2_fc = nn.Linear(hidden_full_layer_unit, hidden_full_layer_unit)
        self.level_2_relu = nn.ReLU()
        self.level_2_compress = nn.Linear(hidden_full_layer_unit, hidden_compress_layer_unit)
        self.level_3_fc = nn.Linear(hidden_full_layer_unit, hidden_full_layer_unit)
        self.level_3_relu = nn.ReLU()
        self.level_4_output = nn.Linear(
            hidden_full_layer_unit + hidden_compress_layer_unit + hidden_compress_layer_unit, output_size)

    def forward(self, x: torch.Tensor):
        x = self.img2linear(x)
        x = self.level_0_adapt(x)
        x = self.level_0_relu(x)
        f1_fc = self.level_1_fc(x)
        f1_fc = self.level_1_relu(f1_fc)
        f2_fc = self.level_2_fc(f1_fc)
        f2_fc = self.level_2_relu(f2_fc)
        s1_compress = self.level_1_compress(f1_fc)
        s2_compress = self.level_2_compress(f2_fc)
        f3_fc = self.level_3_fc(f2_fc)
        f3_fc = self.level_3_relu(f3_fc)
        # print(f3_fc.shape, s1_compress.shape, s2_compress.shape)
        summary = torch.cat((f3_fc, s1_compress, s2_compress), 1)
        out = self.level_4_output(summary)
        return out


if __name__ == "__main__":
    model = AModel(28 * 28, 10)
    print(model)



class SampleNetB(nn.Module):
    def __init__(self, output_class: int, is_training: bool):
        super().__init__()
        self.layers = nn.Sequential()
        self.output_class = output_class
        self.is_training = is_training
        self.build_layers()

    def build_layers(self):
        # in_channel, out_channel, kernel_size, stride, padding, padding_mode, dilation, groups, bias
        # 28x28x1 -> 26*26*8
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3))
        # 26*26*8 -> 13*13*8
        self.layers.append(nn.MaxPool2d(2))
        # 13*13*8 -> 10*10*32
        self.layers.append(nn.Conv2d(in_channels=8, out_channels=32, kernel_size=4))
        # 10*10*32 -> 3200
        self.layers.append(nn.Flatten(1, 3))
        # 3200 -> 256
        self.layers.append(nn.Linear(3200, 320))
        self.layers.append(nn.ReLU())
        # 256 -> 64
        # self.layers.append(nn.Linear(256, 64))
        # self.layers.append(nn.ReLU())
        # 64 -> output_class
        self.layers.append(nn.Linear(320, self.output_class))
        self.layers.append(nn.ReLU())

    def forward(self, x):
        return self.layers.forward(x)
        # for layer in self.layers:
        #     x = layer(x)
        # return x