from torch import nn

class Conv2DSeq(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 stride_list=1,
                 dropout_list=.5,
                 batch_normalization_list=True,
                 activation_list='relu'):
        super(Conv2DSeq, self).__init__()
        if type(kernel_size_list) == int:
            kernel_size_list = [kernel_size_list] * len(out_channels_list)
        stride_list = stride_list
        if type(stride_list) == int:
            stride_list = [stride_list] * len(out_channels_list)
        elif type(stride_list) in (list, tuple):
            for i, stride in enumerate(stride_list):
                if type(stride) == int:
                    pass
                elif type(stride) == list:
                    stride_list[i] = tuple(stride)

                if type(stride) in (tuple, list):
                    assert len(stride) <= 2
                else:
                    raise ValueError('Argument not understood')
        if type(dropout_list) == float:
            dropout_list = [dropout_list] * len(out_channels_list)
        if type(batch_normalization_list) == bool:
            batch_normalization_list = [batch_normalization_list] * len(out_channels_list)
        if type(activation_list) == str:
            activation_list = [activation_list] * len(out_channels_list)

        self.layers = nn.ModuleList([
            Conv2DNorm(i, o, k, s, padding='SAME', dropout=d, batch_normalization=bn, activation=a)
            for (i, o, k, s, d, bn, a) in zip(
                [in_channels] + out_channels_list[:-1],
                out_channels_list,
                kernel_size_list,
                stride_list,
                dropout_list,
                batch_normalization_list,
                activation_list)
        ])

    def __len__(self):
        return len(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Conv2DNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding='SAME', dropout=.5,
                 batch_normalization=True, activation=None):
        super(Conv2DNorm, self).__init__()
        if type(kernel_size) == int:
            kernel_size = [kernel_size, 1]
        if padding == 'SAME':
            padding = (kernel_size[0] - 1) // 2, 0
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        if dropout:
            self.layers.append(nn.Dropout2d(p=dropout))
        if batch_normalization:
            self.layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
