import torch
from torch import nn, optim
from tqdm import tqdm
from common.layers import Conv2DSeq

class Cube(nn.Module):
    def __init__(self, out_dim, **kwargs):
        super(Cube, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net1, self.net2 = nn.ModuleList(), nn.ModuleList()

        self.out_dim = out_dim

    def train(self, dataloaders_dict, criterion, optimizer, num_epochs=100):
        # 1. Device
        print("Device:", self.device)
        # 2. Network to device
        self.net.to(self.device)
        # 3. Loop over epoch
        for _epoch in range(num_epochs):
            for _phase in ["train", "val"]:
                # if _phase == "train":
                #     self.train()
                # else:
                #     self.eval()

                # 5. Init loss per phase
                _epoch_loss = 0.
                _epoch_correct = 0

                # 7. Iterate dataloader
                for _input_data, _labels in tqdm(dataloaders_dict[_phase]):
                    # 8. Dataset to device
                    _input_data = _input_data.to(self.device)
                    _labels = _labels.to(self.device)

                    # 9. Init grad
                    optimizer.zero_grad()

                    # 10. Forward
                    with torch.set_grad_enabled(
                        mode=(_phase == "train")
                    ):
                        _pred = self.forward(_input_data)

                        # 11. Calc loss
                        _loss = criterion(_pred, _labels)

                        # 12. (Training)Calc grad
                        if _phase == "train":
                            _loss.backword()
                            # 12.1. Update params
                            optimizer.step()

                        # 13. Add loss and correct per minibatch per phase
                        _epoch_loss += _loss.item() * _input_data.size(0) # size 맞나?

        # 14. Print epoch summary
        _epoch_loss /= len(dataloaders_dict[_phase].dataset) # len(dataloader) returns num of data
        print("Epoch loss: {:.4f}".format(_epoch_loss))

    def __build_model(self):
        # Expects 224 * self.out_dim
        # net1
        # 1st stage
        self.net1.append(
            Conv2DSeq(
                in_channels=1,
                out_channels_list=[64, 64],
                kernel_size_list=5,
                stride_list=1,
                dropout_list=.5,
                batch_normalization_list=True,
                activation_list='relu'
            )
        )
        self.net1.append(
            nn.MaxPool2d(
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=0  # kernel_size[0] // 2
            )
        )
        # 2nd stage
        self.net1.append(
            Conv2DSeq(
                in_channels=64,
                out_channels_list=[128, 128],
                kernel_size_list=5,
                stride_list=1,
                dropout_list=.5,
                batch_normalization_list=True,
                activation_list='relu'
            )
        )
        self.net1.append(
            nn.MaxPool2d(
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=0  # kernel_size[0] // 2
            )
        )
        # 3rd stage
        self.net1.append(
            Conv2DSeq(
                in_channels=128,
                out_channels_list=[256, 256],
                kernel_size_list=5,
                stride_list=1,
                dropout_list=.5,
                batch_normalization_list=True,
                activation_list='relu'
            )
        )
        self.net1.append(
            nn.MaxPool2d(
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=0  # kernel_size[0] // 2
            )
        )
        # 4th stage
        self.net1.append(
            Conv2DSeq(
                in_channels=256,
                out_channels_list=[512, 512],
                kernel_size_list=5,
                stride_list=1,
                dropout_list=.5,
                batch_normalization_list=True,
                activation_list='relu'
            )
        )
        self.net1.append(
            nn.MaxPool2d(
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=0  # kernel_size[0] // 2
            )
        )
        # 5th stage
        self.net1.append(
            Conv2DSeq(
                in_channels=512,
                out_channels_list=[512, 512],
                kernel_size_list=5,
                stride_list=1,
                dropout_list=.5,
                batch_normalization_list=True,
                activation_list='relu'
            )
        )
        self.net1.append(
            nn.MaxPool2d(
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=0  # kernel_size[0] // 2
            )
        )
        # net2
        self.net2.append(
            nn.Linear(
                in_features=4096,
                out_features=4096,
                bias=True
            )
        )
        self.net2.append(
            nn.Linear(
                in_features=4096,
                out_features=1000,
                bias=True
            )
        )
        self.net2.append(
            nn.Linear(
                in_features=1000,
                out_features=self.out_dim,
                bias=True
            )
        )

    def forward(self, input_data):
        input_data = self.net1(input_data)
        input_data = torch.flatten(input_data)
        return self.net2(input_data)
