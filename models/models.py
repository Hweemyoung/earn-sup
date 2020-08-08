import torch
from torch import nn, optim
from tqdm import tqdm

class Cube(nn.Module):
    def __init__(self, **kwargs):
        super(Cube, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, dataloaders_dict, criterion, optimizer, num_epochs=100):
        # 1. Device
        print("Device:", self.device)
        # 2. Network to device
        self.to(self.device)
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

