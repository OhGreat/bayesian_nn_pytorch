import torch
import torch.nn as nn

from src.modules.linear import BLinear
from src.modules.conv2d import BConv2D


class BCNN(nn.Module):
    def __init__(
        self,
        inference_reps,
        device,
    ) -> None:
        super(BCNN, self).__init__()
        """
        TODO: Write stuff...
        """
        # model name
        self.name = self.__class__.__name__

        self.device = device

        self.layers = nn.Sequential(
            nn.MaxPool2d(2,2),
            BConv2D(
                in_channels=3,
                out_channels=16,
                kernel=(2,2),
                stride=(2,2),
                device=device,
            ),
            nn.MaxPool2d(2,2),
            BConv2D(
                in_channels=16,
                out_channels=16,
                kernel=(2,2),
                stride=(2,2),
                device=device,
            ),
            nn.Flatten(),
            nn.Dropout(0.2),
            BLinear(3136, 64, device=device),
            nn.ReLU(),
            BLinear(64, 2, device=device),
        )

        self.num_params = sum(param.numel() for param in self.parameters())
        
        self.inference_reps = max(inference_reps,1)

    def forward(self, x: torch.Tensor):
        # x = self.layers(x.to(self.device))
        x = torch.stack([self.layers(x) for i in range(self.inference_reps)]).mean(dim=0)
        return x