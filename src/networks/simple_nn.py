import torch
import torch.nn as nn
from src.modules.linear import BLinear

class simpleBayesian(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim,
        output_dim,
        inference_reps,
        device,
        ):
        super(simpleBayesian, self).__init__()
        """
        TODO: Write stuff...
        """
        # model name
        self.name = self.__class__.__name__

        self.device = device

        self.layers = nn.Sequential(
            BLinear(input_dim, hid_dim, device=device),
            nn.ReLU(),
            nn.Dropout(0.2),
            # BLinear(hid_dim, hid_dim, device=device),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            BLinear(hid_dim, output_dim, device=device),
        )

        self.num_params = sum(param.numel() for param in self.parameters())
        
        self.inference_reps = max(inference_reps,1)

    def forward(self, x: torch.Tensor):
        # x = self.layers(x.to(self.device))
        x = torch.stack([self.layers(x) for i in range(self.inference_reps)]).mean(dim=0)
        return x


class simpleNN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, device):
        super(simpleNN, self).__init__()

        # model name
        self.name = self.__class__.__name__

        self.device = device

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hid_dim, device=device),
            nn.ReLU(),
            # nn.Dropout(0.2),
            # BLinear(hid_dim, hid_dim, device=device),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(hid_dim, output_dim, device=device),
        )

        self.num_params = sum(param.numel() for param in self.parameters())
        
    def forward(self,x: torch.Tensor):
        x = self.layers(x.to(self.device))
        return x