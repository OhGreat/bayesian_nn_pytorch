import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BConv2D(nn.Module):

    def __init__(
        self,
        mu_prior: float = 0,
        sigma_prior: float = 0.1,
        mu_posterior: Tuple[float, float] = (0, 0.1),
        sigma_posterior: Tuple[float, float] = (0, 0.1),
        bias: bool = True,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel: Tuple[int, int] = (2, 2),
        stride: Tuple[int, int] = (1, 1),
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        device: str = None,
    ) -> None:
        """
        TODO: Describe stuff...
        Args: 

        
        """
        super(BConv2D, self).__init__()
        # bayesian net params
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.log_sigma_prior = math.log(sigma_prior)
        self.mu_posterior = mu_posterior
        self.sigma_posterior = sigma_posterior
        self.bias = bias
        # convolution params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = max(dilation, 1)
        self.groups = max(groups, 1)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else: self.device = device 

        # each weight is represented by a tuple of mu and sigma parameters of a normal distribution
        # the actual weights are sampled from the normal distribution in the forward pass
        self.W_mu = nn.Parameter(
            torch.empty((
                out_channels,
                in_channels//self.groups,
                *kernel),
                device=device,
            )
        )
        self.W_sigma = nn.Parameter(
            torch.empty((
                out_channels,
                in_channels//self.groups,
                *kernel),
                device=device,
            )
        )
        if self.bias:
            self.bias_mu = nn.Parameter(
                torch.empty(
                    out_channels,
                    device=self.device,
                )
            )
            self.bias_sigma = nn.Parameter(
                torch.empty(
                    out_channels,
                    device=device,
                )
            )
        self.reset_parameters()  # initialize weight's values

    def to_str(self):
        return f"in features: {self.in_channels}, out features: {self.out_channels}\n\
            mu prior: {self.mu_prior}, sigma prior: {self.sigma_prior}\n\
            mu posterior: {self.mu_posterior}, sigma posterior: {self.sigma_posterior}"

    def reset_parameters(self):
        # Initialization from https://arxiv.org/abs/1810.01279
        stdv = 1. / math.sqrt(self.W_mu.size(1))

        self.W_mu.data.uniform_(-stdv, stdv)
        self.W_sigma.data.fill_(self.log_sigma_prior)
        
        if self.bias :
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_sigma.data.fill_(self.log_sigma_prior)

    def get_out_shape(self, img_shape: Tuple):
        h_out = math.floor((self.img_Shape - self.kernel[0] + 2*self.padding) / self.stride + 1)
        w_out = math.floor((self.in_feats[3] - self.kernel[1] + 2*self.padding) / self.stride + 1)
        return (self.in_feats[0], self.n_features, h_out, w_out)

    def forward(self, input: torch.Tensor, sample=True):
        W_sigma = torch.log1p(torch.exp(self.W_sigma))
        W_noise = torch.rand_like(W_sigma)
        
        # TODO: test without sample
        if not sample:  # rarely used so if branching is avoided for performance
            W_sigma = torch.zeros(self.W_mu.size()).to(self.device)
            W_noise = torch.zeros(self.W_mu.size()).to(self.device)

        # reparametrization trick
        weights = self.W_mu + W_sigma * W_noise
        
        if self.bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_sigma))  
            bias_noise = torch.rand_like(bias_sigma)
            # reparametrization trick
            bias = self.bias_mu + bias_noise * bias_sigma
        else:
            bias = None

        return F.conv2d(
            input=input.to(self.device),
            weight=weights,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
