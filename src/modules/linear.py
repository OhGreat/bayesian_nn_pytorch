import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import Normal
from collections import namedtuple


class BLinear(nn.Module):

    def __init__(
        self,
        in_feats,
        out_feats,
        mu_prior: float = 0,
        sigma_prior: float = 0.1,
        mu_posterior: Tuple[float, float] = (0, 0.1),
        sigma_posterior: Tuple[float, float] = (0, 0.1),
        bias: bool = True,
        device: str = None,
    ) -> None:
        """
        TODO: Describe stuff...
        """
        super(BLinear, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.log_sigma_prior = math.log(sigma_prior)
        self.mu_posterior = mu_posterior
        self.sigma_posterior = sigma_posterior
        self.bias = bias
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else: self.device = device 

        # each weight is represented by a tuple of mu and sigma parameters of a normal distribution
        # the actual weights are sampled from the normal distribution in the forward pass
        self.W_mu = nn.Parameter(torch.empty((out_feats, in_feats), device=self.device))
        self.W_sigma = nn.Parameter(torch.empty((out_feats, in_feats), device=self.device))

        if self.bias:
            self.bias_mu = nn.Parameter(torch.empty(out_feats, device=self.device))
            self.bias_sigma = nn.Parameter(torch.empty(out_feats, device=self.device))

        self.reset_parameters()  # initialize weights

    def to_str(self):
        return f"in features: {self.in_feats}, out features: {self.out_feats}\n\
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

        # TODO: fix method with posterior and add param to use it
        # self.W_mu.data.normal_(*self.mu_posterior)
        # self.W_sigma.data.normal_(*self.sigma_posterior)

        # if self.use_bias:
        #     self.bias_mu.data.normal_(*self.mu_posterior)
        #     self.bias_sigma.data.normal_(*self.sigma_posterior)

    def forward(self, input: torch.Tensor, sample=True):
        W_sigma = torch.log1p(torch.exp(self.W_sigma))
        # rand_like is faster than torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
        W_noise = torch.rand_like(W_sigma)
        
        if not sample:  # we usually use sample so we wat to avoid branching
            W_sigma = torch.zeros(self.W_mu.size()).to(self.device)
            W_noise = torch.zeros(self.W_mu.size()).to(self.device)

        # reparametrization trick!
        weights = self.W_mu + W_sigma * W_noise
        
        if self.bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_sigma))  
            bias_noise = torch.rand_like(bias_sigma)
            # reparametrization trick!
            bias = self.bias_mu + bias_noise * bias_sigma
        else:
            bias = None
        
        return F.linear(input.to(self.device), weights, bias)
