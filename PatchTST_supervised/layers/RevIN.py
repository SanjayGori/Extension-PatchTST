# code from https://github.com/ts-kim/RevIN, with minor modifications

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, length: int = None):
        if mode == 'norm':
            self._get_statistics(x)
            self.saved_mean = self.mean
            self.saved_stdev = self.stdev
            if self.subtract_last:
                self.saved_last = self.last
            x = self._normalize(x)
        elif mode == 'denorm':
            mean = self.saved_mean[..., :length] if not self.subtract_last else None
            stdev = self.saved_stdev[..., :length]
            last = self.saved_last[..., :length] if self.subtract_last else None
            x = self._denormalize(x, mean, stdev, last)
        else:
            raise NotImplementedError
        return x
    
    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:, -1:, :].clone().detach()
        else:
            self.mean = x.mean(dim=dim2reduce, keepdim=True).detach()
        self.stdev = x.std(dim=dim2reduce, keepdim=True).detach() + self.eps


    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, mean, stdev, last=None):
        # x: [B, C, pred_len]
        # mean/stdev: [B, C, history_len]
        pred_len = x.shape[-1]
        mean = mean[..., -pred_len:]
        stdev = stdev[..., -pred_len:]
        if last is not None:
            last = last[..., -pred_len:]
            
        x = x * stdev
        if self.subtract_last:
            x = x + last
        else:
            x = x + mean
        return x


