import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False, activation=None, droprate=0.):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out)
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = None
        if droprate > 0.:
            self.dropout = nn.Dropout(p=droprate)

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in
        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)
        bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
