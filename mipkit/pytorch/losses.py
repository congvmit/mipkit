# Copyright (c) 2021 congvm
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
# torch.set_printoptions(precision=4, sci_mode=False, linewidth=150)


class FocalBCE(nn.Module):
    def __init__(self, gamma=2):
        self.gamma = gamma

    def forward(self, logits, targets):
        num_classes = targets.size(1)
        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1-p)
        logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
        loss = logp*((1-p)**self.gamma)
        loss = num_classes*loss.mean()
        return loss