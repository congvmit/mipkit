# Copyright (c) 2021 congvm
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
from torch.nn.modules.loss import _Loss
# torch.set_printoptions(precision=4, sci_mode=False, linewidth=150)


class FocalBCE(_Loss):
    # https://www.kaggle.com/thedrcat/focal-multilabel-loss-in-pytorch-explained
    def __init__(self, gamma=2, *args, **kwargs):
        super(FocalBCE, self).__init__(*args, **kwargs)
        self.gamma = gamma

    def forward(self, input, target):
        num_classes = target.size(1)
        l = input.reshape(-1)
        t = target.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1-p)
        logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
        loss = logp*((1-p)**self.gamma)
        loss = num_classes*loss

        if self.reduction == 'mean' or self.reduction is None:
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class CCCLoss(_Loss):
    def __init__(self, *args, **kwargs):
        super(CCCLoss, self).__init__()

    def forward(self, input, target, seq_lens=None):
        if seq_lens is not None:
            mask = torch.ones_like(target, device=target.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
        else:
            mask = torch.ones_like(target, device=target.device)

        target_mean = torch.sum(
            target * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        input_mean = torch.sum(
            input * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)

        target_var = torch.sum(mask * (target - target_mean) ** 2, dim=1, keepdim=True) / torch.sum(mask, dim=1,
                                                                                                    keepdim=True)
        input_var = torch.sum(mask * (input - input_mean) ** 2, dim=1, keepdim=True) / torch.sum(mask, dim=1,
                                                                                                 keepdim=True)

        cov = torch.sum(mask * (target - target_mean) * (input - input_mean), dim=1, keepdim=True) \
            / torch.sum(mask, dim=1, keepdim=True)

        ccc = torch.mean(2.0 * cov / (target_var + input_var +
                                      (target_mean - input_mean) ** 2), dim=0)
        ccc = ccc.squeeze(0)
        ccc_loss = 1.0 - ccc

        return ccc_loss
