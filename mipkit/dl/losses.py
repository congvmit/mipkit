"""
 The MIT License (MIT)
 Copyright (c) 2021 Cong Vo
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 Provided license texts might have their own copyrights and restrictions
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
"""

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
