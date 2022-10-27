#!/usr/bin/env python3
# coding=utf-8

import torch
from data.field.mini_torchtext.field import RawField


class BasicField(RawField):
    def process(self, example, device=None):
        tensor = torch.tensor(example, dtype=torch.long, device=device)
        # tensor = example.clone().detach().to(device)

        return tensor

# it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).