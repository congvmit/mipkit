import os
import sys

import numpy as np
import torch
from torchvision.models.resnet import resnet50

import mipkit

# Init pretrained model
model = resnet50()

# Move to CUDA
_ = model.to("cuda")
_ = model.eval()

# Inference
x = torch.ones([1, 3, 224, 224]).to("cuda")
y = model(x)
print(y.shape)

# Initialize Hooking Wrapper to get outputs from a specific layers
hooker = mipkit.ForwardHook(model, output_layers=["layer1.0.conv1", "flatten1", "flatten", "fc"])
y = hooker(x)
print(y.shape)  # The output of model

# Get outputs from target layers
print(hooker.layer_outputs["layer1.0.conv1"].shape)
print(hooker.layer_outputs["fc"].shape)
