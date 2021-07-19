import onnx
#import torch
import numpy as np
import os
import sys
from fractions import gcd
from numbers import Number

import torch
from torch import Tensor, nn
from torch.nn import functional as F

A = torch.randn(4,5)

B= torch.randn(2,3)

C = []

C = [A] +[B]

print(C)