import torch
import onnx
x = torch.ones(4,5,6,7,8)
ind1 = torch.tensor([0, 1])
ind2 = torch.tensor([[3], [2]])
ind3 = torch.tensor([[2, 2], [4, 5]])

print(x[2:4, ind1, None, ind2, ind3, :])