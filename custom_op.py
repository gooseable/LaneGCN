


class relu5_func(Function):
    @staticmethod
    def forward(ctx, input):
        return relu5_cuda.relu5(input)
    @staticmethod
    def symbolic(g, *inputs):
        return g.op("Relu5", inputs[0], myattr_f=1.0) 
        # 这里第一个参数"Relu5"表示ONNX输出命名
        # myattr可以随便取，表示一个属性名，_f表示是一个float类型
relu5 = relu5_func.apply



import torch
import torch.nn as nn
import relu5_cuda
import onnx
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import netron

class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(-1)
        x = relu5(x)
        return x

net = TinyNet().cuda()
ipt = torch.ones(2,3,12,12).cuda()
torch.onnx.export(net, (ipt,), 'tinynet.onnx')
print(onnx.load('tinynet.onnx'))
netron.start('tinynet.onnx')
