import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import onnxruntime

def my_index_add(index, A, B):
    count = 0
    out = A.clone()
    for i in index:
        out[i] = out[i].add(B[count])
        count = count + 1
    return out


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
        
#     def forward(self,  index,  A, B):
#         out = A.index_add(0,index,B)
#         # count = 0
#         # out = A.clone()
#         # for i in index:
#         #     out[i] = out[i].add(B[count])
#         #     count = count + 1
#         return out

# trained_model = Net()
# A= torch.ones(5, 3)
B = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9],[100, 110, 120],[1000,1100,1200],[10000,11000,12000]], dtype=torch.float)

print(B[0])

# index = torch.tensor([0, 4, 2,2,4,4])
# input = (index,A,B)
# torch.onnx.export(
#                 trained_model,
#                 input,
#                 "net_demo.onnx",operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,opset_version=11)

# print('finish convert onnx')
# ort_session = onnxruntime.InferenceSession("net_demo.onnx")
# print(len(ort_session.get_inputs()))  # 3

# ort_inputs = {ort_session.get_inputs()[0].name:index.cpu().numpy(),ort_session.get_inputs()[1].name:A.cpu().numpy(),ort_session.get_inputs()[2].name: B.cpu().numpy()}

# ort_outs = ort_session.run(None, ort_inputs)
# print('xxxxxxxxxx onnxruntime output xxxxxxxxxx')
# print(ort_outs)

# net_out = trained_model(index,A,B)
# print('xxxxxxxxxx net output xxxxxxxxxx')
# print(net_out)
