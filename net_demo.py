import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime

def my_index_add(index, A, B):
    # return
    count = 0
    for i in index:
        A[i] = A[i].add(B[count])
        count = count + 1
    return A
# def my_index_add(index, A, B):
#     # return
#     count = 0
#     for i in index:
#         # print(i)
#         A[i] = A[i].add(B[count])
#         # print(A[i])
#         count = count + 1
#     return A

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        

    def forward(self,  index,  A, B):
        # indexA = range(A.shape[0])
        # uniq = torch.unique(index, sorted=True).tolist()
        # # paddingindex = torch.LongTensor(list(set(indexA)-set(uniq)))
        # # index = torch.cat((index, paddingindex))
        # # paddingval = B[0].clone()
        # # paddingval = paddingval.expand(paddingindex.shape[0], B.shape[1])
        # # B = torch.cat((B, paddingval))
        

        # indexA = range(A.shape[0])
        # # uniq = torch.unique(index, sorted=True)
        # uniq = torch.unique(index, sorted=True).tolist()
        # paddingindex = torch.LongTensor(list(set(indexA)-set(uniq)))
        # index = torch.cat((index, paddingindex))
        # paddingval = torch.zeros(1, B.shape[1])
        # paddingval = paddingval.expand(paddingindex.shape[0], B.shape[1])
        # B = torch.cat((B, paddingval))

        # C = A.clone()
        # C = C.index_put((index,), B)
        # A = A.index_put_((index,), B, accumulate=True)
        # out = A.sub(C)
        # return out
        #A[uniq] = A[uniq].sub(C[uniq])
        # count = 0
        # out = A.clone()
        # for i in index:
        #     out[i] = out[i].add(B[count])
        #     count = count + 1
        # return out
        count = 0
        #out = A.clone()
        for i in index:
            A[i] = A[i].add(B[count])
            count = count + 1
        return A
        

        # C = A.clone()
        # C = C.index_put((index,),B)
        # uniq_index = torch.unique(index, sorted=True) #  0 2 4
        # # 找出没有进行运算的行
       
        # A = A.index_put_((index,),B,accumulate=True)

        # # for i in index:
        # #     A[i] = A[i].sub(C[i])
        return A
    # def forward(self,  index,  A, B):
    #     count = 0
    #     for i in index:
    #         A[i].add_(B[count])
    #        # A[i].index_put_((count,), B[count], True)
    #         # A[i]= A[i] + B[count]
    #         count = count + 1
    #     #A = A.index_put((index,),B,accumulate=True)
    #     return A
# 排除用add_,参数数量不对
# 使用add结果不对

trained_model = Net()
A= torch.ones(5, 3)
# dim = torch.zeros(1)
# print(dim)
# B = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
# index = torch.tensor([0, 4, 2])
B = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9],[100, 110, 120],[1000,1100,1200],[10000,11000,12000]], dtype=torch.float)
index = torch.tensor([0, 4, 2,2,4,4])

input = (index,A,B)



# torch.onnx.export(
#                 trained_model,
#                 input,
#                 "net_demo.onnx",opset_version=11)

print('finish convert onnx')
ort_session = onnxruntime.InferenceSession("net_demo.onnx")
print(len(ort_session.get_inputs()))  # 3
print('onnx input shape:',ort_session.get_inputs()[0].shape) #  [5]
# print('onnx input shape:',ort_session.get_inputs()[1].shape) #  [5,3]
# print('onnx input shape:',ort_session.get_inputs()[2].shape) #  [5,3]

ort_inputs = {ort_session.get_inputs()[0].name:index.cpu().numpy(),ort_session.get_inputs()[1].name:A.cpu().numpy(),ort_session.get_inputs()[2].name: B.cpu().numpy()}

ort_outs = ort_session.run(None, ort_inputs)
print('ort_outs:')
print(ort_outs)
# results = ort_outs[1]
# print('--------------------------------------------------------')
# print('pass!!!!!!!!!!')

# net_out = trained_model(index,A,B)
# print('xxxxxxxxxx net out xxxxxxxxxx')
# print(net_out)
'''
tensor([[2.0000e+00, 3.0000e+00, 4.0000e+00],
        [1.0000e+00, 1.0000e+00, 1.0000e+00],
        [1.0800e+02, 1.1900e+02, 1.3000e+02],
        [1.0000e+00, 1.0000e+00, 1.0000e+00],
        [1.0050e+03, 1.1060e+03, 1.2070e+03]])

[1,2,3]+[1,2,3] +[1,1,1] = [3,5,7]

[100,110,120] + [100,110,120] + [15,17,19]  = [215,237,259]     

[1000,1100,1200] + [1000,1100,1200]      [9  11  13]                      [2009,2211,2413]

[array([[3.000e+00, 5.000e+00, 7.000e+00],
       [1.000e+00, 1.000e+00, 1.000e+00],
       [2.150e+02, 2.370e+02, 2.590e+02],
       [1.000e+00, 1.000e+00, 1.000e+00],
       [2.009e+03, 2.211e+03, 2.413e+03]],
'''
# def my_index_add(index, A, B):
#     indexA = range(A.shape[0])
#     uniq = torch.unique(index, sorted=True).tolist()
#     paddingindex = gpu(torch.LongTensor(list(set(indexA)-set(uniq))))
#     index = torch.cat((index, paddingindex))
#     paddingval = B[0].clone()
#     paddingval = paddingval.expand(paddingindex.shape[0], B.shape[1])
#     B = torch.cat((B, paddingval))
    
#     C = A.clone()
#     C = C.index_put((index,), B)
#     A = A.index_put_((index,), B, accumulate=True)
#     out = A.sub(C)
#     return out