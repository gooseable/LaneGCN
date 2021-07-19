
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime

def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        # for k, v in data.items():
        #     data[k] = gpu(v)
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data

def slicetensoradd(index, A, B):
    indexid = gpu(torch.arange(0, index.shape[0], dtype=torch.int64))
    C = gpu(torch.cat((index.unsqueeze(0).t(), indexid.unsqueeze(0).t()), dim=1))
    D = gpu(torch.unique(C, dim=0))
    uniqindex,count = torch.unique(index, return_counts=True)
    indexcnt = torch.cat((uniqindex.unsqueeze(0).t(), count.unsqueeze(0).t()), dim=1)
    # E = torch.tensor([],device='cuda:0', dtype=torch.int64)
    E = gpu(torch.zeros((D[-1][0] + 1, 2), dtype=torch.int64))
    # E = gpu(torch.zeros((D.shape[0], 2), dtype=torch.int64))
    # print(E)
    E[indexcnt[:, 0]] = E[indexcnt[:, 0]].add(indexcnt)
    indexcnt1 = D[E[D[:, 0]][:,1]==1]
    # indexcnt1 = torch.masked_select(D, E[D[:, 0]][:,1]==1)
    # indexcnt1 = torch.index_select(D, dim=0, index=(E[D[:, 0]][:,1]==1))

    # A[indexcnt1[:,0]] = A[indexcnt1[:,0]].add(B[indexcnt1[:,1]])# Warning: ONNX Preprocess - Removing mutation on block inputs. This changes graph semantics.
    newindex = indexcnt1[:,0]
    newB = B[indexcnt1[:,1]]
    A[newindex]
    A = A.index_put((newindex,), newB, accumulate=True)
    # return A
    # A[indexcnt1[:,0]]
    # A = A.index_put((index,), B, accumulate=True)
    indexcnt2 = D[E[D[:, 0]][:,1]>1]
    for i in indexcnt2: #RuntimeWarning: Iterating over a tensor might cause the trace to be incorrect
        # A[i[0]] = A[i[0]].add(B[i[1]])
        A = A.index_put((i[0],), B[i[1]], accumulate=True)
    return A





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
    def forward(self,  index,  A, B):
        out = slicetensoradd(index,A,B)
        # count = 0
        # out = A.clone()
        # for i in index:
        #     out[i] = out[i].add(B[count])
        #     count = count + 1
        return out

trained_model = Net()
A= torch.ones(5, 3)
B = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9],[100, 110, 120],[1000,1100,1200],[10000,11000,12000]], dtype=torch.float)
index = torch.tensor([0, 4, 2,2,4,4])
input = (index,A,B)
torch.onnx.export(
                trained_model,
                input,
                "net_demo.onnx",operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,opset_version=11)

print('finish convert onnx')
ort_session = onnxruntime.InferenceSession("net_demo.onnx")
print(len(ort_session.get_inputs()))  # 3

ort_inputs = {ort_session.get_inputs()[0].name:index.cpu().numpy(),ort_session.get_inputs()[1].name:A.cpu().numpy(),ort_session.get_inputs()[2].name: B.cpu().numpy()}

ort_outs = ort_session.run(None, ort_inputs)
print('xxxxxxxxxx onnxruntime output xxxxxxxxxx')
print(ort_outs)

net_out = trained_model(index,A,B)
print('xxxxxxxxxx net output xxxxxxxxxx')
print(net_out)
