
import torch
A= torch.ones(5, 3)

B = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9],[100, 110, 120],[1000,1100,1200]], dtype=torch.float)
#B = B.contiguous().cuda(non_blocking=True)

index = torch.tensor([0, 4, 2,2,4])


# A.index_fill(0,index,B)
# print(A)








#index = index.contiguous().cuda(non_blocking=True)

#A = A.index_put((index,),B,accumulate=True)
#A[index] = A[index].add_(B)
# my_index_add(index, A, B)
# print(A)

# def my_index_add(index, A, B):
#     A = A.index_put((index,),B,accumulate=True)
#     return A


# def my_index_add(index, A, B):
#     # return
#     count = 0
#     for i in index:
#         # print(i)
#         A[i] = A[i].add(B[count])
#         # print(A[i])
#         count = count + 1
#     return A

def my_index_add(index, A, B):
    count = 0
    for i in index:
        # id = torch.tensor([count])
        print(i.shape)
        A[i].index_put_((i,), B[count], True)
        count  = count + 1
    # return A

A = my_index_add(index, A, B)
#A.index_add_(0,index, B)
print(A)