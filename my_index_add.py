import torch

# def my_index_add(index,A,B):
#     x[index] = x[index].add(t)

def my_index_add(index,A,B):
    # 先找出index中重复的数字，将重复数字对应的B的行数加起来，叠加到A上

    new_index=[]
    count = 0
    new_B = []
    for i in index:
        # A[i] = A[i].add(B[count])
        # count = count + 1
        if i not in new_index:
            print(B[count])
            #new_index.append(i)
            #new_B += B[count]
            #new_B.append(B[count])
            #print(new_B)
        # else:
        #     new_B.append(torch.add(B[count-1],B[count]))
        #     print(new_B)

        new_B =  torch.stack(new_B)
        count = count+1
        #print(new_B)
        #print(new_index)
    A[index] = A[new_index].add(new_B)
    
    # for i in index:
    #     new_B = B[count]
    #     #count = count + 1
    #     if i in new_index:
    #         new_B[count] = torch.add(B[count], B[count+1])
    #         #print(new_li)
    # # 新的index为去掉重复数字的index

#     x[index] = x[index].add(t)

    #return A#

# def my_index_add(index, A, B):
#     # return
#     count = 0
#     for i in index:
#         A[i].add(B[count])
#         # A[i]= A[i] + B[count]
#         count = count + 1

#     return A

# x = torch.ones(5, 3)

# t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9],[100, 110, 120]], dtype=torch.float)
# index = torch.tensor([0, 4, 2,2])

# # print(x[index])
# # print(x[index].add(t))

# #x[index] = x[index].add(t)
# my_index_add(index, x, t)
# #print(x)
# tensor([[  2.,   3.,   4.],
        # [  1.,   1.,   1.],
        # [101., 111., 121.],
        # [  1.,   1.,   1.],
        # [  5.,   6.,   7.]])

# ture:tensor([[  2.,   3.,   4.],
#         [  1.,   1.,   1.],
#         [108., 119., 130.], 
#         [  1.,   1.,   1.],
#         [  5.,   6.,   7.]])

A= torch.ones(5, 3)

B = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9],[100, 110, 120]], dtype=torch.float)
index = torch.tensor([0, 4, 2,2])
new_index=[]
count = 0
new_B = []
for i in index:
    # A[i] = A[i].add(B[count])
    # count = count + 1
    if i not in new_index:
        #print(B[count])
        new_index+=[i]
        new_B += [B[count]]
        #new_B.append(B[count])
        #print(new_B)
    else:
        new_B[-1]=torch.add(B[count-1],B[count])
        #new_B.pop(-2)
        #print(new_B)
    count = count + 1
#print(new_index)
    #     new_B =  torch.stack(new_B)
    #     count = count+1
print(new_B)
print(new_index)
new_index = torch.stack(new_index)

new_B = torch.stack(new_B)
print(new_B)
A[new_index] = A[new_index].add(new_B)
print(A)
    # [tensor([1., 2., 3.]), tensor([4., 5., 6.]), tensor([107., 118., 129.])]  去掉 7 8 9
# tensor_list: [tensor([1., 2., 3.]), tensor([4., 5., 6.])] 

def my_index_add(index, A, B):
    new_index=[]
    count = 0
    new_B = []
    for i in index:
        if i not in new_index:
            #print(B[count])
            new_index+=[i]
            new_B += [B[count]]
            #new_B.append(B[count])
            #print(new_B)
        else:
            new_B[-1]=torch.add(B[count-1],B[count])
            #new_B.pop(-2)
            #print(new_B)
        count = count + 1
    new_index = torch.stack(new_index)
    new_B = torch.stack(new_B)
    A[new_index] = A[new_index].add(new_B)





count = 0
List=[1,2,2,2,2,3,3,3,4,4,4,4]
a = {}
for i in List:
  if List.count(i)>1:
      i#
    a[i] = List.count(i)

for l in range(len(index)):
    for i in index:
        c=[]
        for j in index:
            if j ==i:
                c.append(item[0])
        b.append(c)
    print(b)