import torch 
A= torch.ones(5, 3)

B = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9],[100, 110, 120],[1000,1100,1200],[10000,11000,12000]], dtype=torch.float)
index = torch.tensor([0, 4, 2,2,4,4])

###########    index_put  #############
# indices_list = sym_help._unpack_list(indices_list_value)
indices_list = index
index = indices_list[0]

if len(indices_list) > 1:
    for ind in indices_list[1:]:
        index = index.add(ind)
        # index = add(g, index, ind)
    broadcast_index_shape = index.shape
    #broadcast_index_shape = g.op("Shape", index)
    indices_list = [ind.expand(ind,broadcast_index_shape).unsqueeze(-1) for ind in indices_list]
    # indices_list = [
    #     g.op("Unsqueeze", expand(g, ind, broadcast_index_shape, None), axes_i=[-1]) for ind in indices_list
    # ]
    #index = g.op("Concat", *indices_list, axis_i=-1)
    index = torch.cat(indices_list,-1)
else:
    # broadcast_index_shape = g.op("Shape", index)
    broadcast_index_shape = index.shape
    #index = g.op("Unsqueeze", index, axes_i=[-1])
    index = index.unsqueeze(-1)
sub_data_shape= A.shape[len(indices_list):]
# sub_data_shape = sym_help._slice_helper(
#     g, g.op("Shape", self), axes=[0], starts=[len(indices_list)], ends=[maxsize])

values_shape = torch.cat((broadcast_index_shape,sub_data_shape),0)
# values_shape = g.op("Concat", broadcast_index_shape, sub_data_shape, axis_i=0)
# values = g.op("Reshape", values, values_shape)
values = values.reshape(values_shape)
# dtype = self.type().scalarType() #标量类型
# dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
# dtype = sym_help.scalar_type_to_pytorch_type[dtype]

# zeros = g.op("ConstantOfShape", g.op("Shape", self), value_t=torch.tensor([0], dtype=dtype)) #产生给定value shape的tensor
shape_A = A.shape
zeros = torch.zeros(shape_A)
#zeros = torch.tensor([0])
#result = g.op("ScatterND", zeros, index, values)
result = A.scatter(zeros, index, values)
#result = add(g, self, result)
result = A.add(result)
