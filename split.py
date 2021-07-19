import torch


class Index_add_(torch.autograd.Function):
    @staticmethod
    def symbolic(g,  index, source,result):
        result = g.op('Index_add_', index, source)
        return result
        #return g.op('Index_add_', dim, index, source, outputs=1)

    @staticmethod
    def forward(ctx,index, source,result):   
        #result = g.op("ScatterND", self, index, values)
        result.index_add_(0,index,source)
        return result
        #return  index_add_(dim,index,source)

def test_onnx_export():
    class MyModule(torch.nn.Module):
        def forward(self,  index, source,result):
            return Index_add_().apply(index, source,result)
           # return input[2].Index_add_(0, input[0], input[1])

    A= torch.ones(5, 3)
    # dim = torch.zeros(1)
    # print(dim)
    B = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9],[100, 110, 120],[1000,1100,1200]], dtype=torch.float)
    index = torch.tensor([0, 4, 2,2,4])
    input = (index,B,A)
    torch.onnx.export(
                   MyModule(),
                   input,
                   "custom_op.onnx")

    print('finish convert onnx')

test_onnx_export()

# class Split(torch.autograd.Function):
#     @staticmethod
#     def symbolic(g, input):
#         return g.op('Split', input, outputs=2)

#     @staticmethod
#     def forward(ctx, input):
#         return input[0], input[1]

# def test_onnx_export():
#     class MyModule(torch.nn.Module):
#         def forward(self, input):
#             return Split().apply(input)

#     model_string = torch.onnx.export_to_pretty_string(
#                    MyModule(),
#                    (torch.tensor([0, 1])),
#                    "/tmp/custom_op.onnx")
#     print(model_string)

# test_onnx_export()

print('pass!!!!!!!!!!')