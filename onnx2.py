import numpy as np
import os
import sys
from fractions import gcd
from numbers import Number

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from data import ArgoDataset, collate_fn
from utils import gpu, to_long,  Optimizer, StepLR

from layers import Conv1d, Res1d, Linear, LinearRes, Null
from numpy import float64, ndarray
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)
model_name = os.path.basename(file_path).split(".")[0]

### config ###
config = dict()
"""Train"""
# config["display_iters"] = 205942
# config["val_iters"] = 205942 * 2
config["display_iters"] = 1
config["val_iters"] = 1 * 2
config["save_freq"] = 1.0
config["epoch"] = 0
config["horovod"] = True
config["opt"] = "adam"
config["num_epochs"] = 36
config["lr"] = [1e-3, 1e-4]
config["lr_epochs"] = [32]
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])


if "save_dir" not in config:
    config["save_dir"] = os.path.join(
        root_path, "results", model_name
    )

if not os.path.isabs(config["save_dir"]):
    config["save_dir"] = os.path.join(root_path, "results", config["save_dir"])

config["batch_size"] = 32
config["val_batch_size"] = 32
config["workers"] = 0
config["val_workers"] = config["workers"]


"""Dataset"""
# Raw Dataset
config["train_split"] = os.path.join(
    root_path, "dataset/train/data"
)
config["val_split"] = os.path.join(root_path, "dataset/val/data")
config["test_split"] = os.path.join(root_path, "dataset/test_obs/data")

# Preprocessed Dataset
config["preprocess"] = True # whether use preprocess or not
config["preprocess_train"] = os.path.join(
    root_path, "dataset","preprocess", "train_crs_dist6_angle90.p"
)
config["preprocess_val"] = os.path.join(
    root_path,"dataset", "preprocess", "val_crs_dist6_angle90.p"
)
config['preprocess_test'] = os.path.join(root_path, "dataset",'preprocess', 'test.p')

"""Model"""
config["rot_aug"] = False
config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]
config["num_scales"] = 6
config["n_actor"] = 128
config["n_map"] = 128
config["actor2map_dist"] = 7.0
config["map2actor_dist"] = 6.0
config["actor2actor_dist"] = 100.0
config["pred_size"] = 30
config["pred_step"] = 1
config["num_preds"] = config["pred_size"] // config["pred_step"]
config["num_mods"] = 6
config["cls_coef"] = 1.0
config["reg_coef"] = 1.0
config["mgn"] = 0.2
config["cls_th"] = 2.0
config["cls_ignore"] = 0.2



def actor_gather(actors: List[Tensor]) -> List[Tensor]: # Tuple[Tensor, 
    batch_size = len(actors)
    print('batch_size:',batch_size)
    num_actors = [len(x) for x in actors]

    actors = [x.transpose(1, 2) for x in actors]
    #actors = torch.cat(actors, 0)

    # actor_idcs = []
    # count = 0
    # for i in range(batch_size):
    #     idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
    #     actor_idcs.append(idcs)
    #     count += num_actors[i]
    #return actors, actor_idcs
    return actors

    # def forward(self, x, y):
    #     for i in range(y.size(0)):
    #         x = x + i
    #     return x

class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """
    def __init__(self, config):
        super(ActorNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_in = 3
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = config["n_actor"]
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(n, n, norm=norm, ng=ng)

    # @torch.jit.script
    # def get_lat_res(self, index: int, t: Tensor) -> Tensor:
    #     for i, a_lat in enumerate(self.lateral):
    #         if i == index:
    #             return a_lat(t)

    def forward(self, actors: Tensor) -> Tensor:
        out = actors

        outputs = []
        for grp in self.groups:
            out = grp(out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            # 上下采样 FPN
            out = F.interpolate(out, scale_factor=2.0, mode="linear", align_corners=False)
            # out += self.lateral[i](outputs[i])
            for j, a_lat in enumerate(self.lateral):
                if i == j:
                    out += a_lat(outputs[i])

        out = self.output(out)[:, :, -1]
        return out

class DummyModule(torch.nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()
         #super(Net, self).__init__()
        self.config = config

        self.actor_net = ActorNet(config)
 
   # def forward(self, actors:Tensor) -> Tuple[Tensor, List[Tensor]]: 
        #Dict[Tensor] List[Tensor]:
    def forward(self, actors:List[Tensor], data_ctrs:List[Tensor]) -> Tensor:
    #def forward(self, actors): 如何用多个输入
        actors, actor_idcs = actor_gather(actors)
        actor_ctrs = data_ctrs
        actors = self.actor_net(actors)
        # print(actors)

        # construct map features
        graph:List[List[Tensor]] = data[2]
        nodes, node_idcs, node_ctrs = self.map_net(graph)

        # # actor-map fusion cycle 
        # nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)
        # nodes = self.m2m(nodes, graph)
        # actors = self.m2a(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        # actors = self.a2a(actors, actor_idcs, actor_ctrs)

        # # prediction
        # out = self.pred_net(actors, actor_idcs, actor_ctrs)
        # rot, orig = data[3], data[4]
        # # transform prediction to world coordinates
        # for i in range(len(out["reg"])):
        #     out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
        #         1, 1, 1, -1
        #     )
        return out

        # return actors



# Instantiation and scripting    原始为list，输入需要为tensor
# [15,20,3],[15,2],[1,747],[2,2],[2
dummy_input = [torch.randn(5,2,3, device='cuda')]

#print(dummy_input)
print('input len:',len(dummy_input))

model_scripted = torch.jit.script(DummyModule())

model_scripted = model_scripted.cuda()
model_scripted.eval()


# Check if the forward pass works:
# output = model_scripted(dummy_input)
# print(type(output))
print(type(dummy_input)) # list tensor
#print(type(*dummy_input)) # is tensor 
example_outputs = [model_scripted(dummy_input)]
print('output type:',type(example_outputs)) # is tensor 
# Export to onnx:
torch.onnx.export(model_scripted, 
                  dummy_input, 
                  'loop_actor.onnx', 
                  verbose=True,
                  input_names=['dummy_input'], 
                  #output_names=['example_outputs']
                  example_outputs=example_outputs
                 )      

print("Finish to onnx !!!!!!!!!!!!!!!!!!!")
# Load the model in onnx runtime:
import onnxruntime as rt

sess = rt.InferenceSession("loop.onnx")