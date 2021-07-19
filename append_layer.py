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

from layers import Conv1d, Linear, LinearRes, Null
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

def actor_gather(actors: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    batch_size = len(actors)
    print('batch_size:',batch_size)
    num_actors = [len(x) for x in actors]

    actors = [x.transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs

# from layers import Res1d
class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Res1d, self).__init__()
        #assert(norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace = True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
       
        self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
        self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        
        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                        nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                        nn.GroupNorm(gcd(ng, n_out), n_out))
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
       
        out = self.conv1(x)
        #return out
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out
        #return x

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
            group.append(Res1d(n_in, n_out[i]))
            

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i]))

            groups.append(nn.Sequential(*group))

            #print(groups)
            n_in = n_out[i]
        #groups = group
        self.groups = nn.ModuleList(groups)



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
            outputs+=[out]

        # print(outputs[2].shape) #List中包含3个tensor大小分别为： torch.Size([15, 32, 20]),torch.Size([15, 64, 10]) torch.Size([15, 128, 5])

        #print(outputs)
        # print('xxxxxxxxx -1 xxxxxxxx')
        # print(outputs[-1].shape) # torch.Size([15, 128, 5]),
        # out = self.lateral[-1](outputs[-1]) # 测试过不是loop，list取负数索引可以通过onnx
        # print('xxxxxxxxx out xxxxxxxx')
        # print(out.shape) #  Tensor   torch.Size([15, 128, 5])
        # for i in range(len(outputs) - 2, -1, -1):
        #     # 上下采样 FPN
        #     out = F.interpolate(out, scale_factor=2.0, mode="linear", align_corners=False)
        #     out += self.lateral[i](outputs[i])
            # for j, a_lat in enumerate(self.lateral):
            #     if i == j:
            #         out += a_lat(outputs[i])

        # out = self.output(out)[:, :, -1]
        return out

dummy_input=torch.randn(15,3,20, device='cuda')
model_scripted = torch.jit.script(ActorNet(config))

model_scripted = model_scripted.cuda()
# Check if the forward pass works:

output = model_scripted(dummy_input)
print('output type:')
print(type(output))

torch.onnx.export(model_scripted, 
                  dummy_input, 
                  'loop_actor.onnx', 
                  verbose=True,
                  input_names=['input_data'], 
                  example_outputs=output
                 )

print("Finish to onnx !!!!!!!!!!!!!!!!!!!")