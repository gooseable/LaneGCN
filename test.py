# ---------------------------------------------------------------------------
# Learning Lane Graph Representations for Motion Forecasting
#
# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Ming Liang, Yun Chen
# ---------------------------------------------------------------------------
import torch, onnx, collections
import argparse
import os
import time
os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import sys
from importlib import import_module

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from lanegcn import graph_gather
from data import ArgoTestDataset
from utils import Logger, load_pretrain, gpu, to_long, half,to_int32

import vizdata
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import numpy as np

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


# define parser
parser = argparse.ArgumentParser(description="Argoverse Motion Forecasting in Pytorch")
parser.add_argument(
    "-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true", default=True)
parser.add_argument(
    "--split", type=str, default="test", help='data split, "val" or "test"'
)
parser.add_argument(
    "--weight", default="36.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
def main():
    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, _, collate_fn, net, loss, post_process, opt = model.get_model()

    # load pretrain model
    ckpt_path = args.weight
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])
    net.eval()

    ############### 将ckpt模型转化为torchscript模型 ########################
    ##script_mod = torch.jit.load('36.pt')
    # net.to(torch.device('cpu'))
    #script_mod = torch.jit.script(net)

    # graph, params = torch._C._jit_pass_lower_graph(script_mod.forward.graph, script_mod._c)

    #script_mod = script_mod.half()
    #torch.jit.save(script_mod.state_dict(), '360.pt')
    torch.save(net.state_dict(), '360.pt')
    #model.state_dict()
    # print('net scripted')
    # script_mod = torch.jit.load('36.pt')
    #scp_mod = script_mod.cuda()
    #scp_mod.eval()
    # script_mod = script_mod.cpu()
    ############### 将ckpt模型转化为torchscript模型 


    # Data loader for evaluation
    dataset = ArgoTestDataset(args.split, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
    )

    # begin inference
    preds = {}
    gts = {}
    cities = {}
    for ii, data in tqdm(enumerate(data_loader)):
        data = dict(data)
        with torch.no_grad():
            data['feats'] = gpu(data['feats'])
            data['ctrs'] = gpu(data['ctrs'])
            data['graph'] = graph_gather(to_long(gpu(data['graph'])))

            print(data['graph'][0][0].dtype) # torch.int64
            # for i in data['graph'] :
            #     print(i)
            #     #i =  torch.tensor(i,dtype=torch.float32)
            #     print('------------------------------')
            # #data['graph'] = torch.tensor(graph_gather(gpu(data['graph'])),dtype=torch.float32)

            # #print(data['graph'])  #结果多个tensor的组合
            data['rot'] = gpu(data['rot'])
            data['orig'] = gpu(data['orig'])
            new_data = [data['feats'], data['ctrs'], data['graph'], data['rot'], data['orig']]  #  tensor

            # # new_data= torch.tensor([item.detach().numpy() for item in new_data]).cuda()
            # # new_data shape: ([15,20,3],[15,2],[1,747],[2,2],[2])
            # print("---------------------------ccccccccccccccccccccc")
            print(new_data[0])
            #output = scp_mod(new_data)
           # torch.onnx.export( scp_mod,new_data, 'test.onnx', example_outputs=output)

            #print(data['rot'])   #feats:list 15 20 3   ctrs 15,2  graph  1,747 rot  2,2 orig 2
            feats1 = len(data['feats'][0])
            feats2 = len(data['feats'][0][0])
            feats3 = len(data['feats'][0][0][0])
            print("feats1 : " + str(feats1))
            print("feats2 : " + str(feats2))
            print("feats3 : " + str(feats3))
            # torch.onnx.export(net, (new_data,), '36.onnx', opset_version=11, verbose=True)
            for i in range(1):
                ts = time.time()
                output = net(new_data)
                #output = scp_mod(new_data)
                #torch.onnx.export(scp_mod,new_data, 'test.onnx', example_outputs=output)

                ts1 = time.time()
                print(f'\n{i + 1} run take {ts1 - ts}')


            # output = scp_mod(new_data)
            # print(f'second run take {time.time() - ts1}')
            # scp_mod.save('37.pt')

            # output = net(new_data)
            # output = script_mod(new_data)
            #print('output--------------')
            # for i in output:
            #     print(output)
            results = [x[0:1].detach().cpu().numpy() for x in output["reg"]]
        for i, (argo_idx, pred_traj) in enumerate(zip(data["argo_id"], results)):
            preds[argo_idx] = pred_traj.squeeze()
            cities[argo_idx] = data["city"][i]
            gts[argo_idx] = data["gt_preds"][i][0] if "gt_preds" in data else None

    afl = ArgoverseForecastingLoader("/home/chl/code/lg/LaneGCN-master/dataset/test_obs/tmp/")
    df = afl.get("/home/chl/code/lg/LaneGCN-master/dataset/test_obs/tmp/1.csv").seq_df
    frames = df.groupby("OBJECT_TYPE")
    input_ = np.zeros((20,2), dtype=float)
    gt_ = np.zeros((30,2), dtype=float)
    for group_name, group_data in frames:
        if group_name == 'AGENT':
            input_[:, 0] = group_data['X'][:20]
            input_[:, 1] = group_data['Y'][:20]
            gt_[:, 0] = group_data['X'][20: ]
            gt_[:, 1] = group_data['Y'][20: ]
    #input: (20, 2)
    #preds[1]: (6, 30, 2)
    #gt: none
    #city: str
   vizdata.viz_predictions(input_, preds[1], gt_, cities[1])
    # save for further visualization
    res = dict(
        preds = preds,
        gts = gts,
        cities = cities,
    )
    # torch.save(res,f"{config['save_dir']}/results.pkl")
    
    # evaluate or submit
    if args.split == "val":
        # for val set: compute metric
        from argoverse.evaluation.eval_forecasting import (
            compute_forecasting_metrics,
        )
        # Max #guesses (K): 6
        _ = compute_forecasting_metrics(preds, gts, cities, 6, 30, 2)
        # Max #guesses (K): 1
        _ = compute_forecasting_metrics(preds, gts, cities, 1, 30, 2)
    else:
        # for test set: save as h5 for submission in evaluation server
        from argoverse.evaluation.competition_util import generate_forecasting_h5
        generate_forecasting_h5(preds, f"{config['save_dir']}/submit.h5")  # this might take awhile
    # import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    main()
