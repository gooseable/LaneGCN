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
os.environ["CUDA_VISABLE_DEVICES"]='0'

import pickle
import sys
from importlib import import_module

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# from lanegcn_actor import graph_gather
from data import ArgoTestDataset
from utils import Logger, load_pretrain, gpu, to_long, half

import vizdata
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import numpy as np
import onnxruntime

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


# define parser
parser = argparse.ArgumentParser(description="Argoverse Motion Forecasting in Pytorch")
parser.add_argument(
    "-m", "--model", default="lanegcn_actor", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true", default=True)
parser.add_argument(
    "--split", type=str, default="test", help='data split, "val" or "test"'
)
parser.add_argument(
    "--weight", default="36.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

def to_numpy(tensor):
        # return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        return tensor.detach().cpu().numpy()

# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)

def graph_gather(graphs):
    print('input graph gather:')
    print(len(graphs)) # 1
    print(type(graphs)) # <class 'list'>
    #print(graphs)
    # for key in graphs.keys():
    #     print(type(graphs[key]))

    batch_size = len(graphs)
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = []
    # graph["idcs"] = node_idcs
    graph.append(node_idcs)
    # graph["ctrs"] = [x["ctrs"] for x in graphs]
    graph.append([x["ctrs"] for x in graphs])

    for key in ["feats", "turn", "control", "intersect"]:
        # graph[key] = torch.cat([x[key] for x in graphs], 0)
        graph.append([torch.cat([x[key] for x in graphs], 0)])

    for k1 in ["pre", "suc"]:
        # graph[k1] = []
        graph.append([])
        for i in range(len(graphs[0]["pre"])):
            # graph[k1].append(dict())
            # graph[-1].append([])
            for k2 in ["u", "v"]:
                # graph[k1][i][k2] = torch.cat(
                #     [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                # )
                graph[-1].append(torch.cat(
                    [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                ))

    for k1 in ["left", "right"]:
        # graph[k1] = dict()
        graph.append([])
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [
                # x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                x if x.dim() > 0 else graph[get_graph_index("pre")][0].new_empty().resize_(0)
                for x in temp
            ]
            # graph[k1][k2] = torch.cat(temp)
            graph[-1].append(torch.cat(temp))
    
    # print(graphs)
    return graph

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
    ########################### 0621
    # weights = '360.pt'
    # net.load_state_dict(torch.load(weights, map_location=lambda storage, loc: storage))

    net.eval()
    ########################### 0621

    
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
            data['graph']= graph_gather(to_long(gpu(data['graph'])))
            data['rot'] = gpu(data['rot'])
            data['orig'] = gpu(data['orig'])
            new_data = (data['feats'], data['ctrs'], data['rot'], data['orig'],
                    data['graph'][0], data['graph'][1], data['graph'][2], data['graph'][3],
                    data['graph'][4], data['graph'][5], data['graph'][6], data['graph'][7],
                    data['graph'][8], data['graph'][9])

            # for i in new_data:
            #     print(type(i))
            print('-------------------feats type--------------------')
            print(type(new_data[0]))# list
            ts = time.time()
            output = net(new_data)
            print('output:')
            print(np.shape(output))

            # results_output = [x[0:1].detach().cpu().numpy() for x in  output[1]] # (1, 1, 6, 30, 2)
            # results_output = np.squeeze(results_output)
            # print(np.shape(results_output))
            print('--------------------------------------------------------')
#################################################


        torch.onnx.export(net,(new_data,),"results/lanegcn_actor_11.onnx",export_params=True, opset_version=11,verbose=True)#,opset_version=11

        ############# 验证  ,input_names = ['feats','ctrs','graph','rot','orig'],output_names=['output']
# import onnxruntime
        ort_session = onnxruntime.InferenceSession("results/lanegcn_actor_11.onnx")
        print(len(ort_session.get_inputs()))  # 37个输入
        input1 = torch.cat(data['feats'],0)
        input2 = torch.cat(data['ctrs'],0)
        input4 = torch.cat(data['rot'],0)
        input5 = torch.cat(data['orig'],0)
        ort_inputs = {ort_session.get_inputs()[0].name: input1.cpu().numpy(),ort_session.get_inputs()[1].name: input2.cpu().numpy(),
        ort_session.get_inputs()[2].name: input4.cpu().numpy(),
        ort_session.get_inputs()[3].name: input5.cpu().numpy()}

        ort_outs = ort_session.run(None, ort_inputs)
        # print('ort_outs:')
        # print(np.shape(ort_outs))
        # results = ort_outs[1] # (15, 6, 30, 2)
        # ort_results = results[0]  # # ( 6, 30, 2)
        # print(np.shape(ort_results))
        print('------------------------finish run onnx --------------------------------')
        
        # for i in range(len(ort_results)):
        #     print(i)
        #     np.testing.assert_allclose(results_output[i], ort_results[i], rtol=1e-03, atol=1e-05)
        # # print('---------------------------pass!!-----------------------------')
        # # # # # compare ONNX Runtime and PyTorch results
        # # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

        # print("Exported model has been tested with ONNXRuntime, and the result looks good!")

        # ts1 = time.time()
        #print(f'\n{i + 1} run take {ts1 - ts}')



    #     net_results = [x[0:1].detach().cpu().numpy() for x in  output[1]] 
    #     for i, (argo_idx, pred_traj) in enumerate(zip(data["argo_id"], net_results)):
    #         preds[argo_idx] = pred_traj.squeeze()
    #         cities[argo_idx] = data["city"][i]
    #         gts[argo_idx] = data["gt_preds"][i][0] if "gt_preds" in data else None

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




if __name__ == "__main__":
    main()
