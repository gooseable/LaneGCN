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

from lanegcn import graph_gather
from data import ArgoTestDataset
from utils import Logger, load_pretrain, gpu, to_long, half,to_int32

import vizdata
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import numpy as np
import onnxruntime

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

def to_numpy(tensor):
        # return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        return tensor.detach().cpu().numpy()

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
    ########################### 0621
    # weights = '360.pt'
    # net.load_state_dict(torch.load(weights, map_location=lambda storage, loc: storage))

    net.eval()
    ########################### 0621

    ############### 将ckpt模型转化为torchscript模型 ########################
    ##script_mod = torch.jit.load('36.pt')
    # net.to(torch.device('cpu'))
    #script_mod = torch.jit.script(net)
    # graph, params = torch._C._jit_pass_lower_graph(script_mod.forward.graph, script_mod._c)

    #script_mod = script_mod.half()
    #torch.jit.save(script_mod, '36.pt')
    # print('net scripted')
    # script_mod = torch.jit.load('36.pt')
    # scp_mod = script_mod.cuda()
    # scp_mod.eval()
    # script_mod = script_mod.cpu()
    ############### 将ckpt模型转化为torchscript模型 
    # pt_weights = '36.pt'
    # pt_model = net()
    # pt_model.load_state_dict(torch.load(pt_weights, map_location=lambda storage, loc:storage))

    # pt_model.eval()  # Data loader for evaluation


    
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
        #print(data.keys()) # dict_keys(['orig', 'theta', 'rot', 'feats', 'ctrs', 'argo_id', 'city', 'graph_ctrs', 'graph_num_nodes', 'graph_feats', 'graph_turn', 'graph_control', 'graph_intersect', 'graph_pre', 'graph_suc', 'graph_lane_idcs', 'graph_pre_pairs', 'graph_suc_pairs', 'graph_left_pairs', 'graph_right_pairs', 'graph_left', 'graph_right'])

        with torch.no_grad():
            data['feats'] = gpu(data['feats'])
            data['ctrs'] = gpu(data['ctrs'])
            data['graph']= graph_gather(to_long(gpu(data['graph'])))

            for i in range(10):
                print(len(data['graph'][i]))
                leng = len(data['graph'][i])
                for ii in range(leng):
                    print(data['graph'][i][ii].shape)
            #data= graph_gather(to_long(gpu(data)))
            #print('after graph gather:-------')
            #print(type(data['graph'])) # <class 'list'>

            # input_graph = (data['graph_ctrs'],data['graph_num_nodes'],data['graph_feats'],data['graph_turn'],data['graph_control'],data['graph_intersect'],data['graph_pre'],data['graph_suc'],data['graph_lane_idcs'],data['graph_pre_pairs'],data['graph_suc_pairs'],data['graph_left_pairs'],data['graph_right_pairs'],data['graph_left'],data['graph_right'])
            

            # print('-----------------------graph')
            # print(len(data['graph']))  # 10
            # print(data['graph'])


            #print(data['graph'][0][0].dtype) # torch.int64

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
        
            

#################################################


            torch.onnx.export(net,(new_data,),"lanegcn_37.onnx",export_params=True,opset_version=11)

        ############# Net从MapNet返回
# import onnxruntime
        ort_session = onnxruntime.InferenceSession("lanegcn_37.onnx")
        print(len(ort_session.get_inputs()))  # 30个输入


        input1 = torch.cat(data['feats'],0)
        input2 = torch.cat(data['ctrs'],0)
        input3_2 =  torch.cat(data['graph'][1],0) # ctrs
        input3_3 =  torch.cat(data['graph'][2],0) # feats
        input3_4 =  torch.cat(data['graph'][3],0) # turn
        input3_5 =  torch.cat(data['graph'][4],0) # control
        input3_6 =  torch.cat(data['graph'][5],0) # intersect
        input3_7 =  data['graph'][6][0]
        input3_8 =  data['graph'][6][1]
        input3_9 =  data['graph'][6][2]
        input3_10 =  data['graph'][6][3]
        input3_11 =  data['graph'][6][4]
        input3_12 =  data['graph'][6][5]
        input3_13 =  data['graph'][6][6]
        input3_14 =  data['graph'][6][7]
        input3_15 =  data['graph'][6][8]
        input3_16 =  data['graph'][6][9]
        input3_17 =  data['graph'][6][10]
        input3_18 =  data['graph'][6][11]

        input3_19 =  data['graph'][7][0]
        input3_20 =  data['graph'][7][1]
        input3_21 =  data['graph'][7][2]
        input3_22 =  data['graph'][7][3]
        input3_23 =  data['graph'][7][4]
        input3_24 =  data['graph'][7][5]        
        input3_25 =  data['graph'][7][6]
        input3_26 =  data['graph'][7][7]
        input3_27 =  data['graph'][7][8]
        input3_28 =  data['graph'][7][9]
        input3_29 =  data['graph'][7][10]
        input3_30 =  data['graph'][7][11]

        input3_31 =  data['graph'][8][0]
        input3_32 =  data['graph'][8][1]     

        input3_33 =  data['graph'][9][0]
        input3_34 =  data['graph'][9][1]


        ort_inputs = {ort_session.get_inputs()[0].name: input3_2.detach().cpu().numpy(),ort_session.get_inputs()[1].name: input3_3.detach().cpu().numpy(),ort_session.get_inputs()[2].name: input3_7.detach().cpu().numpy(),ort_session.get_inputs()[3].name: input3_8.detach().cpu().numpy(),ort_session.get_inputs()[4].name: input3_9.detach().cpu().numpy(),ort_session.get_inputs()[5].name: input3_10.detach().cpu().numpy(),ort_session.get_inputs()[6].name: input3_11.detach().cpu().numpy(),ort_session.get_inputs()[7].name: input3_12.detach().cpu().numpy(),ort_session.get_inputs()[8].name: input3_13.detach().cpu().numpy(),ort_session.get_inputs()[9].name: input3_14.detach().cpu().numpy(),ort_session.get_inputs()[10].name: input3_15.detach().cpu().numpy(),ort_session.get_inputs()[11].name: input3_16.detach().cpu().numpy(),ort_session.get_inputs()[12].name: input3_17.detach().cpu().numpy(),ort_session.get_inputs()[13].name: input3_18.detach().cpu().numpy(),ort_session.get_inputs()[14].name: input3_19.detach().cpu().numpy(),ort_session.get_inputs()[15].name: input3_20.detach().cpu().numpy(),ort_session.get_inputs()[16].name: input3_21.detach().cpu().numpy(),ort_session.get_inputs()[17].name: input3_22.detach().cpu().numpy(),ort_session.get_inputs()[18].name: input3_23.detach().cpu().numpy(),ort_session.get_inputs()[19].name: input3_24.detach().cpu().numpy(),ort_session.get_inputs()[20].name: input3_25.detach().cpu().numpy(),ort_session.get_inputs()[21].name: input3_26.detach().cpu().numpy(),ort_session.get_inputs()[22].name: input3_27.detach().cpu().numpy(),ort_session.get_inputs()[23].name: input3_28.detach().cpu().numpy(),ort_session.get_inputs()[24].name: input3_29.detach().cpu().numpy(),ort_session.get_inputs()[25].name: input3_30.detach().cpu().numpy(),ort_session.get_inputs()[26].name: input3_31.detach().cpu().numpy(),ort_session.get_inputs()[27].name: input3_32.detach().cpu().numpy(),ort_session.get_inputs()[28].name: input3_33.detach().cpu().numpy(),ort_session.get_inputs()[29].name: input3_34.detach().cpu().numpy()}


        ort_outs = ort_session.run(None, ort_inputs)
        print('ort_outs:')
        print(ort_outs)

        output = net(new_data)
        results_output = output.detach().cpu().numpy() 
        print('net output:')
        print(results_output)
        results = ort_outs[0]
        print('--------------------------------------------------------')
        
        np.testing.assert_allclose(results_output, results, rtol=1e-03, atol=1e-05)
        print('---------------------------pass!!-----------------------------')

    #     for i, (argo_idx, pred_traj) in enumerate(zip(data["argo_id"], results)):
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
    vizdata.viz_predictions(input_, preds[1], gt_, cities[1])
    # save for further visualization
    res = dict(
        preds = preds,
        gts = gts,
        cities = cities,
    )




if __name__ == "__main__":
    main()
