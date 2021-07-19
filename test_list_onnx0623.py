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
import onnxruntime

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


# define parser
parser = argparse.ArgumentParser(description="Argoverse Motion Forecasting in Pytorch")
parser.add_argument(
    "-m", "--model", default="angle90", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true", default=True)
parser.add_argument(
    "--split", type=str, default="val", help='data split, "val" or "test"'
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)

def to_numpy(tensor):
        # return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        return tensor.cpu().numpy()

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
        print(data.keys()) # dict_keys(['orig', 'theta', 'rot', 'feats', 'ctrs', 'argo_id', 'city', 'graph_ctrs', 'graph_num_nodes', 'graph_feats', 'graph_turn', 'graph_control', 'graph_intersect', 'graph_pre', 'graph_suc', 'graph_lane_idcs', 'graph_pre_pairs', 'graph_suc_pairs', 'graph_left_pairs', 'graph_right_pairs', 'graph_left', 'graph_right'])

        with torch.no_grad():
            data['feats'] = gpu(data['feats'])
            data['ctrs'] = gpu(data['ctrs'])
            data= graph_gather(to_long(gpu(data)))
            print('after graph gather:-------')
            #print(type(data['graph'])) # <class 'list'>

            # input_graph = (data['graph_ctrs'],data['graph_num_nodes'],data['graph_feats'],data['graph_turn'],data['graph_control'],data['graph_intersect'],data['graph_pre'],data['graph_suc'],data['graph_lane_idcs'],data['graph_pre_pairs'],data['graph_suc_pairs'],data['graph_left_pairs'],data['graph_right_pairs'],data['graph_left'],data['graph_right'])
            

            # print('-----------------------graph')
            # print(len(data['graph']))  # 10
            # print(data['graph'])


            #print(data['graph'][0][0].dtype) # torch.int64

            data['rot'] = gpu(data['rot'])
            data['orig'] = gpu(data['orig'])

#     # 添加graph中的10项 ['graph_idcs', 'ctrs', 'feats', 'turn', 'control', 'intersect', 'pre', 'suc', 'left', 'right']):
            data['graph_idcs'] = gpu(data['graph_idcs'])
            data['graph_ctrs'] = gpu(data['graph_ctrs'])
            data['graph_feats'] = gpu(data['graph_feats'])
            data['graph_turn'] = gpu(data['graph_turn'])
            data['graph_control'] = gpu(data['graph_control'])
            data['graph_intersect'] = gpu(data['graph_intersect'])
            data['graph_pre'] = gpu(data['graph_pre'])
            data['graph_suc'] = gpu(data['graph_suc'])
            data['graph_left'] = gpu(data['graph_left'])
            data['graph_right'] = gpu(data['graph_right'])

            # new_data = (data['feats'], data['ctrs'], data['graph'], data['rot'], data['orig']) #  tensor

            new_data = (data['feats'], data['ctrs'], data['rot'], data['orig'],data['graph_idcs'],data['graph_ctrs'],data['graph_feats'],data['graph_turn'],data['graph_control'],data['graph_intersect'],data['graph_pre'],data['graph_suc'],data['graph_left'],data['graph_right'])

            print(data['graph_pre'])
            print('aaaaaaaaaaaaa')
            print(data['graph_right'])
#  <class 'list'>
# <class 'list'>
# <class 'list'>
# <class 'list'>
# <class 'list'>
# <class 'list'>
# <class 'torch.Tensor'>
# <class 'torch.Tensor'>
# <class 'torch.Tensor'>
# <class 'torch.Tensor'>
# <class 'list'>
# <class 'list'>
# <class 'dict'>  # u: v:
# <class 'dict'>

            # for i in new_data:
            #     print(type(i))
            print('-------------------feats type--------------------')
            print(type(new_data[0]))# list
            ts = time.time()
            output = net(new_data)
            print('output:')
            print(output)

            #results_output = [x[0:1].detach().cpu().numpy() for x in  output[1]]
            #print('Net output results  ------------------------')
            #print(results)
            #print('output ------------')
            #print(len(output['reg'][0]))
            #print(len(output['reg'][0][0]))
            # torch.onnx.export(scp_mod,new_data, 'test.onnx', example_outputs=output)
            #dummy_input = new_data
            # print(dummy_input)   verbose=True,

            torch.onnx.export(net,(new_data,),"lanegcn_9.onnx",export_params=True,opset_version=11)

        ############# 验证  ,input_names = ['feats','ctrs','graph','rot','orig'],output_names=['output']
# import onnxruntime
        ort_session = onnxruntime.InferenceSession("lanegcn_9.onnx")
        print(len(ort_session.get_inputs()))
        print('ort_session input name: ')
        print(ort_session.get_inputs()[0].name,ort_session.get_inputs()[1].name)

        # compute ONNX Runtime output prediction
        # ort_inputs = {ort_session.get_inputs().name: (new_data,)}
        # ort_outs = ort_session.run(None, ort_inputs)
        # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input1),ort_session.get_inputs()[1].name: to_numpy(input2),ort_session.get_inputs()[2].name: to_numpy(input4),ort_session.get_inputs()[3].name: to_numpy(input5)}
        input1 = torch.cat(data['feats'],0)
        input2 = torch.cat(data['ctrs'],0)
        input4 = torch.cat(data['rot'],0)
        input5 = torch.cat(data['orig'],0)
        input3_1 =  torch.cat(data['graph'][0],0)
        input3_2 =  torch.cat(data['graph'][1],0)
        input3_3 =  torch.cat(data['graph'][2],0)
        input3_4 =  torch.cat(data['graph'][3],0)
        input3_5 =  torch.cat(data['graph'][4],0)
        input3_6 =  torch.cat(data['graph'][5],0)
        input3_7 =  torch.cat(data['graph'][6],0)
        input3_8 =  torch.cat(data['graph'][7],0)
        input3_9 =  torch.cat(data['graph'][8],0)
        input3_10 =  torch.cat(data['graph'][9],0)


        # ort_inputs = {ort_session.get_inputs()[0].name: input1.cpu().numpy(),ort_session.get_inputs()[1].name: input2.cpu().numpy(),ort_session.get_inputs()[2].name: input4.cpu().numpy(),ort_session.get_inputs()[3].name: input5.cpu().numpy(),ort_session.get_inputs()[4].name: input3_1.cpu().numpy(),ort_session.get_inputs()[5].name: input3_2.cpu().numpy(),ort_session.get_inputs()[6].name: input3_3.cpu().numpy(),ort_session.get_inputs()[7].name: input3_4.cpu().numpy(),ort_session.get_inputs()[8].name: input3_5.cpu().numpy(),ort_session.get_inputs()[9].name: input3_6.cpu().numpy(),ort_session.get_inputs()[10].name: input3_7.cpu().numpy(),ort_session.get_inputs()[11].name: input3_8.cpu().numpy(),ort_session.get_inputs()[12].name: input3_9.cpu().numpy(),ort_session.get_inputs()[13].name: input3_10.cpu().numpy()}

        ort_inputs = {ort_session.get_inputs()[0].name: input1.cpu().numpy(),ort_session.get_inputs()[1].name: input3_1.cpu().numpy(),ort_session.get_inputs()[2].name: input3_2.cpu().numpy(),ort_session.get_inputs()[3].name: input3_3.cpu().numpy(),ort_session.get_inputs()[4].name: input3_4.cpu().numpy(),ort_session.get_inputs()[5].name: input3_5.cpu().numpy(),ort_session.get_inputs()[6].name: input3_6.cpu().numpy(),ort_session.get_inputs()[7].name: input3_7.cpu().numpy(),ort_session.get_inputs()[8].name: input3_8.cpu().numpy(),ort_session.get_inputs()[9].name: input3_9.cpu().numpy(),ort_session.get_inputs()[10].name: input3_10.cpu().numpy()}


        ort_outs = ort_session.run(None, ort_inputs)
        print('ort_outs:')
        print(ort_outs)

        print('--------------------------------------------------------')
        
        # print(ort_session.get_inputs()[0].name,ort_session.get_inputs()[1].name,ort_session.get_inputs()[2].name,ort_session.get_inputs()[3].name)  # feats ctrs 36 37
        # print('11111111111111111')
        # print(type(ort_session.get_inputs()[0])) # tensor
        # print(ort_session.get_inputs()[0])
        # print('222222222222222222')
        # print(ort_session.get_inputs()[1])
        # print('333333333333333333333')
        # print(ort_session.get_inputs()[2])
        # print('4444444444444444444')
        # print(ort_session.get_inputs()[3])

        #print(output)
        # torch_out = torch.cat(output,0)
        # #torch_out = torch.cat(torch_out,0)
        # print(type(torch_out))
        # # # compare ONNX Runtime and PyTorch results
        # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

        # print("Exported model has been tested with ONNXRuntime, and the result looks good!")

        ts1 = time.time()
        #print(f'\n{i + 1} run take {ts1 - ts}')


    #     results = ort_outs[1]
    #    # results = [x[0:1].detach().cpu().numpy() for x in ort_outs]
    #     for i, (argo_idx, pred_traj) in enumerate(zip(data["argo_id"], results_output)):
    #         preds[argo_idx] = pred_traj.squeeze()
    #         cities[argo_idx] = data["city"][i]
    #         gts[argo_idx] = data["gt_preds"][i][0] if "gt_preds" in data else None

    # afl = ArgoverseForecastingLoader("/home/chl/code/lg/LaneGCN-master/dataset/test_obs/tmp/")
    # df = afl.get("/home/chl/code/lg/LaneGCN-master/dataset/test_obs/tmp/1.csv").seq_df
    # frames = df.groupby("OBJECT_TYPE")
    # input_ = np.zeros((20,2), dtype=float)
    # gt_ = np.zeros((30,2), dtype=float)
    # for group_name, group_data in frames:
    #     if group_name == 'AGENT':
    #         input_[:, 0] = group_data['X'][:20]
    #         input_[:, 1] = group_data['Y'][:20]
    #         gt_[:, 0] = group_data['X'][20: ]
    #         gt_[:, 1] = group_data['Y'][20: ]
    # #input: (20, 2)
    # #preds[1]: (6, 30, 2)
    # #gt: none
    # #city: str
    # vizdata.viz_predictions(input_, preds[1], gt_, cities[1])
    # # save for further visualization
    # res = dict(
    #     preds = preds,
    #     gts = gts,
    #     cities = cities,
    # )




if __name__ == "__main__":
    main()
