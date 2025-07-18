import os
import numpy as np
import time
import wandb
import sys
from task_vectors import TaskVector
from eval import eval_single_dataset, eval_single_dataset_with_loss
from args import parse_arguments
import torch
from tqdm import tqdm
import datetime
import torch.nn.functional as F
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import random
from collections import OrderedDict
import torch.nn as nn
from multi_head import MultiheadAttention_Custom
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid') 
plt.rc('font',family='Times New Roman')
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
# from model import get_model
sys.path.append('./')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger



def get_redundant_task_vector(key, vectors, iter_num = 300, ratio = 1):

    # iter_num = 0

    vectors = vectors.cuda() # vectors.shape = [8, 2304, 768]

    merging_vector = torch.nn.Parameter((torch.sum(vectors, dim = 0)))

    optimizer = torch.optim.Adam([merging_vector], lr=1e-5, weight_decay= 0)

    l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim= -1))

    # index_list = random.sample(range(0,vectors.shape[1]), int(vectors.shape[1]*ratio))
    # print('length of index_list: ',len(index_list))
    # index_list = torch.LongTensor(index_list)

    # random_cons = torch.rand(vectors.shape).cuda()

    for i in tqdm(range(iter_num)):
        disturbing_vectors = merging_vector.unsqueeze(0)- vectors
        inner_product = torch.matmul(disturbing_vectors , vectors.transpose(1,2)) 

        loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1) )
        optimizer.zero_grad()          
        loss.backward()
        optimizer.step()

    return merging_vector.data.detach().cpu()

def decompose_task_vectors(task_vectors, iter_num = 300, ratio = 1):

    merged_task_vector = {}
    keys = [key for key in task_vectors[0].vector.keys() if ('attn.in_proj_weight' in key or 'attn.out_proj.weight' in key or 'mlp.c_fc.weight' in key or 'mlp.c_proj.weight' in key)]
    for key in tqdm(keys):
        merged_task_vector[key] = torch.zeros_like(task_vectors[0].vector[key])
        values = deepcopy(torch.stack([task_vector.vector[key] for task_vector in task_vectors]))
        merigng_vector = get_redundant_task_vector(key, values, iter_num, ratio = ratio)
        merged_task_vector[key] += merigng_vector

    return TaskVector(vector=merged_task_vector)



def get_output(name, outputs):
    def hook(model, in_f, out_f):

        output_data = out_f.reshape(-1,out_f.shape[-1]).detach().cpu()

        if name in outputs:
            outputs[name].append(output_data)
        else:
            outputs[name] = []
            outputs[name].append(output_data)
    return hook

from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle

def hook_model(model, outputs):

    for name, module in model.named_modules():
        # print(name)
        if 'mlp.c_fc' in name:

            module.register_forward_hook(get_output(name, outputs))
        # return model


    
def plot_interference(wudi, ta, datasets, args):

    fig, axes = plt.subplots(2, 4, figsize=(19, 9),dpi=1000)
    # plt.style.use('ggplot')  # 选择一个接近 ICML 的风格

    for num in range(len(datasets)):

        wudi_data = []
        ta_data = []
        keys = []

        dataset_name = datasets[num]


        d = 0
        for key in wudi:
            wudi_data.append(wudi[key][num])
            ta_data.append(ta[key][num])
            keys.append("Layer "+str(d))
            d = d+1


        # 设置柱状图的宽度
        bar_width = 0.35
        x = np.arange(len(keys))  # 类别的 x 位置

        # 创建柱状图
        axes[num // 4, num % 4].bar(x - bar_width/2, wudi_data, width=bar_width, label='WUDI-Merging', color='blue')
        axes[num // 4, num % 4].bar(x + bar_width/2, ta_data, width=bar_width, label='Task Arithmetic', color='orange')

        # 添加标题和标签
        axes[num // 4, num % 4].set_title(dataset_name, fontsize=16)
        axes[num // 4, num % 4].set_xlabel('Layers', fontsize=14)
        axes[num // 4, num % 4].set_ylabel('Relative Error', fontsize=14)
        axes[num // 4, num % 4].set_xticks(x)  # 设置 x 轴的标签
        axes[num // 4, num % 4].set_xticklabels(keys, rotation=45)  # 设置 x 轴的标签
        axes[num // 4, num % 4].legend()  # 显示图例



    # 调整布局
    plt.tight_layout()
    plt.savefig(args.model+'_each_interference.svg',format="svg", dpi = 1000 )  # 请根据需要修改文件名
    plt.show()

    if args.model == 'ViT-B-32':
        title = "ViT-B/32"
    elif args.model == 'ViT-B-16':
        title = "ViT-B/16"
    elif args.model == 'ViT-L-14':
        title = "ViT-L/14"


    wudi_data = []
    ta_data = []
    keys = []
    d = 0
    for key in wudi:
        wudi_data.append(np.mean(wudi[key]))
        ta_data.append(np.mean(ta[key]))
        keys.append("Layer " + str(d))
        d += 1

    # 创建一个新的图形
    plt.figure(figsize=(7, 6),dpi=1000)

    # 创建柱状图
    plt.bar(x - bar_width/2, wudi_data, width=bar_width, label='WUDI-Merging', color='blue')
    plt.bar(x + bar_width/2, ta_data, width=bar_width, label='Task Arithmetic', color='orange')

    # 添加标题和标签
    plt.title("Interference of Each Layer("+title +")", fontsize=16)
    plt.xlabel('Layers', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.xticks(x, keys, rotation=45)  # 设置 x 轴的标签
    plt.legend()  # 显示图例

    # 调整布局
    plt.tight_layout()
    plt.savefig(args.model + '_mean_interference.svg',format="svg", dpi = 1000 )  # 请根据需要修改文件名
    plt.show()


    #     # 显示图形
    # plt.tight_layout()
    # plt.savefig(args.model+'_mean_interference.png')
    # plt.show()



def get_outputs_fromkeys(datasets, args, wudi_model, ta_model):
    num = 0
    

    layer_difference_wudi = {}
    layer_difference_ta = {}

    for dataset_name in tqdm(datasets):

        wudi_output = {}
        ta_output = {}
        ft_output = {}

        ft_path = 'checkpoints/'+args.model+'/'+dataset_name+'/finetuned.pt'

        ft_model = torch.load(ft_path).to(args.device)
        wudi_model = wudi_model.to(args.device)
        ta_model = ta_model.to(args.device)

        hook_model(ft_model, ft_output)
        hook_model(wudi_model, wudi_output)
        hook_model(ta_model, ta_output)

        dataset = get_dataset(dataset_name, ft_model.val_preprocess, location=args.data_location, batch_size=args.bs)
        dataloader = get_dataloader(dataset,is_train=True, args=args)
        # pretrained_model.eval()
        for i in tqdm(range(args.sample_size // args.bs)):
            data = next(iter(dataloader))  # get one batch
            data = maybe_dictionarize(data)
            x = data['images'].to(args.device)
            outputs = ft_model(x)
            outputs = wudi_model(x)
            outputs = ta_model(x)

        i = 0

        print(wudi_output)

        for key in ft_output:
            wudi = torch.stack(wudi_output[key],dim=0)
            ta = torch.stack(ta_output[key],dim=0)
            ft = torch.stack(ft_output[key],dim=0)

            wudi = wudi.reshape(-1,wudi.shape[-1])
            ta = ta.reshape(-1,ta.shape[-1])
            ft = ft.reshape(-1,ft.shape[-1])


            # print(wudi.shape)
            # print(torch.norm(wudi - ft, p=2,dim=1).shape)
            # print(ta.shape)
            # print(torch.norm(ta - ft, p=2,dim=1).shape)
            # print(ft.shape)

            # diff_wudi = torch.mean(torch.norm(wudi - ft, p=2,dim=1)**2)
            # diff_ta = torch.mean(torch.norm(ta - ft, p=2,dim=1)**2)

            diff_wudi = torch.mean(torch.norm(wudi - ft, p=2,dim=1)/torch.norm(ft, p=2,dim=1))
            diff_ta = torch.mean(torch.norm(ta - ft, p=2,dim=1)/torch.norm(ft, p=2,dim=1))

            if key in layer_difference_wudi:
                layer_difference_wudi[key].append(diff_wudi)
            else:
                layer_difference_wudi[key] = [diff_wudi]

            if key in layer_difference_ta:
                layer_difference_ta[key].append(diff_ta)
            else:
                layer_difference_ta[key] = [diff_ta]

    plot_interference(layer_difference_wudi, layer_difference_ta, datasets, args)



    #     plot_change(cos_list, direc_list, value_list, dataset_name, num, axes)
    #     num += 1
    # plt.tight_layout()
    # plt.subplots_adjust(right=0.9) 
    # plt.savefig('fig/change.svg',format='svg', dpi=1000)
        # break



    # return inputs


        
if __name__ == '__main__':

    # setup_seed(0)

    exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']

    args = parse_arguments()
    args.data_location = 'data'
    args.save = 'checkpoints/' + args.model
    args.logs_path = 'logs/' + args.model
    args.sample_size = 20
    args.bs = 20
    pretrained_checkpoint = 'checkpoints/'+ args.model +'/zeroshot.pt'
    pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()

    print("args = ", args)

    task_vectors = [TaskVector(pretrained_checkpoint, 'checkpoints/'+args.model+'/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets]
    task_vector = decompose_task_vectors(task_vectors, iter_num = args.iter)
    wudi_model = task_vector.apply_to(pretrained_checkpoint, scaling_coef= 1 )

    task_vectors = [TaskVector(pretrained_checkpoint, 'checkpoints/'+args.model+'/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets]
    task_vector_sum = sum(task_vectors)
    ta_model = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef= 0.4)

    output_diff = get_outputs_fromkeys(exam_datasets, args, wudi_model, ta_model)
