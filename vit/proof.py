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



def get_redundant_task_vector(key, vectors, layer_input, args, iter_num = 300,):
    if args.model == 'ViT-B-32':
        patch_num = (224//32)**2+1
    elif args.model == 'ViT-B-16':
        patch_num = (224//16)**2+1
    elif args.model == 'ViT-L-14':
        patch_num = (224//14)**2+1

    layer_input = layer_input.cuda()
    layer_input = layer_input.reshape(8, args.sample_size * patch_num, -1)

    # # if 'in_proj' in key:
    # #     print(layer_input[:, :, : layer_input.shape[-1]//3] - layer_input[:, :, layer_input.shape[-1]//3 : layer_input.shape[-1]*2//3])
    # #     print(layer_input[:, :, layer_input.shape[-1]//3 : layer_input.shape[-1]*2//3] - layer_input[:, :, layer_input.shape[-1]*2//3 : layer_input.shape[-1]])
    vectors = vectors.cuda() 


    merging_vector = torch.nn.Parameter((torch.sum(vectors, dim = 0)))
    # optimizer = torch.optim.AdamW([merging_vector], lr=1e-5)
    if 'in_proj' in key:
        layer_input = layer_input[:, :, : layer_input.shape[-1]//3]
    # l2_norms = torch.norm(layer_input.reshape(8, -1), p=2, dim=1)

    # for i in tqdm(range(iter_num)):

    #     inner_product = torch.matmul(layer_input, merging_vector.unsqueeze(0).transpose(1, 2)- vectors.transpose(1, 2)) / l2_norms.unsqueeze(-1).unsqueeze(-1)
    #     loss =  torch.norm(inner_product,p=2) 
    #     optimizer.zero_grad()          
    #     loss.backward()
    #     optimizer.step()
    print(key)
    for j in range(len(vectors)):
        for i in range(len(layer_input)):
            inp = layer_input[i]
            # print((vectors[j].T @ torch.linalg.pinv(torch.matmul(vectors[j], vectors[j].T)) @ vectors[j]).shape)
            # print(torch.linalg.pinv(torch.matmul(vectors[j], vectors[j].T)))
            # print((vectors[j].T @ torch.linalg.pinv(torch.matmul(vectors[j], vectors[j].T)) @ vectors[j]).shape)
            recons_error = inp.T - vectors[j].T @ torch.linalg.pinv(torch.matmul(vectors[j], vectors[j].T)) @ vectors[j] @ inp.T
            # print(recons_error)
            print('task:',j,'input',i)
            # print(torch.norm(inp,p=2))
            # print(torch.norm(recons_error,p=2))
            print(torch.square(torch.norm(recons_error,p=2)/torch.norm(inp,p=2)))
            # rate = torch.norm(recons_error,p=2)/torch.norm(inp,p=2)
            # print(rate)

    return merging_vector.data.detach().cpu()

def decompose_task_vectors(task_vectors, inputs, args, iter_num = 300):
    merged_task_vector = {}
    keys = [key for key in task_vectors[0].vector.keys() if ('attn.in_proj_weight' in key or 'attn.out_proj.weight' in key or 'mlp.c_fc.weight' in key or 'mlp.c_proj.weight' in key)]

    for key in tqdm(keys):

        merged_task_vector[key] = torch.zeros_like(task_vectors[0].vector[key])
        values = deepcopy(torch.stack([task_vector.vector[key] for task_vector in task_vectors]))
        merigng_vector = get_redundant_task_vector(key, values, torch.stack(inputs[key]), args, iter_num)
        merged_task_vector[key] += merigng_vector

    return TaskVector(vector=merged_task_vector)



def get_input(name, inputs):
    def hook(model, in_f, out_f):
        in_data = torch.cat(in_f,dim= - 1).detach().cpu()
        in_data = in_data.reshape(in_data.shape[0]*in_data.shape[1],-1)

        if name in inputs:
            inputs[name].append(in_data)
        else:
            inputs[name] = []
            inputs[name].append(in_data)
    return hook

from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle

def hook_model(model, inputs):
    for i in range(len(model.model.visual.transformer.resblocks)):
        attn = model.model.visual.transformer.resblocks[i].attn 
        # print(type(attn))
        multi_attn = MultiheadAttention_Custom(attn.embed_dim ,attn.num_heads,attn.batch_first)
        # print(type(multi_attn))
        model.model.visual.transformer.resblocks[i].attn = multi_attn.to(args.device)
        model.model.visual.transformer.resblocks[i].attn.load_state_dict(attn.state_dict())

    for name, module in model.named_modules():
        if 'attn' == name.split('.')[-1]:
            key = name + '.in_proj_weight'
            module.register_forward_hook(get_input(key, inputs))
        # return model

def get_change(dataset_name, key, A, B):

    if args.model == 'ViT-B-32':
        patch_num = (224//32)**2+1
    elif args.model == 'ViT-B-16':
        patch_num = (224//16)**2+1
    elif args.model == 'ViT-L-14':
        patch_num = (224//14)**2+1

    A = torch.stack(A)
    B = torch.stack(B)
    A = A.reshape(args.sample_size * patch_num,-1)
    B = B.reshape(args.sample_size * patch_num,-1)

    norm_A = torch.norm(A, dim=1, keepdim=True)
    norm_B = torch.norm(B, dim=1, keepdim=True)

    length_change_percentage = (torch.abs(norm_B - norm_A) / norm_A).reshape(-1)

    direction_A = A / norm_A 
    direction_B = B / norm_B
    
    cosine_similarity = 1 - torch.einsum('ij,ij->i', direction_A, direction_B) 
    cosine_similarity = 1 - torch.sum(direction_A*direction_B,dim=1)
    print(cosine_similarity.shape)
    #print(length_change_percentage.shape, cosine_similarity.shape)
    # plt.figure(figsize=(10, 8))
    print(key)
    print(cosine_similarity.mean(),length_change_percentage.mean())
    return  cosine_similarity.mean(),length_change_percentage.mean()

    
def plot_change(x, y, value, dataset_name, num, axes ):

    ax = axes[num // 4, num % 4]  # 获取当前子图的轴
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    scatter = ax.scatter(x, y, c=value, cmap='plasma', s=120, alpha=0.6)

    # 添加颜色条
    if (num + 1) % 4 == 0:  # 每行最后一个子图添加颜色条
        cax = add_right_cax(ax, pad=0.01, width=0.012, a = (num)//4 )  # 获取当前轴并添加颜色条轴
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('Depth of Layer',fontdict={'size': 16})

    ax.set_xlabel('$\Delta$Direction', fontdict={'size': 16})
    ax.set_ylabel('$\Delta$Magnitude', fontdict={'size': 16})
    ax.set_title(dataset_name, fontdict={'size': 16})

    

def add_right_cax(ax, pad, width, a):
    pos = ax.get_position()
    if a == 0:
        cax = plt.axes([pos.x1 + pad, pos.y0+0.05, width, pos.y1 - pos.y0])  # 创建新的轴
    else:
        cax = plt.axes([pos.x1 + pad, pos.y0-0.02, width, pos.y1 - pos.y0])  # 创建新的轴
    return cax


def get_inputs_fromkeys(datasets, args):
    num = 0
    fig, axes = plt.subplots(2, 4, figsize=(19, 9),dpi=1000)
    for dataset_name in tqdm(datasets):

        pretrain_input = {}
        ft_input = {}

        ft_path = 'checkpoints/'+args.model+'/'+dataset_name+'/finetuned.pt'

        pretrained_model = torch.load(pretrained_checkpoint).to(args.device)
        hook_model(pretrained_model, pretrain_input)
        ft_model = torch.load(ft_path).to(args.device)
        hook_model(ft_model, ft_input)



        dataset = get_dataset(dataset_name, pretrained_model.val_preprocess, location=args.data_location, batch_size=args.bs)
        dataloader = get_dataloader(dataset,is_train=True, args=args)
        # pretrained_model.eval()
        for i in tqdm(range(args.sample_size // args.bs)):
            data = next(iter(dataloader))  # get one batch
            data = maybe_dictionarize(data)
            x = data['images'].to(args.device)
            outputs = pretrained_model(x)
            outputs = ft_model(x)
        cos_list = []
        direc_list = []
        value_list = []
        i = 0
        for key in pretrain_input:
            cos, direc = get_change(dataset_name, key, pretrain_input[key], ft_input[key])
            cos_list.append(cos)
            direc_list.append(direc)
            value_list.append(i)
            i += 1

        plot_change(cos_list, direc_list, value_list, dataset_name, num, axes)
        num += 1
    plt.tight_layout()
    plt.subplots_adjust(right=0.9) 
    plt.savefig('fig/change.svg',format='svg', dpi=1000)
        # break


    del pretrained_model, outputs, x 
    torch.cuda.empty_cache()

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

    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))

    if args.use_wandb:
        run = wandb.init(config=args, name=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), save_code=False)
        print("real_config", run.config)

    print("args = ", args)
    inputs = get_inputs_fromkeys(exam_datasets, args)
