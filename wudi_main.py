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


    vectors = vectors.cuda() # vectors.shape = [8, 2304, 768]

    merging_vector = torch.nn.Parameter((torch.sum(vectors, dim = 0)))

    optimizer = torch.optim.Adam([merging_vector], lr=1e-5, weight_decay= 0)

    l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim= -1))

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



if __name__ == '__main__':

    setup_seed(0)

    exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    # exam_datasets  = dataset
    # print(exam_datasets)
    args = parse_arguments()
    args.data_location = 'data'
    args.save = 'checkpoints/' + args.model
    args.logs_path = 'logs/' + args.model
    pretrained_checkpoint = 'checkpoints/'+ args.model +'/zeroshot.pt'
    pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()

    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))

    if args.use_wandb:
        run = wandb.init(config=args, name=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), save_code=False)
        print("real_config", run.config)

    print("args = ", args)

    ratio = 1 

    print('ratio:', ratio)

    task_vectors = [TaskVector(pretrained_checkpoint, 'checkpoints/'+args.model+'/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets]


    task_vector = decompose_task_vectors(task_vectors, iter_num = args.iter, ratio = ratio)
    image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef= 1 )
    log.info('*'*20 + 'scaling_coef:' + str(1) + '*'*20)


    accs = []
    run_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    # run_datasets = exam_datasets
    results_metrics = {}
    for dataset in run_datasets:
        # metrics = eval_single_dataset(image_encoder, dataset, args)
        metrics, ce_loss = eval_single_dataset_with_loss(image_encoder, dataset, args)
        results_metrics[dataset] = metrics.get('top1')*100
        log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
        accs.append(metrics.get('top1')*100)
        print(f"{dataset} cross entropy loss = {ce_loss}")
        print(f"{dataset} acc = {metrics.get('top1')*100}")
    log.info('Avg ACC:' + str(np.mean(accs)) + '%')

    print('Avg ACC:' + str(np.mean(accs)) + '%')
    if args.use_wandb:
        results_metrics.update({'model': args.model, 'scaling_coef': args.scaling_coef_, 'avg_acc': np.mean(accs)})
        wandb.log(results_metrics)
        run.finish()
