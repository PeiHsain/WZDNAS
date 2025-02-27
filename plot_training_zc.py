# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import cv2
import os
import sys
from datetime import datetime
import yaml
import torch
import numpy as np
import torch.nn as nn
import tqdm
import shutil
import scipy.stats as stats
# import _init_paths
import sys, json, copy
sys.path.insert(0, 'lib')

# import timm packages
from timm.utils import CheckpointSaver, update_summary

# import apex as distributed package otherwise we use torch.nn.parallel.distributed as distributed package
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model
    USE_APEX = True
except ImportError:
    from torch.nn.parallel import DataParallel as DDP
    USE_APEX = False

# import models and training functions
from lib.utils.flops_table import FlopsEst
from lib.core.izdnas import train_epoch_dnas, train_epoch_dnas_V2, train_epoch_zdnas
from lib.models.structures.supernet import gen_supernet
from lib.utils.util import convert_lowercase, get_logger, \
    create_optimizer_supernet, create_supernet_scheduler, stringify_theta, write_thetas, export_thetas
from lib.utils.datasets import create_dataloader
from lib.utils.general import check_img_size, labels_to_class_weights, is_parallel, compute_loss, test, ModelEMA, random_testing
from lib.utils.torch_utils import select_device
from lib.config import cfg
import argparse
import random
import glob
from natsort import natsorted
from itertools import combinations
import imageio
import matplotlib.pyplot as plt


def analyze_zcmap(model, zero_cost_func, image_idx):
    """
    analyze the model ranking in stage.
    """
    prob = model.softmax_sampling(detach=True)
    
    zc_map = zero_cost_func(prob, image_idx)
    zc_map = str(zc_map)
    # print(zc_map)

    return zc_map

def arch_generator(zc_map):
    # f= open(filename, 'r')
    # for idx, item in enumerate(f):
        # if idx % 2 == 1:
    if True:
        # idx_string, *content = item.split(' ')
        
        # Parse Epoch and Iteration
        # idx_string = idx_string[1:-1]
        # _, epoch_idx, iter_idx = idx_string.split('-')
        # epoch_idx, iter_idx = int(epoch_idx), int(iter_idx)
        
        # Parse Architecture Prob
        # map = str(zc_map)
        arch_prob_raw = ''.join(zc_map)
        arch_prob_raw = arch_prob_raw.replace("gamma", "g").replace("n_bottlenecks", "n")
        arch_prob_raw = arch_prob_raw.replace("inf", "0")
        

        # print(epoch_idx, iter_idx)
        # if (epoch_idx==1 and iter_idx <= 50): continue
        
        arch = eval(arch_prob_raw)
        # print(arch)
        # if iter_idx % 200 == 0:
        if True:
            yield arch

def analyze_map_func2(arch_info_list, img_filename):
    # zc_maps1 = arch_info1['naswot_map']
    # zc_maps2 = arch_info2['naswot_map']
    # arch1    = arch_info1['arch']
    # arch2    = arch_info2['arch']
    write_img  = img_filename is not None
    fig, axes = plt.subplots(8)
    # fig, axes = plt.subplots(9)
    fig.suptitle('Pruned zc scores')
    # print(len(arch_info_list))
    for stage_id in range(len(arch_info_list)):
        score_list = []
        rank_list = []
        
        keys = arch_info_list[stage_id].keys()
        print(keys)
        
        # for idx, arch_info in enumerate(arch_info_list):
        #     print(arch_info)
        zc_map = arch_info_list[stage_id]
        print(zc_map)
        score  = np.array([zc_map[key] for key in keys])
        rank   = (-score ).argsort()[::-1]
        
        score_list.append(score)
        rank_list.append(rank)

        candidiate_num  = len(rank)
        comp_list  = score_list
        color_list = ['r', 'g', 'b', 'c', 'k', 'm']
        # arch_list  = [arch_info['arch'] for arch_info in arch_info_list]
        x = np.arange(candidiate_num) * 0.8
        
        with_val = 0.1
        for i, score_arr in enumerate(comp_list):
            axes[stage_id].bar(x - with_val* (i-len(comp_list)/3), height=score_arr, width=with_val, color=[color_list[i]]*candidiate_num, align='edge')
        #######################################
        # Basic Math Information
        #######################################
        margin = 0.2
        all_scores = np.concatenate(comp_list)
        center  = all_scores.mean()
        min_val = all_scores.min() - 0.05
        max_val = all_scores.max() + 0.05
        
        #######################################
        # Set Plot Style
        #######################################
        axes[stage_id].set_ylim([min_val, max_val])
        axes[stage_id].set_xticks(x, list(keys))
        axes[stage_id].set_ylabel(f'Depth={stage_id}')
        axes[stage_id].legend([f'Arch{stage}' for stage in range(len(comp_list))], labelcolor=color_list)
        
        for ii in range(3,11,4): axes[stage_id].axvline((ii+0.5)*0.8, color='black')
        arr_size  = (max_val-min_val)*with_val
        
        # Rank Plot
        for ii, (rank, score, color) in enumerate(zip(rank_list,comp_list,color_list)):
            for iii in range(4):
                loc = x[rank[iii]] - with_val * (ii-len(comp_list)/3) #+ 0.024
                axes[stage_id].text(loc, score[rank[iii]]-arr_size*1.2, str(iii+1), color=color)

    fig.set_size_inches(15.5, 15.5)
    fig.tight_layout()
    fig.savefig(img_filename)
    # plt.tight_layout()
    # plt.savefig(img_filename, dpi=300)
    # plt.close(fig)
    # fig.set_size_inches(15.5, 15.5)
    # fig.tight_layout()
    # fig.savefig(img_filename)
    # if img_filename is not None: 
    # return fig

def analyze_ranking_epoch_info(model, args, zero_cost_func):
    IMAGE_IDX = 0
    NUM_STAGES = len(model.searchable_block_idx)

    searchable_block_name = [f'blocks.{block_id}' for block_id in model.searchable_block_idx ]
    print('searchable_block_name', searchable_block_name)

    model.load_state_dict(torch.load(args.pre_weights))

    model.eval()
    zc_map = analyze_zcmap(model, zero_cost_func, IMAGE_IDX)

    gen = arch_generator(zc_map)
    # print(gen)
    # for idx_info, arch in gen:
            # print(f'idx_string={idx_info}')
    for arch in gen:
        # print(arch)
        analyze_map_func2(arch, 'zc_score.jpg')
        # analyze_map_func2([{'naswot_map': gen}], 'zc_score.jpg')



def parse_config_args(exp_name):
    parser = argparse.ArgumentParser(description=exp_name)
    ###################################################################################
    # Commonly Used Parameter !!
    ###################################################################################
    parser.add_argument('--cfg',  type=str, default='config/search/exp_v4.yaml',           help='configuration of cream')
    parser.add_argument('--data', type=str, default='config/dataset/voc_dnas.yaml',              help='data.yaml path')
    parser.add_argument('--hyp',  type=str, default='config/training/hyp.zerocost.yaml', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--model',type=str, default='config/model/Search-YOLOv4-CSP.yaml',       help='model path')
    parser.add_argument('--exp_series', type=str, default='exp_series', help="name of experiments")
    parser.add_argument('--pre_weights', type=str, default='', help='pretrained model weights')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--zc', type=str, default='naswot', help='zero cost metrics')
    
    ###################################################################################
    
    
    # ###################################################################################
    # # Seldom Used Parameter
    # ###################################################################################
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    # parser.add_argument('--collect-samples', type=int, default=0, help='Sample a lot of different architectures with corresponding flops, if not 0 then samples specified number and exits the programm')
    # parser.add_argument('--collect-synflows', type=int, default=0, help='Sample a lot of different architectures with corresponding synflows, if not 0 then samples specified number and exits the programm')
    # parser.add_argument('--resume-theta-training', default='', type=str, help='load pretrained thetas')
    # ###################################################################################
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    converted_cfg = convert_lowercase(cfg)

    return args, converted_cfg

def main():
    args, cfg = parse_config_args('super net training')
    
    #######################################
    # Model Config
    #######################################
    with open(args.model ) as f:
        model_args   = yaml.load(f, Loader=yaml.FullLoader)
    search_space = model_args['search_space']

    # task_name = args.nas if args.nas != '' else 'DNAS-25'
    # TASK_FLOPS      = task_dict[task_name]['GFLOPS']     # e.g TASK_FLOPS  = 5  means 50 GFLOPs
    # TASK_PARAMS     = task_dict[task_name]['PARAMS']     # e.g TASK_PARAMS = 32 means 32 million parameters.
    SEARCH_SPACES   = model_args['search_space']
    USE_AMP         = False
    FLOP_RESOLUTION = (None, 3, cfg.search_resolution, cfg.search_resolution)
    
    output_dir = 'tmp'

    if args.local_rank == 0:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        logger_path = os.path.join(output_dir, "train.log")
        with open(logger_path, 'w') as file:
            pass
        logger = get_logger(logger_path)
    else:
        logger = None

    #######################################
    # Dataset Config
    #######################################
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_weight_path = data_dict['train_weight']
    train_thetas_path = data_dict['train_thetas']
    test_path         = data_dict['val']
    nc, names = (1, ['item']) if args.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, args.data)  # check

    # Number of class in data config would override other config
    if data_dict['nc'] != cfg.DATASET.NUM_CLASSES:
        logger.info(f"args.data with nc={data_dict['nc']} override the {args.cfg} cfg.DATASET.NUM_CLASSES={cfg.DATASET.NUM_CLASSES}")
        cfg.DATASET.NUM_CLASSES = data_dict['nc']
    if data_dict['nc'] != model_args['nc']:
        logger.info(f"args.data with nc={data_dict['nc']} override the {args.model} args.model['nc']={model_args['nc']}")
        model_args['nc'] = data_dict['nc']
        
    #######################################
    # Hyper Parameter Config
    #######################################
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset


    # initialize distributed parameters
    device = select_device(args.device, batch_size=cfg.DATASET.BATCH_SIZE)
    cfg.NUM_GPU = torch.cuda.device_count()
    cfg.WORKERS = torch.cuda.device_count()
    
    args.world_size = 1
    args.global_rank = -1
    if args.local_rank == 0:
        logger.info(
            'Training on Process %d with %d GPUs.',
                args.local_rank, cfg.NUM_GPU)

    # fix random seeds
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model, sta_num, resolution = gen_supernet(
        model_args,
        num_classes=cfg.DATASET.NUM_CLASSES,
        verbose=cfg.VERBOSE,
        logger=logger,
        init_temp=cfg.TEMPERATURE.INIT)

    # number of choice blocks in supernet
    if args.local_rank == 0:
        logger.info('Supernet created, param count: %.2f M', (
            sum([m.numel() for m in model.parameters()]) / 1e6))
        logger.info('resolution: %d', (cfg.DATASET.IMAGE_SIZE))

    # initialize flops look-up table
    model_est = FlopsEst(model, input_shape=(None, 3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE), search_space=SEARCH_SPACES, signature=args.model)

    optimizer, theta_optimizer = create_optimizer_supernet(cfg, model, USE_APEX)
    model.module.update_main() if is_parallel(model) else model.update_main()

    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in [cfg.DATASET.IMAGE_SIZE] * 2]

    cuda = device.type != 'cpu'
    if cuda and torch.cuda.device_count() > 1 and args.local_rank != -1:
        model = torch.nn.DataParallel(model)
        print(f'Using {torch.cuda.device_count()} GPUs. device: {device}')

    model = model.to(device)

    # dataloader_weight, dataset_weight = create_dataloader(train_weight_path, imgsz, cfg.DATASET.BATCH_SIZE, gs, args, hyp=hyp, augment=True,
    #                                         cache=args.cache_images, rect=args.rect,
    #                                         world_size=args.world_size)
    
    # dataloader_thetas, dataset_thetas = create_dataloader(train_thetas_path, imgsz, cfg.DATASET.BATCH_SIZE, gs, args, hyp=hyp, augment=True,
    #                                         cache=args.cache_images, rect=args.rect,
    #                                         world_size=args.world_size)
    
    dataloader_weight, dataset_weight = create_dataloader(train_thetas_path, imgsz, cfg.DATASET.BATCH_SIZE, gs, args, hyp=hyp, augment=True,
                                            cache=args.cache_images, rect=args.rect,
                                            world_size=args.world_size)
    dataloader_thetas, dataset_thetas = create_dataloader(train_thetas_path, imgsz, cfg.DATASET.BATCH_SIZE, gs, args, hyp=hyp, augment=True,
                                            cache=args.cache_images, rect=args.rect,
                                            world_size=args.world_size)
    
    img_pairs = []
    target_pairs = []
    for iter_idx, (uimgs, targets, paths, _) in enumerate(dataloader_weight):
        # imgs = (batch=2, 3, height, width)
        imgs     = uimgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        targets  = targets.to(device)
        
        imgs=imgs
        targets=targets
        
        img_pairs.append(imgs)
        target_pairs.append(targets)
        
        if iter_idx == 10: break

    mlc = np.concatenate(dataset_weight.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader_weight)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, args.data, nc - 1)


    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset_weight.labels, nc).to(device)  # attach class weights
    model.names = names
    print('[Info] cfg.TEMPERATURE.INIT', cfg.TEMPERATURE.INIT)
    print('[Info] cfg.TEMPERATURE.FINAL', cfg.TEMPERATURE.FINAL)

    is_ddp = is_parallel(model)
    ##################################################################
    ### Choice a Zero-Cost Method
    ##################################################################  
    model.eval() 
    prob = model.softmax_sampling(detach=True)
    
    from lib.zero_proxy import naswot, snip
    PROXY_DICT = {
        'naswot': naswot.calculate_zero_cost_map2,
        'snip':   snip.calculate_zero_cost_map,    
    }
    zc_function_list = {
        'naswot':  lambda arch_prob, idx, short_name=None: PROXY_DICT['naswot'](model, arch_prob, img_pairs[idx][:2], target_pairs[idx][:2], short_name=short_name),
        'snip':    lambda arch_prob, idx, short_name=None: PROXY_DICT['snip'](model, arch_prob, img_pairs[idx], target_pairs[idx], short_name=short_name),
    }
    if args.zc not in PROXY_DICT.keys():
        raise Value(f"key {args.zc} is not registered in PROXY_DICT")
    
    analyze_ranking_epoch_info(model, args, zc_function_list[args.zc])
    
if __name__ == '__main__':
    main()
