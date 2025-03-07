# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from yacs.config import CfgNode as CN
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
__C = CN()
cfg = __C
__C.AUTO_RESUME = True
__C.DATA_DIR = './data/imagenet'
__C.MODEL = 'cream'
__C.RESUME_PATH = './experiments/ckps/resume.pth.tar'
__C.SAVE_PATH = './experiments/ckps/'
__C.SEED = 42
__C.LOG_INTERVAL = 50
__C.RECOVERY_INTERVAL = 0
__C.WORKERS = 4
__C.NUM_GPU = 1
__C.SAVE_IMAGES = False
__C.AMP = False
__C.OUTPUT = 'output/path/'
__C.EVAL_METRICS = 'prec1'
__C.TTA = 0  # Test or inference time augmentation
__C.LOCAL_RANK = 0
__C.VERBOSE = False
# dataset configs
__C.DATASET = CN()
__C.DATASET.NUM_CLASSES = 1000
__C.DATASET.IMAGE_SIZE = 224  # image patch size
__C.DATASET.INTERPOLATION = 'bilinear'  # Image resize interpolation type
__C.DATASET.BATCH_SIZE = 32  # batch size
__C.DATASET.NO_PREFECHTER = False
__C.DATASET.PIN_MEM = True
__C.DATASET.VAL_BATCH_MUL = 4
# model configs
__C.NET = CN()
# __C.NET.SELECTION = 14
# __C.NET.GP = 'avg'  # type of global pool ["avg", "max", "avgmax", "avgmaxc"]
# __C.NET.DROPOUT_RATE = 0.0  # dropout rate
# model ema parameters
__C.NET.EMA = CN()
__C.NET.EMA.USE = True
__C.NET.EMA.FORCE_CPU = False  # force model ema to be tracked on CPU
__C.NET.EMA.DECAY = 0.9998
__C.SEARCH_RESOLUTION = 640
__C.SEARCH_SPACE = CN()
__C.SEARCH_SPACE.BOTTLENECK_CSP = 0
__C.SEARCH_SPACE.BOTTLENECK_CSP2 = 0
# optimizer configs
__C.OPT = 'sgd'
__C.OPT_EPS = 1e-2
__C.MOMENTUM = 0.9
__C.WEIGHT_DECAY = 1e-4
__C.THETA_OPTIMIZER = CN()
__C.THETA_OPTIMIZER.OPT = 'adam'
__C.THETA_OPTIMIZER.OPT_EPS = 1e-2 # eps for adam
__C.THETA_OPTIMIZER.LR = 1e-4
__C.THETA_OPTIMIZER.MOMENTUM = 0.9
__C.THETA_OPTIMIZER.WEIGHT_DECAY = 1e-4
# scheduler configs
__C.SCHED = 'sgd'
__C.LR_NOISE_PCT = 0.67
__C.LR_NOISE_STD = 1.0
__C.WARMUP_LR = 1e-4
__C.MIN_LR = 1e-5
__C.EPOCHS = 200
__C.START_EPOCH = None
__C.DECAY_EPOCHS = 30.0
__C.COOLDOWN_EPOCHS = 10
__C.PATIENCE_EPOCHS = 10
__C.DECAY_RATE = 0.1
__C.LR = 1e-2
__C.LR_NOISE = None
__C.META_LR = 1e-4
__C.FREEZE_EPOCH = 40
__C.TEMPERATURE = CN()
__C.TEMPERATURE.INIT= 5.0
__C.TEMPERATURE.FINAL=0.1
__C.WARMUP_EPOCH = 100000
__C.CONVERGE_EPOCH = -2
__C.UPDATE_METHOD='ver1'
# data augmentation parameters
# __C.AUGMENTATION = CN()

    

#########################################
# IZDNAS Configuration
#########################################
__C.STAGE2 = CN()
__C.STAGE2.EPOCHS = 40
__C.STAGE2.ITERATIONS = 2880
__C.STAGE2.ZCMAP_UPDATE_ITER = 50
__C.STAGE2.ALPHA = 0.005
__C.STAGE2.GAMMA =  0.03
__C.STAGE2.THETA = 0.005
__C.STAGE2.OMEGA =  0.03
__C.STAGE2.HARDWARE_FREEZE_EPOCHS = 0
__C.STAGE2.TEMPERATURE = CN()
__C.STAGE2.TEMPERATURE.INIT  = 5.0
__C.STAGE2.TEMPERATURE.FINAL = 5.0


    

          
    
    
  
# __C.AUGMENTATION.AA = 'rand-m9-mstd0.5'
# __C.AUGMENTATION.COLOR_JITTER = 0.4
# __C.AUGMENTATION.RE_PROB = 0.2  # random erase prob
# __C.AUGMENTATION.RE_MODE = 'pixel'  # random erase mode
# __C.AUGMENTATION.MIXUP = 0.0  # mixup alpha
# __C.AUGMENTATION.MIXUP_OFF_EPOCH = 0  # turn off mixup after this epoch
# __C.AUGMENTATION.SMOOTHING = 0.1  # label smoothing parameters
# batch norm parameters (only works with gen_efficientnet based models
# currently)
__C.BATCHNORM = CN()
__C.BATCHNORM.SYNC_BN = False
__C.BATCHNORM.BN_TF = False
__C.BATCHNORM.BN_MOMENTUM = 0.1  # batchnorm momentum override
__C.BATCHNORM.BN_EPS = 1e-5  # batchnorm eps override
# supernet training hyperparameters
# __C.SUPERNET = CN()
# __C.SUPERNET.UPDATE_ITER = 1300
# __C.SUPERNET.SLICE = 4
# __C.SUPERNET.POOL_SIZE = 10
# __C.SUPERNET.RESUNIT = False
# __C.SUPERNET.DIL_CONV = False
# __C.SUPERNET.UPDATE_2ND = True
# __C.SUPERNET.FLOPS_MAXIMUM = 600
# __C.SUPERNET.FLOPS_MINIMUM = 0
# __C.SUPERNET.PICK_METHOD = 'meta'  # pick teacher method
# __C.SUPERNET.META_STA_EPOCH = 20  # start using meta picking method
# __C.SUPERNET.HOW_TO_PROB = 'pre_prob'  # sample method
# __C.SUPERNET.PRE_PROB = (0.05, 0.2, 0.05, 0.5, 0.05, 0.15)  # sample prob in 'pre_prob'
# Experiment Setting
__C.BETA_REG = False