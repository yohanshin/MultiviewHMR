from models.mlmc import build_model
from models.smpl import build_body_model
from models.smplify.smplify_3d import build_smplify3d
from data.h36m import setup_human36m_dataloaders
from data.mpi_inf import setup_mpi_dataloaders
from data.mix_dataset import setup_mixed_dataloaders
from data.data_utils import prepare_batch
from utils.cmd import get_cfg, get_cmd
from utils.optim import build_optimizer, build_scheduler
from utils.loss import build_loss_function
from utils.checkpoint import build_logger
from utils.distributed_learning import init_distributed

import torch
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import matplotlib.pyplot as plt

import argparse
from tqdm import tqdm, trange
from os import path as osp


def eval_model(curr_epoch, val_dataloader, 
               model, body_model, prediction,
               loss_function, logger, 
               batch_size, device, dtype, data_name='h36m',
               mpjpe_list=[], recone_list=[], 
               mpjpe=0, recone=0, val_iter=1):

    print('Evaluate the model ... ')

    iterator = enumerate(val_dataloader)
    pose2rot = False if prediction == 'rotmat' else True
    
    model.eval()
    with torch.no_grad():
        for t in range(len(val_dataloader)):
            _, batch = next(iterator)
            images, keypoints_3d_gt, proj_matricies = prepare_batch(batch, device)
            
            if images.shape[0] != batch_size:
                # In this case data batch size is not compatible with SMPL model batch size
                continue
            
            pred_pose, pred_betas, pred_global_orient, pred_vertices \
                = model(images, proj_matricies, batch)

            pred_output = body_model(betas=pred_betas, 
                                     body_pose=pred_pose, 
                                     global_orient=pred_global_orient, 
                                     pose2rot=pose2rot)

            _mpjpe, _recone = loss_function.eval(pred_output, keypoints_3d_gt)
            mpjpe = mpjpe + _mpjpe
            recone = recone + _recone
            mean_mpjpe = mpjpe / val_iter
            mean_recone = recone / val_iter
            val_iter = val_iter + 1
    
    logger({'%s Val MPJPE'%data_name: mean_mpjpe, '%s Val RECONE'%data_name: mean_recone}, curr_epoch, is_val=True)
    model.train()

    print('Evaluate results: MPJPE: %.3f mm  |  RECONE: %.3f mm'%(mean_mpjpe, mean_recone))

    return mean_mpjpe, mean_recone


def train_one_epoch(curr_iter, curr_epoch, train_dataloader, 
                    model, body_model, prediction,
                    optimizer, loss_function, 
                    smplify, weakly_supervise, use_precal_smplify,
                    logger, batch_size, device, dtype):

    iterator = enumerate(train_dataloader)
    pose2rot = False if prediction == 'rotmat' else True
    
    with torch.autograd.enable_grad():
        with torch.autograd.set_detect_anomaly(True):
            for i_iter in range(len(train_dataloader)):
                _, batch = next(iterator)
                images, keypoints_3d_gt, proj_matricies = prepare_batch(batch, device)
                
                if images.shape[0] != batch_size:
                    # In this case data batch size is not compatible 
                    # with SMPL model batch size
                    if weakly_supervise:
                        continue
                
                pred_pose, pred_betas, pred_global_orient, pred_vertices \
                    = model(images, proj_matricies, batch)

                pred_output = body_model(betas=pred_betas, 
                                         body_pose=pred_pose, 
                                         global_orient=pred_global_orient, 
                                         pose2rot=pose2rot)

                opt_pose = torch.from_numpy(batch['smplify_pose']).to(device=device, dtype=dtype)
                opt_betas = torch.from_numpy(batch['smplify_shape']).to(device=device, dtype=dtype)
                opt_global_orient = torch.from_numpy(batch['smplify_global_orient']).to(device=device, dtype=dtype)

                opt_output = body_model(betas=opt_betas, body_pose=opt_pose, global_orient=opt_global_orient, 
                                        pose2rot=True)

                print_log = ((curr_iter + 1) % len(train_dataloader)) % logger.write_freq == 0
                loss, loss_dict = loss_function(pred_output, keypoints_3d_gt, proj_matricies, 
                                                opt_output=opt_output, print_log=print_log, has_smpl=batch['has_smpl'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                curr_iter = curr_iter + 1
                logger(loss_dict, curr_iter)

    return curr_iter


def main(cfg, args, curr_iter=0, start_epoch=0, last_statedict_pth=None):
    
    if cfg.TRAIN.DISTRIBUTED:
        is_distributed = init_distributed(args)
        if is_distributed:
            device = torch.device(args.local_rank)
        else:
            device = torch.device(0)

    else:
        is_distributed = False
        device = cfg.DEVICE
    
    dtype = torch.float32
    batch_size = cfg.TRAIN.BATCH_SIZE
    
    prediction = cfg.MODEL.VHMR.PREDICTION if cfg.MODEL.META_ARCH == 'VHMR' else 'rotmat'
    weakly_supervise = cfg.TRAIN.WEAKLY_SUPERVISE
    precal_criterion = cfg.TRAIN.USE_PRECAL_BEFORE

    # Build SMPL model
    body_model = build_body_model(cfg)

    # Build neural network model
    model = build_model(cfg)

    # Build SMPLify-3D model
    smplify = build_smplify3d(cfg)

    # Build loss function
    loss_function = build_loss_function(cfg)

    # Build optimizer
    optimizer = build_optimizer(cfg, model)

    # Build dataloader
    if cfg.DATASET.TYPE == 'human36m':
        train_dataloader, val_dataloader = setup_human36m_dataloaders(cfg)
    elif cfg.DATASET.TYPE == 'mpi-inf-3dhp':
        train_dataloader, val_dataloader = setup_mpi_dataloaders(cfg)
    else:
        train_dataloader, val_h36m_dataloader, val_mpi_dataloader = setup_mixed_dataloaders(cfg)

    # Build logger
    logger = build_logger(cfg, len(train_dataloader))
    
    # Multi-GPU setting
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device])
        body_model = DistributedDataParallel(body_model, device_ids=[device])

    # Initialize weight if resume training
    if osp.exists(cfg.TRAIN.INIT_WEIGHT):
        model.load_state_dict(torch.load(cfg.TRAIN.INIT_WEIGHT)['state_dict'])
        optimizer.load_state_dict(torch.load(cfg.TRAIN.INIT_WEIGHT)['optimizer'])
        curr_iter = torch.load(cfg.TRAIN.INIT_WEIGHT)['iteration']
        start_epoch = torch.load(cfg.TRAIN.INIT_WEIGHT)['epoch']
        print("Pretrained model loaded ... Resume training from epoch {}".format(start_epoch))
    
    if cfg.TRAIN.SCHEDULER:
        optimizer = build_scheduler(cfg, optimizer, curr_iter, len(train_dataloader))
        
    print("Start training {} ...".format(cfg.TRAIN.NAME))
    for epoch in range(start_epoch, cfg.TRAIN.EPOCH):
        use_precal_smplify = precal_criterion > epoch

        curr_iter = train_one_epoch(curr_iter, epoch, train_dataloader,
                                    model, body_model, prediction, optimizer, loss_function, 
                                    smplify, weakly_supervise, use_precal_smplify, 
                                    logger, batch_size, device, dtype)

        if cfg.DATASET.TYPE in ['human36m', 'mpi-inf-3dhp']:
            mpjpe, recone = eval_model(epoch, val_dataloader, 
                                    model, body_model, prediction, 
                                    loss_function,  logger, 
                                    batch_size, device, dtype)

            logger.save_checkpoint(model, optimizer, epoch, curr_iter, mpjpe, recone)
        
        else:
            h36m_mpjpe, h36m_recone = eval_model(epoch, val_h36m_dataloader, 
                                    model, body_model, prediction, 
                                    loss_function,  logger, 
                                    batch_size, device, dtype, data_name='h36m')

            mpi_mpjpe, mpi_recone = eval_model(epoch, val_mpi_dataloader, 
                                    model, body_model, prediction, 
                                    loss_function,  logger, 
                                    batch_size, device, dtype, data_name='mpi-inf')

            logger.save_checkpoint(model, optimizer, epoch, curr_iter, h36m_mpjpe, h36m_recone, mpi_mpjpe, mpi_recone)
        
    
    
if __name__ == '__main__':
    args = get_cmd().parse_args()
    cfg, args = get_cfg(args)

    main(cfg, args)