from models.mlmc import build_model
from models.smpl import build_body_model
from models.smplify.smplify_3d import build_smplify3d

from data.h36m import setup_human36m_dataloaders
from data.new_h36m import setup_new_human36m_dataloaders
from data.mpi_inf import setup_mpi_dataloaders
from data.mix_dataset import setup_mixed_dataloaders
from data.data_utils import prepare_batch
from utils.cmd import get_cfg, get_cmd
from utils.optim import build_optimizer, build_scheduler
from utils.loss import build_loss_function
from utils.checkpoint import build_logger
from utils.distributed_learning import init_distributed
from utils.visualization import *
from utils.pose_utils import *

import torch
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import matplotlib.pyplot as plt
import cv2

import argparse
from tqdm import tqdm, trange
from os import path as osp
import time


from models.spin import hmr


def compute_error(pred_output, keypoints_3d_gt, J_regressor_):

    def align_two_joints(gt_joints, pred_joints, opt_joints=None):
        gt_pelvis = gt_joints[:, 14]
        pred_pelvis = pred_joints[:, 14]
        gt_joints = gt_joints - gt_pelvis.unsqueeze(1)
        pred_joints = pred_joints - pred_pelvis.unsqueeze(1)

        return pred_joints, gt_joints

    num_joints = 14
    batch_size = keypoints_3d_gt.shape[0]
    J_regressor = J_regressor_[None, :].expand(keypoints_3d_gt.shape[0], -1, -1).to(keypoints_3d_gt.device)

    keypoints_3d_pred = torch.matmul(J_regressor, pred_output.vertices)
    keypoints_3d_pred = keypoints_3d_pred[:, constants.H36M_TO_J17, :] * 1e3
    keypoints_3d_gt = keypoints_3d_gt

    keypoints_3d_gt, keypoints_3d_pred = align_two_joints(keypoints_3d_gt, keypoints_3d_pred)

    keypoints_3d_gt = keypoints_3d_gt.detach().cpu().numpy()[:, :num_joints]
    keypoints_3d_pred = keypoints_3d_pred.detach().cpu().numpy()[:, :num_joints]

    mpjpes = []
    pa_mpjpes = []
    pcks = []
    pa_pcks = []
    aucs = []
    pa_aucs = []

    for i, (gt, pred) in enumerate(zip(keypoints_3d_gt, keypoints_3d_pred)):
        auc = 0
        
        jpe  = np.sqrt(((gt - pred)**2).sum(-1))
        mpjpes += [jpe.mean()]
        for k in range(1, 151, 5):
            pck = np.sum(jpe <= k) / 14.0
            auc += pck / 30.0

        pcks += [pck]
        aucs += [auc]

        pa_pred = compute_similarity_transform(pred, gt)
        pa_jpe = np.sqrt(((gt - pa_pred)**2).sum(-1))
        pa_mpjpes += [pa_jpe.mean()]
        pa_auc = 0
        for k in range(1, 151, 5):
            pa_pck = np.sum(pa_jpe <= k) / 14.0
            pa_auc += pa_pck / 30.0
        
        pa_pcks += [pa_pck]
        pa_aucs += [pa_auc]

    mpjpe = np.array(mpjpes).mean()
    pa_mpjpe = np.array(pa_mpjpes).mean()
    pck = np.array(pcks).mean()
    pa_pck = np.array(pa_pcks).mean()
    auc = np.array(aucs).mean()
    pa_auc = np.array(pa_aucs).mean()

    return mpjpe, pa_mpjpe, pck, pa_pck, auc, pa_auc


def main(cfg, args, val_iter=1):
    render = True
    output_fldr = args.output

    batch_size = cfg.TRAIN.BATCH_SIZE

    # Build neural network model
    model = build_model(cfg)

    # Initialize weight if resume training
    if osp.exists(cfg.TRAIN.INIT_WEIGHT):
        model.load_state_dict(torch.load(cfg.TRAIN.INIT_WEIGHT)['state_dict'])
        print("Evaluating model loaded ...")

    model.eval()
    
    # Model Summary
    total_params = sum(p.numel() for p in model.parameters())
    print('Total number of parameters is {}'.format(total_params))

    # Build SMPL model
    body_model = build_body_model(cfg, render=render)

    loss_function = build_loss_function(cfg)
    J_regressor = torch.from_numpy(np.load(constants.JOINT_REGRESSOR_H36M)).float()

    # Build dataloader
    if cfg.DATASET.TYPE == 'human36m':
        val_dloader = setup_new_human36m_dataloaders(cfg)
        # val_dloader = setup_human36m_dataloaders(cfg)
    else:
        val_dloader = setup_mpi_dataloaders(cfg)

    print('==> Data loaded...')
    
    mpjpe, pa_mpjpe, pck, pa_pck, auc, pa_auc = 0, 0, 0, 0, 0, 0
    iterator = enumerate(val_dloader)
    with torch.no_grad():
        with tqdm(total=len(val_dloader)) as prog_bar:
            for t in range(len(val_dloader)):
                _, batch = next(iterator)

                if batch['frame'][0] > 1500 or batch['frame'][0] < 1000:
                    continue

                images, keypoints_3d_gt, proj_matricies = prepare_batch(batch, cfg.DEVICE)

                if images.shape[0] != batch_size:
                    # In this case data batch size is not compatible with SMPL model batch size
                    continue

                # images = images[:, 0].expand(1, 4, 3, 224, 224)
                # proj_matricies = proj_matricies[:, 0].expand(1, 4, 3, 4)
                
                pred_pose, pred_betas, pred_global_orient, pred_vertices \
                    = model(images, proj_matricies, batch)

                pred_output = body_model(betas=pred_betas, 
                                        body_pose=pred_pose, 
                                        global_orient=pred_global_orient, 
                                        pose2rot=False)

                if cfg.DATASET.TYPE == 'mpi-inf-3dhp':
                    # For Evaluating MPI-INF-3DHP dataset, PCK / AUC are the another error metric
                    _mpjpe, _pa_mpjpe, _pck, _pa_pck, _auc, _pa_auc = compute_error(pred_output, keypoints_3d_gt, J_regressor)
                    pck += _pck
                    pa_pck += _pa_pck
                    auc += _auc
                    pa_auc += _pa_auc

                else:
                    _mpjpe, _pa_mpjpe = loss_function.eval(pred_output, keypoints_3d_gt)
                
                mpjpe = mpjpe + _mpjpe
                pa_mpjpe = pa_mpjpe + _pa_mpjpe
                mean_mpjpe = mpjpe / val_iter
                mean_pa_mpjpe = pa_mpjpe / val_iter

                val_iter = val_iter + 1

                msg = 'MPJPE = %.3f,   PA MPJPE = %.3f'%(mean_mpjpe, mean_pa_mpjpe)
                prog_bar.set_postfix_str(msg)
                prog_bar.update(1)
                prog_bar.refresh()
                
                if render:
                    # generate_figure(batch['org_cameras'], pred_output, body_model, 
                    #                 batch['org_images'], keypoints_3d_gt, iters=t)

                    tmp_generate_figure(batch['org_cameras'], pred_output, body_model, batch['org_images'], 
                                        keypoints_3d_gt, batch['action'][0], batch['frame'][0])

                # opt_pose = torch.from_numpy(batch['smplify_pose']).to(device=pred_pose.device)
                # opt_betas = torch.from_numpy(batch['smplify_shape']).to(device=pred_pose.device)
                # opt_global_orient = torch.from_numpy(batch['smplify_global_orient']).to(device=pred_pose.device)
                
                # opt_output = body_model(betas=opt_betas, body_pose=opt_pose, global_orient=opt_global_orient, 
                #                         pose2rot=True)
                
                # if render:
                #     generate_figure(batch['org_cameras'], opt_output, body_model, 
                #                     batch['org_images'], keypoints_3d_gt, iters=t)

                # import pdb; pdb.set_trace()

    print('Evaluate results: MPJPE: %.3f mm  |  RECONE: %.3f mm'%(mean_mpjpe, mean_pa_mpjpe))
    if cfg.DATASET.TYPE == 'mpi-inf-3dhp':
        print('PCK : %.1f | AUC : %.1f |   PCK (PA) : %.1f | AUC (PA) : %.1f'%(
            pck / val_iter * 100, auc / val_iter * 100, pa_pck / val_iter * 100, pa_auc / val_iter * 100))
    

if __name__ == '__main__':
    args = get_cmd().parse_args()
    cfg, args = get_cfg(args)

    main(cfg, args)