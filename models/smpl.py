from cfg import constants

import torch
import numpy as np
import smplx
from smplx import SMPL as _SMPL
from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints

import os.path as osp

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, joint_regressor, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(joint_regressor)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output

    def get_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 38, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor])
        joints_extra = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor_extra])
        joints = torch.cat((joints, joints_extra), dim=1)

        return joints

def build_body_model(cfg, batch_size=None, render=False):
    gender = 'male' if render else 'neutral'
    
    device = cfg.DEVICE
    if batch_size is None:
        batch_size = cfg.TRAIN.BATCH_SIZE
    body_model_folder = osp.join(cfg.SMPL.ROOT_PTH, cfg.SMPL.TYPE)
    
    body_model = SMPL(cfg.SMPL.JOINT_REGRESSOR_TRAIN_EXTRA,
                      body_model_folder,
                      gender=gender,
                      batch_size=batch_size,
                      create_transl=False).to(device)

    return body_model