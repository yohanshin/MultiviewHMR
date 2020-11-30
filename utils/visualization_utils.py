from cfg import constants

import torch
import numpy as np

import trimesh
import pyrender
import neural_renderer as nr
import cv2

import os

CONNECTIVITY_DICT = {
    'cmu': [(0, 2), (0, 9), (1, 0), (1, 17), (2, 12), (3, 0), (4, 3), (5, 4), (6, 2), (7, 6), (8, 7), (9, 10), (10, 11), (12, 13), (13, 14), (15, 1), (16, 15), (17, 18)],
    'coco': [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (5, 11), (6, 12), (11, 12)],
    # "mpii": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 9), (8, 12), (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)],
    "human36m": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 16), (9, 16), (8, 12), (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)],
    "kth": [(0, 1), (1, 2), (5, 4), (4, 3), (6, 7), (7, 8), (11, 10), (10, 9), (2, 3), (3, 9), (2, 8), (9, 12), (8, 12), (12, 13)],
    # "mpii": [(0,1), (1,2), (5,4), (4,3), (2,14), (3,14), (14,16), (16,12), (12,17), (17,18), (6,7), (7,8), (11,10), (10,9), (8,12), (9,12)
    "mpii": [(0,1), (1,2), (5,4), (4,3), (2,14), (3,14), (14,15), (15,12), (12,16), (16,13), (6,7), (7,8), (11,10), (10,9), (8,12), (9,12)]
}

COLOR_DICT = {
    'coco': [
        (102, 0, 153), (153, 0, 102), (51, 0, 153), (153, 0, 153),  # head
        (51, 153, 0), (0, 153, 0),  # left arm
        (153, 102, 0), (153, 153, 0),  # right arm
        (0, 51, 153), (0, 0, 153),  # left leg
        (0, 153, 102), (0, 153, 153),  # right leg
        (153, 0, 0), (153, 0, 0), (153, 0, 0), (153, 0, 0)  # body
    ],

    'human36m': [
        (0, 153, 102), (0, 153, 153), (0, 153, 153),  # right leg
        (0, 51, 153), (0, 0, 153), (0, 0, 153),  # left leg
        (153, 0, 0), (153, 0, 0),  # body
        (153, 0, 102), (153, 0, 102),  # head
        (153, 153, 0), (153, 153, 0), (153, 102, 0),   # right arm
        (0, 153, 0), (0, 153, 0), (51, 153, 0)   # left arm
    ],

    'mpii': [
        (0, 153, 102), (0, 153, 153), (0, 153, 153),  # right leg
        (0, 51, 153), (0, 0, 153), (0, 0, 153),  # left leg
        (153, 0, 0), (153, 0, 0),  # body
        (153, 0, 102),  # head
        (153, 153, 0), (153, 153, 0), (153, 102, 0),   # right arm
        (0, 153, 0), (0, 153, 0), (51, 153, 0)   # left arm
    ],

    'kth': [
        (0, 153, 102), (0, 153, 153),  # right leg
        (0, 51, 153), (0, 0, 153),  # left leg
        (153, 102, 0), (153, 153, 0),  # right arm
        (51, 153, 0), (0, 153, 0),  # left arm
        (153, 0, 0), (153, 0, 0), (153, 0, 0), (153, 0, 0), (153, 0, 0), # body
        (102, 0, 153) # head
    ]
}

JOINT_NAMES_DICT = {
    'coco': {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
    'human36m': {
        0: 'Right Ankle',
        1: 'Right Knee',
        2: 'Right Hip',
        3: 'Left Hip',
        4: 'Left Knee',
        5: 'Left Ankle',
        6: 'Sacrum',
        7: 'Spine',
        8: 'Neck',
        9: 'Head',
        10: 'Right Wrist',
        11: 'Right Elbow',
        12: 'Right Shoulder',
        13: 'Left Shoulder',
        14: 'Left Elbow',
        15: 'Left Wrist'
    }
}


def align_two_joints(gt_joints, pred_joints, opt_joints=None, data_type='human36m'):
    
    def centering_joints(joints):
        if data_type == 'human36m':
            sacrum_center = joints[:, 14]
        elif data_type == 'cmu':
            lpelvis, rpelvis = joints[:, 6].clone(), joints[:, 12].clone()
            sacrum_center = (lpelvis + rpelvis)/2
        
        joints_ = joints - sacrum_center.unsqueeze(1)

        return joints_

    
    gt_joints = gt_joints[:, :, [0, 2, 1]]
    gt_joints[:, :, 1] = -gt_joints[:, :, 1]
    gt_joints = centering_joints(gt_joints)
    
    flip = torch.tensor([1, 1, 1], 
        device=pred_joints.device, dtype=pred_joints.dtype)
    pred_joints = pred_joints * flip
    pred_joints = centering_joints(pred_joints)

    if opt_joints is not None:
        flip = torch.tensor([1, 1, 1], 
            device=opt_joints.device, dtype=opt_joints.dtype)
        opt_joints = opt_joints * flip
        opt_joints = centering_joints(opt_joints)

        return gt_joints, pred_joints, opt_joints

    return gt_joints, pred_joints


def get_segment_idx():

    segments = dict()
    segments['larm1'] = [13, 14]
    segments['larm2'] = [14, 15]
    segments['rarm1'] = [10, 11]
    segments['rarm2'] = [11, 12]
    segments['lleg1'] = [3, 4]
    segments['lleg2'] = [4, 5]
    segments['rleg1'] = [0, 1]
    segments['rleg2'] = [1, 2]
    segments['back1'] = [6, 7]
    segments['back2'] = [7, 8]

    return segments


def set_range(ax, offset=[0, 0, 0]):
    x, y, z = offset
    ax.set_xlim(-65 + x, 65 + x)
    ax.set_ylim(-65 + y, 65 + y)
    ax.set_zlim(-65 + z, 65 + z)
    

def project2D_by_camera_info(x3d, K, R, t, dist=None):
    """
    x3d : B X N X 4 numpy array
    K : 3 X 3 numpy array
    R : 3 X 3 numpy array
    t : 3 X 1 numpy array
    dist : 5 X 1 numpy array
    """

    conf_mask = np.ones((x3d.shape[0], x3d.shape[1]))

    x2d = np.zeros_like(x3d[:, :, :2])
    R2_criterion = np.zeros_like(x3d[:, :, 0])

    for J in range(x3d.shape[1]):
        """ J is joint index """
        
        x = np.dot(R, x3d[:, J].T) + t
        xp = x[:2] / x[2]

        if dist is not None:
            X2 = xp[0] * xp[0]
            Y2 = xp[1] * xp[1]
            XY = X2 * Y2
            R2 = X2 + Y2
            R4 = R2 * R2
            R6 = R4 * R2
            R2_criterion[:, J] = R2

            radial = 1.0 + dist[0] * R2 + dist[1] * R4 + dist[4] * R6
            tan_x = 2.0 * dist[2] * XY + dist[3] * (R2 + 2.0 * X2)
            tan_y = 2.0 * dist[3] * XY + dist[2] * (R2 + 2.0 * Y2)

            xp[0, :] = radial * xp[0, :] + tan_x
            xp[1, :] = radial * xp[1, :] + tan_y

        pt = np.dot(K[:2, :2], xp) + K[:2, 2:]
        x2d[:, J, :] = pt.T
    
    x2d = x2d.astype('int32')
    x_visible = np.logical_and(x2d[:, :, 0] >= 0, x2d[:, :, 0] < image.shape[0])
    y_visible = np.logical_and(x2d[:, :, 1] >= 0, x2d[:, :, 1] < image.shape[1])
    visible = np.logical_and(x_visible, y_visible)
    vis_mask = np.logical_and(visible, R2_criterion < 1.)
    mask = np.logical_and(conf_mask, vis_mask)

    return x2d, mask

def project2D_by_proj_matrix(x3d, proj_matrix):
    """
    x3d : B X N X 4 numpy array
    proj_matrix : 3 X 4 numpy array
    """

    conf_mask = np.ones((x3d.shape[0], x3d.shape[1]))
    R2_criterion = np.zeros_like(x3d[:, :, 0])
    x3d = x3d[0]

    x3d_hom = np.concatenate((x3d, np.ones_like(x3d[:, :1])), axis=-1)
    x2d_hom = x3d_hom @ proj_matrix.T
    x2d = (x2d_hom.T[:-1] / x2d_hom.T[-1]).T

    x2d = x2d[None].astype('int32')
    x_visible = np.logical_and(x2d[:, :, 0] >= 0, x2d[:, :, 0] < image.shape[0])
    y_visible = np.logical_and(x2d[:, :, 1] >= 0, x2d[:, :, 1] < image.shape[1])
    visible = np.logical_and(x_visible, y_visible)
    
    vis_mask = np.logical_and(visible, R2_criterion < 1.)
    mask = np.logical_and(conf_mask, vis_mask)

    return x2d, mask

def project2D_SPIN(x3d, K, R, t):
    """
    x3d : B X N X 4 numpy array
    K : 3 X 3 numpy array
    R : 3 X 3 numpy array
    t : 3 X 1 numpy array
    """

    conf_mask = np.ones_like(x3d[:, :, 0])
    R2_criterion = np.zeros_like(conf_mask)
    # Transform points
    x2d_hom = np.einsum('bij,bkj->bki', R[None], x3d)
    x2d_hom = x2d_hom + t.T

    # Apply perspective distortion
    x2d_hom = x2d_hom / x2d_hom[:, :, -1:]

    # Apply camera intrinsics
    x2d_hom = np.einsum('bij,bkj->bki', K[None], x2d_hom)
    x2d = x2d_hom[:, :, :-1]

    x2d = x2d.astype('int32')
    x_visible = np.logical_and(x2d[:, :, 0] >= 0, x2d[:, :, 0] < image.shape[0])
    y_visible = np.logical_and(x2d[:, :, 1] >= 0, x2d[:, :, 1] < image.shape[1])
    visible = np.logical_and(x_visible, y_visible)
    
    vis_mask = np.logical_and(visible, R2_criterion < 1.)
    mask = np.logical_and(conf_mask, vis_mask)

    return x2d, mask