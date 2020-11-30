from scipy.io import loadmat
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import sys
sys.path[-1] = '/home/soyongs/research/codes/MultiViewMoCap/'
from utils.visualization import *

from os import path as osp


def world_to_image(P3D, R, t, K):
    """ Project global 3D points to image plane
    """
    extrinsics = np.hstack([R, t])
    proj_matrix = K[:3, :3].dot(extrinsics)
    
    P3D_hom = np.hstack([P3D, np.ones((len(P3D), 1))])
    P2D_hom = P3D_hom @ proj_matrix.T
    P2D = (P2D_hom.T[:-1] / P2D_hom.T[-1]).T
    
    return P2D


def cam_to_image(P3D, K):
    """ Project local 3D points to image plane
    """
    P3D_hom = np.hstack([P3D, np.ones((len(P3D), 1))])
    P2D_hom = K[:3] @ P3D_hom
    P2D = (P2D_hom.T[:-1] / P2D_hom.T[-1]).T
    
    return P2D


def cam_to_world(loc3D, R, T):
    """ Convert local 3D points to global coordinate
    """
    glob3D = R.T.dot(loc3D.T - T[:, None])

    return glob3D.T


def world_to_cam(glob3d, R, t):
    """ Convert global 3D points to local coordinate
    """
    loc3d = R @ glob3d.T + t[:, None]
    
    return loc3d.T


def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []

    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i*7+5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i*7+6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3,:3]
        T = RT[:3,3]
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)

    file.close()
    return Ks, Rs, Ts

def process_train_data(extract_image=False, generate_label=True):

    J28_TO_J17 = [25, 24, 23, 18, 19, 20, 16, 15, 14, 9, 10, 11, 5, 7, 4, 3, 6]
    
    # J28_TO_J17 = [4, 18, 19, 20, 23, 24, 25, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16]

    subject_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    seq_list = ['Seq1', 'Seq2']
    video_list = list(range(3)) + list(range(4, 9))

    output_file = {'subject_names': subject_list, 
                   'camera_names': ['0', '1', '2', '4', '5', '6', '7', '8'],
                   'sequence_names': seq_list}

    output_file['table'] = []
    table_dtype = np.dtype([
        ('subject_idx', np.int8),
        ('frame_idx', np.int16),
        ('sequence_idx', np.int16),
        ('keypoints_2d', np.float32, (8, 17, 2)),
        ('keypoints', np.float32, (17, 3)),
        ('bbox_by_camera_tlbr', np.int16, (len(video_list), 4)),
        ('R', np.float32, (len(video_list),3,3)),
        ('t', np.float32, (len(video_list),3,1)),
        ('K', np.float32, (len(video_list),3,3))])


    for subject_idx, subject in enumerate(subject_list):
        for seq_idx, seq in enumerate(seq_list):
            annot_file = osp.join(subject, seq, 'annot.mat')
            annot2d = loadmat(annot_file)['annot2']
            annot3d = loadmat(annot_file)['annot3']

            calib_file = osp.join(subject, seq, 'camera.calibration')
            Ks, Rs, Ts = read_calibration(calib_file, video_list)
            video_fldr = osp.join(subject, seq, 'imageSequence')

            keypoints_2d_list, keypoints_3d_univ_list, bboxes_list, valid_frame_list = [], [], [], []
            for cam_idx, video in enumerate(video_list):
                video_file = osp.join(video_fldr, 'video_{}.avi'.format(video))
                image_fldr = osp.join(video_fldr, 'image_{}'.format(video))

                if extract_image:
                    if osp.exists(image_fldr):
                        continue
                    os.makedirs(image_fldr+'_tmp')
                    os.makedirs(image_fldr)
                    command = 'ffmpeg -i %s -qscale:v 1 %s/frame'%(video_file, image_fldr+'_tmp')+'_%06d.jpg'
                    print('command line: {}'.format(command))
                    os.system(command)

                    prev_files1 = image_fldr + '_tmp/*0.jpg'
                    prev_files2 = image_fldr + '_tmp/*5.jpg'
                    target_pth = image_fldr + '/'
                    command = 'mv %s %s'%(prev_files1, target_pth)
                    os.system(command)
                    command = 'mv %s %s'%(prev_files2, target_pth)
                    os.system(command)
                    command = 'rm -rf %s'%(image_fldr + '_tmp')
                    os.system(command)

                if generate_label:
                    _, _, image_list = next(os.walk(image_fldr))
                    image_list.sort()
                    keypoints_2d, keypoints_3d, frame_idxs, bboxes, all_visible = [], [], [], [], []
                    for img_i in image_list:
                        img_idx = int(img_i[-10:-4]) - 1
                        S17_2d = annot2d[video][0][img_idx].reshape(28, 2)[J28_TO_J17]
                        S17_3d = annot3d[video][0][img_idx].reshape(28, 3)[J28_TO_J17]
                        
                        # Previous method
                        S17_3d_univ = cam_to_world(S17_3d, Rs[cam_idx], Ts[cam_idx])
                        
                        bbox = [min(S17_2d[:, 0]), min(S17_2d[:, 1]),
                                max(S17_2d[:, 0]), max(S17_2d[:, 1])]
                                                
                        x_in = np.logical_and(S17_2d[:, 0] <= 2048, S17_2d[:, 0] >= 0)
                        y_in = np.logical_and(S17_2d[:, 1] <= 2048, S17_2d[:, 1] >= 0)
                        all_joints_visible = np.logical_and(x_in, y_in)
                        
                        all_joints_visible = all_joints_visible.sum() == all_joints_visible.shape[0]
                        
                        bbox = np.array(bbox).astype('int16')
                        bboxes.append(bbox[None, None])
                        world_to_camera_coord(S17_3d_univ, Rs[cam_idx], Ts[cam_idx])
                        keypoints_2d.append(S17_2d[None])
                        keypoints_3d.append(S17_3d_univ[None])
                        frame_idxs.append(img_i[-10:-4])
                        all_visible.append(all_joints_visible)
                    
                    keypoints_2d = np.concatenate(keypoints_2d)
                    keypoints_3d_univ_list.append(np.concatenate(keypoints_3d))
                    
                    keypoints_2d_list.append(keypoints_2d)
                    bboxes_list.append(np.concatenate(bboxes))
                    valid_frame_list.append(np.array(all_visible)[None])

            valid_frames = np.concatenate(valid_frame_list).prod(axis=0).astype('bool')
            keypoints_2d = np.concatenate([k2d[:, None] for k2d in keypoints_2d_list], axis=1)
            keypoints_2d = keypoints_2d[valid_frames].astype('int')
            keypoints_3d_univ = np.concatenate([k3d_univ[:, None] for k3d_univ in keypoints_3d_univ_list], axis=1)

            if keypoints_3d_univ.std(axis=1).max() > 5.0:
                import pdb; pdb.set_trace()

            keypoints_3d_univ = keypoints_3d_univ.mean(axis=1)[valid_frames]
            bboxes = np.concatenate(bboxes_list, axis=1)[valid_frames]
            
            Ks = np.repeat(np.array(Ks)[None], len(bboxes), axis=0)[:, :, :3, :3]            
            Rs = np.repeat(np.array(Rs)[None], len(bboxes), axis=0)
            ts = np.repeat(np.array(np.array(ts))[None], len(bboxes), axis=0)[:, :, :, None]

            table_segment = np.empty(valid_frames.sum(), dtype=table_dtype)
            table_segment['keypoints_2d'] = keypoints_2d
            table_segment['keypoints'] = keypoints_3d_univ
            table_segment['frame_idx'] = np.array(frame_idxs)[valid_frames]
            table_segment['sequence_idx'] = np.array(len(bboxes) * [seq_idx])
            table_segment['subject_idx'] = np.array(len(bboxes) * [subject_idx])
            table_segment['bbox_by_camera_tlbr'] = bboxes
            table_segment['R'] = Rs
            table_segment['t'] = ts
            table_segment['K'] = Ks
            
            output_file['table'].append(table_segment)
            
    total_frame = np.concatenate([output_file['table'][exp]['t'] for exp in range(len(output_file['table']))]).shape[0]

    final_table = np.empty(total_frame, dtype=table_dtype)
    keys = ['keypoints_2d', 'keypoints', 'frame_idx', 'sequence_idx', 
            'subject_idx', 'bbox_by_camera_tlbr', 'R', 't', 'K']
    
    for key in keys:
        final_table[key] = np.concatenate([output_file['table'][exp][key] for exp in range(len(output_file['table']))])

    output_file['table'] = final_table

    output_destination = 'mpi_to_S17_train.npy'
    np.save(output_destination, output_file)