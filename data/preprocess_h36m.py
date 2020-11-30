import os
if not "CDF_LIB" in os.environ:
    base_dir = "/home/soyongs/Downloads/cdf37_1-dist"
    os.environ["CDF_BASE"] = base_dir
    os.environ["CDF_BIN"] = base_dir + "/bin"
    os.environ["CDF_INC"] = base_dir + "/include"
    os.environ["CDF_LIB"] = base_dir + "/lib"
    os.environ["CDF_JAVA"] = base_dir + "/java"
    os.environ["CDF_HELP"] = base_dir + "/lib/help"

import sys
import cv2
import glob
import h5py
import numpy as np
import argparse
from spacepy import pycdf
from tqdm import tqdm


una_dinosauria_root = 'data/dataset/human36m/extra/una-dinosauria-data/h36m'

def cam_to_world(loc3D, R, T):
    """ Convert local 3D points to global coordinate
    """
    glob3D = R.T.dot(loc3D.T - T)

    return glob3D.T


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


def world_to_cam(glob3d, R, t):
    """ Convert global 3D points to local coordinate
    """
    loc3d = R @ glob3d.T + t[:, None]
    
    return loc3d.T


def h36m_extract(dataset_path, out_path, extract_img=False):

    output = {
        'subject_names': ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'],
        'camera_names': ['54138969', '55011271', '58860488', '60457274'],
        'action_names': [
            'Directions-1', 'Directions-2',
            'Discussion-1', 'Discussion-2',
            'Eating-1', 'Eating-2',
            'Greeting-1', 'Greeting-2',
            'Phoning-1', 'Phoning-2',
            'Posing-1', 'Posing-2',
            'Purchases-1', 'Purchases-2',
            'Sitting-1', 'Sitting-2',
            'SittingDown-1', 'SittingDown-2',
            'Smoking-1', 'Smoking-2',
            'TakingPhoto-1', 'TakingPhoto-2',
            'Waiting-1', 'Waiting-2',
            'Walking-1', 'Walking-2',
            'WalkingDog-1', 'WalkingDog-2',
            'WalkingTogether-1', 'WalkingTogether-2']
    }

    output['cameras'] = np.empty(
        (len(output['subject_names']), len(output['camera_names'])),
        dtype=[
            ('R', np.float32, (3,3)),
            ('t', np.float32, (3,1)),
            ('K', np.float32, (3,3)),
            ('dist', np.float32, 5)
        ]
    )

    output['table'] = []
    table_dtype = np.dtype([
        ('subject_idx', np.int8),
        ('action_idx', np.int8),
        ('frame_idx', np.int16),
        ('scale', np.float32, (len(output['camera_names']))),
        ('center', np.float32, (len(output['camera_names']), 2)),
        ('keypoints_2d', np.float32, (len(output['camera_names']), 32, 2)),
        ('keypoints', np.float32, (32, 3)),
        ('keypoints_cam', np.float32, (len(output['camera_names']), 32, 3)),
        ('bbox_by_camera_tlbr', np.int16, (len(output['camera_names']), 4))])


    # 3D and 2D poses
    poses_fldr = {'3D': 'D3_Positions_mono', '3D-world': 'D3_Positions_mono_univ', '2D': 'D2_Positions'}

    # go over each user
    for subj_idx, subj_name in enumerate(output['subject_names']):
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, subj_name, 'MySegmentsMat', 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, subj_name, 'MyPoseFeatures')
        # path with videos
        vid_path = os.path.join(dataset_path, subj_name, 'Videos')

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, poses_fldr['3D'], '*.cdf'))
        seq_list.sort()

        # Get camera params
        cameras_params = h5py.File(os.path.join(una_dinosauria_root, 'cameras.h5'), 'r')


        for camera_idx, camera in enumerate(output['camera_names']):
            camera_params = cameras_params[subj_name.replace('S', 'subject')]['camera%d' % (camera_idx+1)]
            camera_retval = output['cameras'][subj_idx][camera_idx]
            
            def camera_array_to_name(array):
                return ''.join(chr(int(x[0])) for x in array)
            assert camera_array_to_name(camera_params['Name']) == camera

            camera_retval['R'] = np.array(camera_params['R']).T
            camera_retval['t'] = -camera_retval['R'] @ camera_params['T']

            camera_retval['K'] = 0
            camera_retval['K'][:2, 2] = camera_params['c'][:, 0]
            camera_retval['K'][0, 0] = camera_params['f'][0]
            camera_retval['K'][1, 1] = camera_params['f'][1]
            camera_retval['K'][2, 2] = 1.0

            camera_retval['dist'][:2] = camera_params['k'][:2, 0]
            camera_retval['dist'][2:4] = camera_params['p'][:, 0]
            camera_retval['dist'][4] = camera_params['k'][2, 0]
        
        assert len(seq_list) % 4 == 0
        with tqdm(desc=subj_name, total=len(seq_list[::4]), leave=True) as seq_bar:
            for action_idx, seq_i in enumerate(seq_list[::4]):
                
                # sequence info
                seq_name = seq_i.split('/')[-1]
                action, camera_, _ = seq_name.split('.')
                # action = action.replace(' ', '-')
                action = output['action_names'][action_idx]
                # irrelevant sequences
                
                if action == '_ALL':
                    continue
                
                bboxes_cams_, centers_cams_, scales_cams_, S32_2ds_cams_, S32_3ds_cams_, S32_3ds_univ_cams_ = [], [], [], [], [], []
                
                with tqdm(total=len(output['camera_names']), leave=False) as cam_bar:
                    for camera_idx, camera in enumerate(output['camera_names']):

                        subjects_, actions_, frames_, bboxes_, centers_, scales_, S32_2ds_, S32_3ds_, S32_3ds_univ_\
                            = [], [], [], [], [], [], [], [], []
                        
                        seq_i_ = seq_i.replace('54138969', camera)
                        cam_info = output['cameras'][subj_idx][camera_idx]
                        S32_2d = pycdf.CDF(seq_i_.replace(poses_fldr['3D'], poses_fldr['2D']))['Pose'][0].reshape(-1, 32, 2)
                        S32_3d = pycdf.CDF(seq_i_)['Pose'][0].reshape(-1, 32, 3)
                        S32_3d_univ = cam_to_world(S32_3d.reshape(-1, 3), cam_info['R'], cam_info['t']).reshape(-1, 32, 3)

                        seq_name_ = seq_name.replace('54138969', camera)
                        bbox_file = os.path.join(bbox_path, seq_name_.replace('cdf', 'mat'))
                        bbox_h5py = h5py.File(bbox_file)

                        # video file
                        if extract_img:
                            vid_file = os.path.join(vid_path, seq_name_.replace('cdf', 'mp4'))
                            imgs_path = os.path.join(dataset_path, 'MuVHMR', 'images', subj_name, camera)
                            vidcap = cv2.VideoCapture(vid_file)
                            success, image = vidcap.read()
                            if not os.path.isdir(imgs_path):
                                os.makedirs(imgs_path, True)
                            
                        # structs we use
                        with tqdm(total=S32_2d.shape[0]//5, leave=False) as frame_bar:
                            for frame_i in range(S32_2d.shape[0]):
                                if extract_img:
                                    success, image = vidcap.read()
                                    if not success:
                                        break

                                # check if you can keep this frame
                                if frame_i % 5 == 0:
                                    # image name
                                    imgname = '%s_%06d.jpg' % (action, frame_i+1)
                                    
                                    # save image
                                    if extract_img:
                                        img_out = os.path.join(imgs_path, imgname)
                                        cv2.imwrite(img_out, image)

                                    # read GT bounding box
                                    mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]].value.T
                                    ys, xs = np.where(mask==1)
                                    bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
                                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                                    scale = 0.9*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.

                                    subjects_.append(subj_idx)
                                    actions_.append(action_idx)
                                    frames_.append(frame_i+1)
                                    bboxes_.append(bbox)
                                    centers_.append(center)
                                    scales_.append(scale)
                                    S32_2ds_.append(S32_2d[frame_i])
                                    S32_3ds_.append(S32_3d[frame_i])
                                    S32_3ds_univ_.append(S32_3d_univ[frame_i])
                                    
                                    frame_bar.update(1)
                                    frame_bar.refresh()

                        cam_bar.update(1)
                        cam_bar.refresh()

                        bboxes_cams_.append(np.array(bboxes_))
                        centers_cams_.append(centers_)
                        scales_cams_.append(scales_)
                        S32_2ds_cams_.append(np.array(S32_2ds_))
                        S32_3ds_cams_.append(np.array(S32_3ds_))
                        S32_3ds_univ_cams_.append(np.array(S32_3ds_univ_))
            
                S32_3ds_univ_ = np.array(S32_3ds_univ_cams_).transpose((1, 0, 2, 3))
                assert S32_3ds_univ_.std(1).max() < 1e-2
                S32_3ds_univ_ = S32_3ds_univ_.mean(1)

                bboxes_ = np.array(bboxes_cams_).transpose((1, 0, 2))
                centers_ = np.array(centers_cams_).transpose((1, 0, 2))
                scales_ = np.array(scales_cams_).transpose()
                S32_2ds_ = np.array(S32_2ds_cams_).transpose((1, 0, 2, 3))
                S32_3ds_ = np.array(S32_3ds_cams_).transpose((1, 0, 2, 3))

                table_segment = np.empty(len(frames_), dtype=table_dtype)
                table_segment['subject_idx'] = subjects_
                table_segment['action_idx'] = actions_
                table_segment['frame_idx'] = frames_
                table_segment['keypoints_2d'] = S32_2ds_
                table_segment['keypoints_cam'] = S32_3ds_
                table_segment['keypoints'] = S32_3ds_univ_
                table_segment['bbox_by_camera_tlbr'] = bboxes_
                table_segment['center'] = centers_
                table_segment['scale'] = scales_
                
                output['table'].append(table_segment)

                seq_bar.update(1)
                seq_bar.refresh()
    
    output['table'] = np.concatenate(output['table'])
    assert output['table'].ndim == 1

    print("Total frames in Human3.6Million:", len(output['table']))
    np.save(os.path.join(out_path, 'H36M_label.npy'), output)

if __name__ == '__main__':
    h36m_extract('data/dataset/human36m-raw', 'data/dataset/human36m-raw/MuVHMR', False)