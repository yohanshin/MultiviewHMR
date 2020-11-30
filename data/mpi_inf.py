from data.image import *
from data import data_utils as d_utils
from utils.multiview import Camera
from utils.visualization import *

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import pickle
import os

from copy import deepcopy, copy


class MPI_INF_3DHP(Dataset):
    def __init__(self, 
                 root_pth='data/dataset/mpi_inf_3dhp',
                 label_pth='data/dataset/mpi_inf_3dhp/mpi_to_S17_v2.npy',
                 precalculated_pth='data/precalculated/',
                 precalculated_smplify_file = 'precal_mpi_inf.npy',
                 image_shape=(224, 224),
                 scale_bbox=1.2,
                 retain_every_n_frames_in_test=1,
                 ignore_cameras=[1, 3, 4, 5],
                 train=True,
                 kind='mpii',
                 norm_image=True,
                 crop=True,
                 with_damaged_actions=False,
                 eval_only=False,
                 **kwargs):
        super(MPI_INF_3DHP, self).__init__()
        
        self.root_pth = root_pth
        self.labels = np.load(label_pth, allow_pickle=True).item()
        self.image_shape = None if image_shape is None else tuple(image_shape)
        self.ignore_cameras = ignore_cameras
        self.scale_bbox = scale_bbox
        self.norm_image = norm_image
        self.crop = crop
        self.eval_only = eval_only

        # n_cameras = len(self.labels['camera_names']) - len(ignore_cameras)
        precalculated_smplify_pth = os.path.join(precalculated_pth, precalculated_smplify_file)

        train_subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
        test_subjects = ['S8']

        train_subjects = list(self.labels['subject_names'].index(x) for x in train_subjects)
        test_subjects  = list(self.labels['subject_names'].index(x) for x in test_subjects)

        indices = []
        if train:
            mask = np.isin(self.labels['table']['subject_idx'], train_subjects, assume_unique=True)
            indices.append(np.nonzero(mask)[0])
        else:
            mask = np.isin(self.labels['table']['subject_idx'], test_subjects, assume_unique=True)
            indices.append(np.nonzero(mask)[0][::retain_every_n_frames_in_test])

        self.labels['table'] = self.labels['table'][np.concatenate(indices)]

        assert self.labels['table']['keypoints'].shape[1] == 17, "Use a newer 'labels' file"

        if precalculated_smplify_pth is not None:
            smplify_result = np.load(precalculated_smplify_pth, allow_pickle=True)
            self.smplify_pose = smplify_result['pose']
            self.smplify_global_orient = smplify_result['orient']
            self.smplify_shape = smplify_result['betas']
        else:
            self.smplify_pose = None
    
    def __len__(self):
        return len(self.labels['table'])

    def __getitem__(self, idx):
        sample = defaultdict(list)
        shot = self.labels['table'][idx]
        subject = self.labels['subject_names'][shot['subject_idx']]
        sequence = self.labels['sequence_names'][shot['sequence_idx']]
        frame_idx = shot['frame_idx']

        for camera_idx, camera_name in enumerate(self.labels['camera_names']):
            if camera_idx in self.ignore_cameras:
                continue
            
            # bbox = shot['bbox_by_camera_tlbr'][camera_idx][[1, 0, 3, 2]]
            bbox = shot['bbox_by_camera_tlbr'][camera_idx]
            bbox_length = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            if bbox_height == 0:
                continue

            center = [(bbox[3]+bbox[1])/2, (bbox[2]+bbox[0])/2]
            scale = 0.9*max(bbox[3]-bbox[1], bbox[2]-bbox[0])/200.
            
            length = int(max(bbox_length, bbox_height) / 2)
            center = [int((bbox[2] + bbox[0])/2), int((bbox[3] + bbox[1])/2)]
            bbox = [center[0] - length, center[1] - length, center[0] + length, center[1] + length]


            image_path = os.path.join(
                self.root_pth, subject, sequence, 'imageSequence',
                'image_' + camera_name, 'frame_%06d.jpg'%frame_idx
                )
            assert os.path.isfile(image_path), '%s does not exist!'%image_path
            image = cv2.imread(image_path)[:, :, ::-1]
            retval_camera = Camera(shot['R'][camera_idx],
                                   shot['t'][camera_idx],
                                   shot['K'][camera_idx],
                                   name=self.labels['camera_names'][camera_idx]
                                   )

            bbox = get_square_bbox_SPIN(center, scale, image.shape[:2], self.image_shape)

            if self.eval_only:
                val_res = (480, 480)
                big_bbox = scale_bbox(deepcopy(bbox), 1.5)
                val_image = crop_image(deepcopy(image), big_bbox)
                val_image_shape_before_resize = val_image.shape[:2]
                val_image = resize_image(val_image, val_res)

                val_retval_camera = deepcopy(retval_camera)
                val_retval_camera.update_after_crop(big_bbox)
                val_retval_camera.update_after_resize(val_image_shape_before_resize, val_res)
                
                sample['org_images'].append(val_image)
                sample['org_cameras'].append(deepcopy(val_retval_camera))

            if self.crop:
                bbox = scale_bbox(bbox, self.scale_bbox)
                image = crop_image(image, bbox)
                retval_camera.update_after_crop(bbox)

            if self.image_shape is not None:
                image_shape_before_resize = image.shape[:2]
                image = resize_image(image, self.image_shape)
                retval_camera.update_after_resize(image_shape_before_resize, self.image_shape)

                sample['image_shape_before_resize'].append(image_shape_before_resize)

            if self.eval_only:
                sample['images_before_norm'].append(image)
            
            if self.norm_image:
                image = normalize_image(image)

            sample['images'].append(image)
            sample['detections'].append(bbox + (1.0,))
            sample['cameras'].append(retval_camera)
            sample['proj_matrices'].append(retval_camera.projection)

        sample['keypoints_3d'] = shot['keypoints']

        if self.smplify_pose is not None:
            sample['smplify_pose'] = self.smplify_pose[idx]
            sample['smplify_global_orient'] = self.smplify_global_orient[idx]
            sample['smplify_shape'] = self.smplify_shape[idx]

        sample['indexes'] = idx
        sample['dataset'] = 'MPI_INF_3DHP'
        sample['has_smpl'] = False
        sample.default_factory = None
        return sample


def setup_mpi_dataloaders(cfg):
    print('Load MPI-INF Dataset...')

    val_dataset = MPI_INF_3DHP(
                            image_shape=cfg.DATASET.IMAGE_SHAPE,
                            train=False,
                            kind=cfg.DATASET.KIND,
                            norm_image=cfg.DATASET.NORM_IMAGE,
                            crop=cfg.DATASET.CROP_IMAGE,
                            retain_every_n_frames_in_test=cfg.DATASET.RETAIN_EVERY_N_FRAMES_IN_TEST,
                            eval_only=cfg.EVAL)

    val_dataloader = DataLoader(val_dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  shuffle=cfg.DATASET.VAL_SHUFFLE,
                                  sampler=None,
                                  collate_fn=d_utils.make_collate_fn(randomize_n_views=cfg.DATASET.RANDOMIZE_N_VIEWS,
                                                                     min_n_views=cfg.DATASET.MIN_N_VIEWS,
                                                                     max_n_views=cfg.DATASET.MAX_N_VIEWS),
                                  num_workers=cfg.DATASET.NUM_WORKERS,
                                  worker_init_fn=d_utils.worker_init_fn,
                                  pin_memory=True)

    if cfg.EVAL:
        return val_dataloader
    
    train_dataset = MPI_INF_3DHP(
                            image_shape=cfg.DATASET.IMAGE_SHAPE,
                            train=True,
                            kind=cfg.DATASET.KIND,
                            norm_image=cfg.DATASET.NORM_IMAGE,
                            crop=cfg.DATASET.CROP_IMAGE)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  shuffle=cfg.DATASET.TRAIN_SHUFFLE,
                                  sampler=None,
                                  collate_fn=d_utils.make_collate_fn(randomize_n_views=cfg.DATASET.RANDOMIZE_N_VIEWS,
                                                                     min_n_views=cfg.DATASET.MIN_N_VIEWS,
                                                                     max_n_views=cfg.DATASET.MAX_N_VIEWS),
                                  num_workers=cfg.DATASET.NUM_WORKERS,
                                  worker_init_fn=d_utils.worker_init_fn,
                                  pin_memory=True)

    return train_dataloader, val_dataloader