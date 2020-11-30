from data.image import *
from data import data_utils as d_utils
from utils.multiview import Camera
from cfg import constants

import torch
from torchvision.transforms import Normalize
from torch.utils.data import Dataset, DataLoader
import numpy as np

import cv2
from collections import defaultdict
from copy import copy, deepcopy
import pickle
import os


class H36MDataset(Dataset):
    def __init__(self,
                 image_shape=(224, 224),
                 scale_bbox=1.0,
                 retain_every_n_frames_in_train=1,
                 retain_every_n_frames_in_test=50,
                 ignore_cameras=[],
                 undistort_images=False,
                 train=True,
                 kind='mpii',
                 norm_image=True,
                 crop=True,
                 with_damaged_actions=False,
                 eval_only=False,
                 **kwargs
                 ):

        super(H36MDataset, self).__init__()

        self.root_pth = constants.H36M_ROOT_PTH
        self.labels = np.load(os.path.join(constants.H36M_ROOT_PTH, constants.H36M_LABEL), allow_pickle=True).item()
        self.undistort_images = undistort_images
        self.image_shape = None if image_shape is None else tuple(image_shape)
        self.ignore_cameras = ignore_cameras
        self.scale_bbox = scale_bbox
        self.norm_image = norm_image
        self.crop = crop
        self.eval_only = eval_only
        self.normalizer = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        smpl_params_file = os.path.join(constants.H36M_ROOT_PTH, constants.H36M_SMPL_PARAMS)
        train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        test_subjects = ['S9', 'S11']

        train_subjects = list(self.labels['subject_names'].index(x) for x in train_subjects)
        test_subjects  = list(self.labels['subject_names'].index(x) for x in test_subjects)

        indices = []
        if train:
            mask = np.isin(self.labels['table']['subject_idx'], train_subjects, assume_unique=True)
            indices.append(np.nonzero(mask)[0][::retain_every_n_frames_in_train])
        else:
            mask = np.isin(self.labels['table']['subject_idx'], test_subjects, assume_unique=True)

            if not with_damaged_actions:
                mask_S9 = self.labels['table']['subject_idx'] == self.labels['subject_names'].index('S9')

                damaged_actions = 'Greeting-2', 'SittingDown-2', 'Waiting-1'
                damaged_actions = [self.labels['action_names'].index(x) for x in damaged_actions]
                mask_damaged_actions = np.isin(self.labels['table']['action_idx'], damaged_actions)

                mask &= ~(mask_S9 & mask_damaged_actions)
            indices.append(np.nonzero(mask)[0][::retain_every_n_frames_in_test])

        self.labels['table'] = self.labels['table'][np.concatenate(indices)]

        # smpl_params_file = None
        if smpl_params_file is not None and train:
            smplify_result = np.load(smpl_params_file, allow_pickle=True)
            self.smplify_pose = smplify_result['pose'][np.concatenate(indices)]
            self.smplify_global_orient = smplify_result['orient'][np.concatenate(indices)]
            self.smplify_shape = smplify_result['betas'][np.concatenate(indices)]
        else:
            self.smplify_pose = None

    def __len__(self):
        return len(self.labels['table'])

    def __getitem__(self, idx):
        sample = defaultdict(list)
        shot = self.labels['table'][idx]
        subject = self.labels['subject_names'][shot['subject_idx']]
        action = self.labels['action_names'][shot['action_idx']]
        frame_idx = shot['frame_idx']

        for camera_idx, camera_name in enumerate(self.labels['camera_names']):
            if camera_idx in self.ignore_cameras:
                continue

            # load bounding box
            bbox = shot['bbox_by_camera_tlbr'][camera_idx][[1,0,3,2]] # TLBR to LTRB
            bbox_height = bbox[2] - bbox[0]
            if bbox_height == 0:
                # convention: if the bbox is empty, then this view is missing
                continue

            center = [(bbox[3]+bbox[1])/2, (bbox[2]+bbox[0])/2]
            scale = 0.9*max(bbox[3]-bbox[1], bbox[2]-bbox[0])/200.

            # load image
            image_path = os.path.join(
                self.root_pth, 'images', subject, camera_name, '%s_%06d.jpg' % (action, frame_idx))
            assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
            image = cv2.imread(image_path)[:, :, ::-1]
            
            # load camera
            shot_camera = self.labels['cameras'][shot['subject_idx'], camera_idx]
            retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'], shot_camera['dist'], camera_name)

            bbox = get_square_bbox_SPIN(center, scale, image.shape[:2], self.image_shape)
            
            if self.eval_only:
                val_res = (640, 640)
                # big_bbox = scale_bbox(deepcopy(bbox), 1.5)
                # val_image = crop_image(deepcopy(image), big_bbox)
                
                val_image = deepcopy(image)
                val_image_shape_before_resize = val_image.shape[:2]
                val_image = resize_image(val_image, val_res)

                val_retval_camera = deepcopy(retval_camera)
                # val_retval_camera.update_after_crop(big_bbox)
                val_retval_camera.update_after_resize(val_image_shape_before_resize, val_res)
                
                sample['org_images'].append(val_image)
                sample['org_cameras'].append(deepcopy(val_retval_camera))

            if self.crop:
                bbox = scale_bbox(bbox, self.scale_bbox)
                # crop image
                image = crop_image(image, bbox)
                retval_camera.update_after_crop(bbox)
            
            if self.image_shape is not None:
                # resize                
                image_shape_before_resize = image.shape[:2]
                image = resize_image(image, self.image_shape)
                retval_camera.update_after_resize(image_shape_before_resize, self.image_shape)

            if self.norm_image:
                image = normalize_image(image)

            sample['images'].append(image)
            sample['detections'].append(bbox + (1.0,)) # TODO add real confidences
            sample['cameras'].append(retval_camera)
            sample['proj_matrices'].append(retval_camera.projection)

        sample['keypoints_3d'] = shot['keypoints'][constants.J32_TO_J17]
        sample['keypoints_3d_cam'] = shot['keypoints_cam'][:, constants.J32_TO_J17]

        if self.smplify_pose is not None:
            sample['smplify_pose'] = self.smplify_pose[idx]
            sample['smplify_global_orient'] = self.smplify_global_orient[idx]
            sample['smplify_shape'] = self.smplify_shape[idx]

        # save sample's index
        sample['indexes'] = idx        
        sample['dataset'] = 'Human36M'
        sample['action'] = action
        sample['frame'] = frame_idx
        sample['has_smpl'] = True
        sample.default_factory = None
        
        return sample


def setup_human36m_dataloaders(cfg, **kwargs):

    print('Load H36M Dataset...')
    
    val_dataset = H36MDataset(image_shape=cfg.DATASET.IMAGE_SHAPE,
                              scale_bbox=cfg.DATASET.SCALE_BBOX,
                              undistort_images=cfg.DATASET.UNDISTORT_IMAGE,
                              retain_every_n_frames_in_test=cfg.DATASET.RETAIN_EVERY_N_FRAMES_IN_TEST,
                              train=False,
                              kind=cfg.DATASET.KIND,
                              norm_image=cfg.DATASET.NORM_IMAGE,
                              ignore_cameras=cfg.DATASET.IGNORE_CAMERAS,
                              crop=cfg.DATASET.CROP_IMAGE,
                              eval_only=cfg.EVAL)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.TRAIN.BATCH_SIZE,
                                shuffle=cfg.DATASET.VAL_SHUFFLE,
                                sampler=None,
                                collate_fn=d_utils.make_collate_fn(
                                    randomize_n_views=cfg.DATASET.RANDOMIZE_N_VIEWS,
                                    min_n_views=cfg.DATASET.MIN_N_VIEWS,
                                    max_n_views=cfg.DATASET.MAX_N_VIEWS),
                                num_workers=cfg.DATASET.NUM_WORKERS,
                                worker_init_fn=d_utils.worker_init_fn,
                                pin_memory=True)
    
    if cfg.EVAL:
        return val_dataloader
    
    train_dataset = H36MDataset(image_shape=cfg.DATASET.IMAGE_SHAPE,
                                scale_bbox=cfg.DATASET.SCALE_BBOX,
                                undistort_images=cfg.DATASET.UNDISTORT_IMAGE,
                                retain_every_n_frames_in_train=cfg.DATASET.RETAIN_EVERY_N_FRAMES_IN_TRAIN,
                                train=True,
                                kind=cfg.DATASET.KIND,
                                norm_image=cfg.DATASET.NORM_IMAGE,
                                crop=cfg.DATASET.CROP_IMAGE)
    train_sampler = None

    train_dataloader = DataLoader(train_dataset,
                                batch_size=cfg.TRAIN.BATCH_SIZE,
                                shuffle=cfg.DATASET.TRAIN_SHUFFLE,
                                sampler=train_sampler,
                                collate_fn=d_utils.make_collate_fn(
                                    randomize_n_views=cfg.DATASET.RANDOMIZE_N_VIEWS,
                                    min_n_views=cfg.DATASET.MIN_N_VIEWS,
                                    max_n_views=cfg.DATASET.MAX_N_VIEWS),
                                num_workers=cfg.DATASET.NUM_WORKERS,
                                worker_init_fn=d_utils.worker_init_fn,
                                pin_memory=True)
    
    return train_dataloader, val_dataloader