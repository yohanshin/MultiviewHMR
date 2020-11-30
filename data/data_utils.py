from data.image import *

import numpy as np
import torch


def make_collate_fn(randomize_n_views=False, min_n_views=10, max_n_views=31):

    def collate_fn(items):
        items = list(filter(lambda x: x is not None, items))
        if len(items) == 0:
            print("All items in batch are None")
            return None

        batch = dict()
        total_n_views = min(len(item['images']) for item in items)

        indexes = np.arange(total_n_views)
        if randomize_n_views:
            n_views = np.random.randint(min_n_views, min(total_n_views, max_n_views) + 1)
            indexes = np.random.choice(np.arange(total_n_views), size=n_views, replace=False)
        else:
            indexes = np.arange(total_n_views)

        batch['images'] = np.stack([np.stack([item['images'][i] for item in items], axis=0) for i in indexes], axis=0).swapaxes(0, 1)
        batch['detections'] = np.array([[item['detections'][i] for item in items] for i in indexes]).swapaxes(0, 1)
        batch['cameras'] = [[item['cameras'][i] for item in items] for i in indexes]

        batch['keypoints_3d'] = [item['keypoints_3d'] for item in items]
        batch['indexes'] = [item['indexes'] for item in items]
        batch['has_smpl'] = [item['has_smpl'] for item in items]
        batch['action'] = [item['action'] for item in items]
        batch['frame'] = [item['frame'] for item in items]

        try:
            batch['keypoints_3d_cam'] = [item['keypoints_3d_cam'] for item in items]
        except:
            pass

        try:
            batch['org_images'] = np.stack([np.stack([item['org_images'][i] for item in items], axis=0) for i in indexes], axis=0).swapaxes(0, 1)
            batch['org_cameras'] = [[item['org_cameras'][i] for item in items] for i in indexes]
        except:
            pass

        try:
            batch['smplify_pose'] = np.array([item['smplify_pose'] for item in items])
            batch['smplify_global_orient'] = np.array([item['smplify_global_orient'] for item in items])
            batch['smplify_shape'] = np.array([item['smplify_shape'] for item in items])
        except:
            pass
        
        return batch

    return collate_fn


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def prepare_batch(batch, device, prepare_images=True):
    # images
    if prepare_images:
        images_batch = []
        for image_batch in batch['images']:
            image_batch = image_batch_to_torch(image_batch)
            image_batch = image_batch.to(device)
            images_batch.append(image_batch)

        images_batch = torch.stack(images_batch, dim=0)
    else:
        images_batch = None

    # 3D keypoints
    keypoints_3d_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)).float().to(device)
    proj_matricies_batch = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in batch['cameras']], dim=0).transpose(1, 0)  
    proj_matricies_batch = proj_matricies_batch.float().to(device)

    return images_batch, keypoints_3d_batch_gt, proj_matricies_batch