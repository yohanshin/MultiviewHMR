from utils import volumetric, multiview
from utils.multiview import triangulate_point_from_multiple_views_linear_torch

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import random
from copy import deepcopy


"""
This Code is mainly referred to the paper 
Iskakov et al, 'Learnable Triangulation for Human Pose'
please refer to the original code at https://github.com/karfly/learnable-triangulation-pytorch
"""


def unprojection(features, proj_matricies, coord_volumes, aggregation_method='softmax'):
    device = features.device
    batch_size, n_views, dim_features = features.shape[:3]
    feature_shape = tuple(features.shape[3:])
    volume_shape = coord_volumes.shape[1:4]
    volume_batch = torch.zeros(batch_size, dim_features, *volume_shape, device=device)

    # TODO: speed up this this loop
    for b in range(batch_size):
        coord_volume = coord_volumes[b]
        grid_coord = coord_volume.reshape((-1, 3))

        volume_batch_to_aggregate = torch.zeros(n_views, dim_features, *volume_shape, device=device)

        for v in range(n_views):
            feature = features[b, v]
            feature = feature.unsqueeze(0)

            grid_coord_proj = multiview.project_3d_points_to_image_plane_without_distortion(
                proj_matricies[b, v], grid_coord, convert_back_to_euclidean=False
            )

            invalid_mask = grid_coord_proj[:, 2] <= 0.0  # depth must be larger than 0.0

            grid_coord_proj[grid_coord_proj[:, 2] == 0.0, 2] = 1.0  # not to divide by zero
            grid_coord_proj = multiview.homogeneous_to_euclidean(grid_coord_proj)

            # transform to [-1.0, 1.0] range
            grid_coord_proj_transformed = torch.zeros_like(grid_coord_proj)
            grid_coord_proj_transformed[:, 0] = 2 * (grid_coord_proj[:, 0] / feature_shape[0] - 0.5)
            grid_coord_proj_transformed[:, 1] = 2 * (grid_coord_proj[:, 1] / feature_shape[1] - 0.5)
            grid_coord_proj = grid_coord_proj_transformed

            # prepare to F.grid_sample
            grid_coord_proj = grid_coord_proj.unsqueeze(1).unsqueeze(0)
            try:
                current_volume = F.grid_sample(feature, grid_coord_proj, align_corners=True)
            except TypeError: # old PyTorch
                current_volume = F.grid_sample(feature, grid_coord_proj)

            # zero out non-valid points
            current_volume = current_volume.view(dim_features, -1)
            current_volume[:, invalid_mask] = 0.0

            # reshape back to volume
            current_volume = current_volume.view(dim_features, *volume_shape)

            # collect
            volume_batch_to_aggregate[v] = current_volume

        # agregate resulting volume
        if aggregation_method == 'sum':
            volume_batch[b] = volume_batch_to_aggregate.sum(0)
        elif aggregation_method == 'mean':
            volume_batch[b] = volume_batch_to_aggregate.mean(0)
        elif aggregation_method == 'max':
            volume_batch[b] = volume_batch_to_aggregate.max(0)[0]
        elif aggregation_method == 'softmax':
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate.clone()
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate_softmin.view(n_views, -1)
            volume_batch_to_aggregate_softmin = F.softmax(volume_batch_to_aggregate_softmin, dim=0)
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate_softmin.view(n_views, dim_features, *volume_shape)

            volume_batch[b] = (volume_batch_to_aggregate * volume_batch_to_aggregate_softmin).sum(0)
        else:
            raise ValueError("Unknown aggregation_method: {}".format(aggregation_method))

    return volume_batch


class VolumeGenerator(nn.Module):
    def __init__(self,
                 volume_size=64,
                 input_channels=256,
                 output_channels=32,
                 cuboid_side=2500.0,
                 aggregation_method='softmax',
                 use_triangulation=False,
                 kind='mpii',
                 device='cuda',
                 dataset='human36m',
                 **kwargs):
        super(VolumeGenerator, self).__init__()

        self.volume_size = volume_size
        self.cuboid_side = cuboid_side
        self.aggregation_method = aggregation_method

        self.process_feature = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1)
        )

        self.use_triangulation = use_triangulation
        self.kind = kind
        self.dataset = dataset

        self.to(device)


    def forward(self, features, proj_matricies, batch, use_gt=True):
        features_shape = tuple(features.shape[-2:])
        images_shape = tuple(batch['images'].shape[2:-1])
        batch_size, n_views = batch['images'].shape[:2]
        device = features.device
        proj_matricies_org = proj_matricies.clone()

        # Update camera configs
        new_cameras = deepcopy(batch['cameras'])
        for v in range(n_views):
            for b in range(batch_size):
                new_cameras[v][b].update_after_resize(images_shape, features_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)
        proj_matricies = proj_matricies.float().to(device)

        cuboids = []
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)

        for b in range(batch_size):

            base_point = np.array([0, 0, 0])
            
            # build cuboid
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point - sides / 2

            cuboid = volumetric.Cuboid3D(position, sides)
            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), 
                                        torch.arange(self.volume_size, device=device), 
                                        torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            if self.kind == "coco":
                axis = [0, 1, 0]  # y axis
            elif self.kind == "mpii":
                axis = [0, 0, 1]  # z axis

            if self.use_triangulation:
                images_center = torch.tensor(images_shape)/2
                images_center = images_center.expand(n_views, 2).to(device=device)
                center = triangulate_point_from_multiple_views_linear_torch(proj_matricies_org[b], images_center)

            else:
                base_point = batch['keypoints_3d'][b][6, :3]
                center = torch.from_numpy(base_point).type(torch.float).to(device)

            # rotate
            coord_volume = coord_volume - center
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
            coord_volume = coord_volume + center
            coord_volumes[b] = coord_volume

        features = features.view(-1, *features.shape[2:])
        features = self.process_feature(features)
        features = features.view(batch_size, n_views, *features.shape[1:])
        
        volumes = unprojection(features, proj_matricies, coord_volumes, aggregation_method=self.aggregation_method)
        
        return volumes


def build_volume_generator(cfg):
    input_channels = cfg.MODEL.BACKBONE.DECONV_FILTERS[-1] if cfg.MODEL.BACKBONE.DECONV_LAYERS != 0 else 2048

    return VolumeGenerator(volume_size=cfg.MODEL.AGGREGATION.VOLUME_SIZE,
                           input_channels=input_channels,
                           output_channels=cfg.MODEL.AGGREGATION.OUTPUT_CHANNELS,
                           cuboid_side=cfg.MODEL.AGGREGATION.CUBOID_SIDE,
                           use_triangulation=cfg.MODEL.AGGREGATION.USE_TRIANGULATION,
                           kind=cfg.DATASET.KIND,
                           dataset=cfg.DATASET.TYPE,
                           volume_aggregation_method=cfg.MODEL.AGGREGATION.METHOD)