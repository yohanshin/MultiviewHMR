from models.backbone import build_backbone
from models.aggregation import build_volume_generator
from models.regressor import build_volumetric_regressor

import torch
import torch.nn as nn

import os.path as osp


class MarkerlessMoCap(nn.Module):
    def __init__(self, backbone, aggregator, volumetric_regressor, device):
        super(MarkerlessMoCap, self).__init__()

        self.backbone = backbone
        self.aggregator = aggregator
        self.volumetric_regressor = volumetric_regressor

        self.device = device
        self.to(self.device)

    def forward(self, images, proj_matricies, batch):
        batch_size, n_views = images.shape[:2]
        
        # Extract features from multi-view images
        images = images.view(-1, *images.shape[2:])
        features = self.backbone(images)
        features = features.view(batch_size, n_views, *features.shape[1:])
        
        # Aggregate multi-view features into volumetric model
        volumes = self.aggregator(features, proj_matricies, batch)

        # Predict SMPL parameters and vertices if landmarks regressor
        pred_rotmat, pred_betas, pred_global_oient, pred_vertices = \
            self.volumetric_regressor(volumes)

        return pred_rotmat, pred_betas, pred_global_oient, pred_vertices


def build_model(cfg):

    backbone = build_backbone(cfg)
    aggregator = build_volume_generator(cfg)
    volumetric_regressor = build_volumetric_regressor(cfg)

    return MarkerlessMoCap(backbone, aggregator, volumetric_regressor, cfg.DEVICE)