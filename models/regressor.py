from models.v2v import *
from utils.geometry import *

import torch
import torch.nn as nn
from torch.nn import functional as F

import math


class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size, pool_type='max'):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size
        self.pool_type = pool_type

    def forward(self, x):
        
        if self.pool_type == 'max':
            return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)
        elif self.pool_type == 'avg':
            return F.avg_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)
        else:
            NotImplementedError, "pooling type {} has not been implemented".format(self.pool_type)


class Encoder(nn.Module):
    def __init__(self, input_channel, channels, volume_size=32):
        super().__init__()
        self.volume_size = volume_size

        self.layer1 = Res3DBlock(input_channel, channels[0])
        self.pool1 = Pool3DBlock(2, 'max')
        self.layer2 = Res3DBlock(channels[0], channels[1])
        self.pool2 = Pool3DBlock(2, 'max')
        self.layer3 = Res3DBlock(channels[1], channels[2])
        self.pool3 = Pool3DBlock(2, 'max')        
        self.layer_fin = Res3DBlock(channels[4], channels[4])

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.layer_fin(x)

        return x


class VolumetricHMR(nn.Module):
    def __init__(self, 
                 input_channel, 
                 large_model=False,
                 volume_size=32,
                 encoder_channels=(128, 512, 512, 2048),
                 smpl_mean_params='data/dataset/SPIN/data/smpl_mean_params.npz',
                 device='cuda',
                 **kwargs):

        super(VolumetricHMR, self).__init__()

        self.encoder = Encoder(input_channel=input_channel, channels=encoder_channels, volume_size=volume_size)
        
        self.back_layers = nn.Sequential(
            Res3DBlock(encoder_channels[-1], encoder_channels[-1]),
            Basic3DBlock(encoder_channels[-1], encoder_channels[-1], 1),
            Basic3DBlock(encoder_channels[-1], encoder_channels[-1], 1),
        )
        
        self.maxpool = Pool3DBlock(2, 'max')
        self.avgpool = Pool3DBlock(2, 'avg')

        dec_input = encoder_channels[-1] * 8 + 24*6 + 10

        self.fc1 = nn.Linear(dec_input, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, 24 * 6)
        self.decshape = nn.Linear(1024, 10)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        
        self.to(device)

    def forward(self, x, init_pose=None, init_shape=None, n_iter=3):

        batch_size = x.shape[0]

        x = self.encoder(x)
        x = self.back_layers(x)
        x = x.view(batch_size, -1)

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        
        pred_pose = init_pose
        pred_shape = init_shape

        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape

        pred_pose = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        
        pred_global_orient = pred_pose[:, 0].unsqueeze(1)
        pred_pose = pred_pose[:, 1:]

        return pred_pose, pred_shape, pred_global_orient, None


def build_volumetric_regressor(cfg):
    model = VolumetricHMR(input_channel=cfg.MODEL.AGGREGATION.OUTPUT_CHANNELS,
                          large_model=cfg.MODEL.VHMR.LARGE_MODEL,
                          encoder_channels=cfg.MODEL.VHMR.ENCODER_CHANNELS,
                          volume_size=cfg.MODEL.AGGREGATION.VOLUME_SIZE,
                          smpl_mean_params=cfg.SMPL.MEAN_PARAMS,
                          device=cfg.DEVICE)

    if cfg.MODEL.VHMR.INIT_WEIGHT:
        pretrained_dict = torch.load(cfg.MODEL.VHMR.INIT_WEIGHT)['model']
        init_list = ['decpose.weight', 'decpose.bias', 'decshape.weight', 'decshape.bias']
    
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in init_list}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model