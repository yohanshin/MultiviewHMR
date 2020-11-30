from data import data_utils as d_utils
from data.new_h36m import H36MDataset
from data.mpi_inf import MPI_INF_3DHP

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MixtureDataset(Dataset):
    def __init__(self, cfg, train=True):
        self.cfg = cfg
        self.train = train

        self.dataset_list = ['h36m', 'mpi-inf-3dhp']
        self.dataset_dict = {'h36m': 0, 'mpi-inf-3dhp': 1}
        self.datasets = [self.get_h36m(), self.get_mpi_inf_3dhp()]
        total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
        self.lengths = [len(ds) for ds in self.datasets]
        self.partition = [0.75, 1.0]
        
    def __getitem__(self, idx):
        p = np.random.rand()
        for i in range(2):
            if p <= self.partition[i]:
                return self.datasets[i][idx % len(self.datasets[i])]

    def __len__(self):
        return max(self.lengths)

    def get_h36m(self):
        return H36MDataset(root_pth=self.cfg.DATASET.ROOT_PTH,
                           image_shape=self.cfg.DATASET.IMAGE_SHAPE,
                           scale_bbox=self.cfg.DATASET.SCALE_BBOX,
                           undistort_images=self.cfg.DATASET.UNDISTORT_IMAGE,
                           train=True,
                           kind=self.cfg.DATASET.KIND,
                           norm_image=self.cfg.DATASET.NORM_IMAGE,
                           crop=self.cfg.DATASET.CROP_IMAGE)

    def get_mpi_inf_3dhp(self):
        return MPI_INF_3DHP(image_shape=self.cfg.DATASET.IMAGE_SHAPE,
                            scale_bbox=self.cfg.DATASET.SCALE_BBOX,
                            train=True,
                            kind=self.cfg.DATASET.KIND,
                            norm_image=self.cfg.DATASET.NORM_IMAGE,
                            crop=self.cfg.DATASET.CROP_IMAGE)


def setup_mixed_dataloaders(cfg):
    print('Load Mixed Dataset...')
    
    train_dataset = MixtureDataset(cfg, train=True)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  shuffle=True,
                                  sampler=None,
                                  collate_fn=d_utils.make_collate_fn(randomize_n_views=cfg.DATASET.RANDOMIZE_N_VIEWS,
                                                                     min_n_views=cfg.DATASET.MIN_N_VIEWS,
                                                                     max_n_views=cfg.DATASET.MAX_N_VIEWS),
                                  num_workers=cfg.DATASET.NUM_WORKERS,
                                  worker_init_fn=d_utils.worker_init_fn,
                                  pin_memory=True)

    val_h36m_dataset = H36MDataset(
                                   precalculated_pth=cfg.DATASET.PRECAL_PTH,
                                   image_shape=cfg.DATASET.IMAGE_SHAPE,
                                   scale_bbox=cfg.DATASET.SCALE_BBOX,
                                   undistort_images=cfg.DATASET.UNDISTORT_IMAGE,
                                   train=False,
                                   kind=cfg.DATASET.KIND,
                                   norm_image=cfg.DATASET.NORM_IMAGE,
                                   crop=cfg.DATASET.CROP_IMAGE)

    val_mpi_dataset = MPI_INF_3DHP(
                                   image_shape=cfg.DATASET.IMAGE_SHAPE,
                                   scale_bbox=cfg.DATASET.SCALE_BBOX,
                                   train=False,
                                   kind=cfg.DATASET.KIND,
                                   norm_image=cfg.DATASET.NORM_IMAGE,
                                   crop=cfg.DATASET.CROP_IMAGE)

    val_h36m_dataloader = DataLoader(val_h36m_dataset,
                                     batch_size=cfg.TRAIN.BATCH_SIZE,
                                     shuffle=cfg.DATASET.VAL_SHUFFLE,
                                     sampler=None,
                                     collate_fn=d_utils.make_collate_fn(randomize_n_views=cfg.DATASET.RANDOMIZE_N_VIEWS,
                                                                   min_n_views=cfg.DATASET.MIN_N_VIEWS,
                                                                   max_n_views=cfg.DATASET.MAX_N_VIEWS),
                                     num_workers=cfg.DATASET.NUM_WORKERS,
                                     worker_init_fn=d_utils.worker_init_fn,
                                     pin_memory=True)

    val_mpi_dataloader = DataLoader(val_mpi_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                    shuffle=cfg.DATASET.VAL_SHUFFLE,
                                    sampler=None,
                                    collate_fn=d_utils.make_collate_fn(randomize_n_views=cfg.DATASET.RANDOMIZE_N_VIEWS,
                                                                   min_n_views=cfg.DATASET.MIN_N_VIEWS,
                                                                   max_n_views=cfg.DATASET.MAX_N_VIEWS),
                                    num_workers=cfg.DATASET.NUM_WORKERS,
                                    worker_init_fn=d_utils.worker_init_fn,
                                    pin_memory=True)

    return train_dataloader, val_h36m_dataloader, val_mpi_dataloader