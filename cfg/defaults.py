from yacs.config import CfgNode as CN
import torch

_C = CN()
_C.DEVICE = 'cuda'
_C.DTYPE = 'float32'
_C.EVAL = False

###################################################################################################
_C.MODEL = CN()
_C.MODEL.META_ARCH = 'VHMR'

###################################################################################################
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.ARCHITECTURE = "resnet50"
_C.MODEL.BACKBONE.INIT_WEIGHT = "data/pretrained/resnet50.pth"
_C.MODEL.BACKBONE.INPLANES = 64
_C.MODEL.BACKBONE.DECONV_LAYERS = 0
_C.MODEL.BACKBONE.DECONV_FILTERS = (256, 256, 256)
_C.MODEL.BACKBONE.DECONV_KERNELS = (4, 4, 4)

###################################################################################################
_C.MODEL.AGGREGATION = CN()
_C.MODEL.AGGREGATION.OUTPUT_CHANNELS = 256
_C.MODEL.AGGREGATION.VOLUME_SIZE = 16
_C.MODEL.AGGREGATION.CUBOID_SIDE = 2500.0
_C.MODEL.AGGREGATION.USE_GT_PELVIS = False
_C.MODEL.AGGREGATION.USE_TRIANGULATION = False
_C.MODEL.AGGREGATION.VOLUME_MULTIPLIER = 1.0
_C.MODEL.AGGREGATION.METHOD = 'softmax'

###################################################################################################
_C.MODEL.VHMR = CN()
_C.MODEL.VHMR.ENCODER_CHANNELS = (64, 256, 512, 512, 512)
_C.MODEL.VHMR.LARGE_MODEL = False
_C.MODEL.VHMR.INIT_WEIGHT = ""
_C.MODEL.VHMR.PREDICTION = 'rotmat'

###################################################################################################
_C.SMPL = CN()
_C.SMPL.ROOT_PTH = 'data/models'
_C.SMPL.TYPE = 'smpl'
_C.SMPL.PRIOR = 'gmm'
_C.SMPL.JOINT_REGRESSOR_TRAIN_EXTRA = 'data/models/J_regressor_extra.npy'
_C.SMPL.JOINT_REGRESSOR_H36M = 'data/models/J_regressor_h36m.npy'
_C.SMPL.MEAN_PARAMS = "data/dataset/SPIN/data/smpl_mean_params.npz"

###################################################################################################
_C.SMPLIFY = CN()
_C.SMPLIFY.LR = 1e-3
_C.SMPLIFY.MAXITERS = 10
_C.SMPLIFY.OPTIMIZER_TYPE = 'adam'
_C.SMPLIFY.PRECAL_FILENAME = 'smplify_prev.npy'

###################################################################################################
_C.TRAIN = CN()
_C.TRAIN.DISTRIBUTED = False
_C.TRAIN.NAME = 'exp'
_C.TRAIN.OPTIMIZER = 'Adam'
_C.TRAIN.SCHEDULER = False
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.LR_DEFAULTS = 1e-4
_C.TRAIN.LR_BACKBONE = 1e-5
_C.TRAIN.BETAS = (0.9, 0.999)
_C.TRAIN.EPOCH = 100
_C.TRAIN.INIT_WEIGHT = ''
_C.TRAIN.WEAKLY_SUPERVISE = False
_C.TRAIN.USE_PRECAL_BEFORE = 20

###################################################################################################
_C.TRAIN.LOGGER = CN()
_C.TRAIN.LOGGER.OUTPUT_PTH = 'logger'
_C.TRAIN.LOGGER.CHECKPOINT = 1
_C.TRAIN.LOGGER.WRITE_FREQ = 20

###################################################################################################
_C.LOSS = CN()
_C.LOSS.NUM_JOINTS = 17
_C.LOSS.RHO = 100
_C.LOSS.LW_KEYPOINTS_3D = 500.0
_C.LOSS.LW_KEYPOINTS_2D = 0.001
_C.LOSS.LW_MODEL_PARAMS_POSE = 2.0
_C.LOSS.LW_MODEL_PARAMS_VERTICES = 1.0
_C.LOSS.LW_MODEL_PARAMS_BETAS = 0.002

###################################################################################################
_C.DATASET = CN()
_C.DATASET.TYPE = 'human36m'
_C.DATASET.KIND = 'mpii'
_C.DATASET.ROOT_PTH = 'data/dataset/human36m/processed'
_C.DATASET.LABEL_PTH = "data/dataset/human36m/extra/h36m_to_S24.npy"
_C.DATASET.PRECAL_PTH = "data/precalculated/"
_C.DATASET.IMAGE_SHAPE = (224, 224)
_C.DATASET.SCALE_BBOX = 1.0
_C.DATASET.RETAIN_EVERY_N_FRAMES_IN_TRAIN = 1
_C.DATASET.RETAIN_EVERY_N_FRAMES_IN_TEST = 10
_C.DATASET.UNDISTORT_IMAGE = True
_C.DATASET.NORM_IMAGE = True
_C.DATASET.CROP_IMAGE = True
_C.DATASET.RANDOMIZE_N_VIEWS = False
_C.DATASET.TRAIN_SHUFFLE = True
_C.DATASET.VAL_SHUFFLE = False
_C.DATASET.MIN_N_VIEWS = 8
_C.DATASET.MAX_N_VIEWS = 31
_C.DATASET.NUM_WORKERS = 1
_C.DATASET.IGNORE_CAMERAS = ()