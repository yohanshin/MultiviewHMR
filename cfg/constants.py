"""
We create a superset of joints containing the OpenPose joints together with the ones that each dataset provides.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are the following:
"""

H36M_ROOT_PTH = 'data/dataset/human36m-raw/MuVHMR/'
MPI_INF_ROOT_PTH = 'data/dataset/mpi_inf_3dhp'

H36M_LABEL = 'H36M_label.npy'
H36M_SMPL_PARAMS = 'H36M_smpl.npy'
MPI_INF_LABEL = 'mpi_to_S17_v2.npy'
MPI_INF_SMPL_PARAMS = 'mpi_inf_smpl.npy'

BODY_SEGMENTS_IDX = {
    'larm1': [13, 14], 'larm2': [14, 15], 'rarm1': [10, 11], 'rarm2': [11, 12],
    'lleg1': [3, 4], 'lleg2': [4, 5], 'rleg1': [0, 1], 'rleg2': [1, 2],
    'back1': [6, 7], 'back2': [7, 8]
}

SMPL_TO_H36 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
J32_TO_J17 = [3, 2, 1, 6, 7, 8, 27, 26, 25, 17, 18, 19, 24, 15, 11, 12, 14]
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]

# Indices to get the 14 LSP joints from the ground truth joints
H36M_TO_J14 = H36M_TO_J17[:14]
J24_TO_J14 = J24_TO_J17[:14]

JOINT_REGRESSOR_H36M = 'data/dataset/SPIN/data/J_regressor_h36m.npy'

### SMPL COnstants

JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',
'Left Hip',
'Left Knee',
'Left Ankle',
'Right Wrist',
'Right Elbow',
'Right Shoulder',
'Left Shoulder',
'Left Elbow',
'Left Wrist',
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)', 
'Thorax (MPII)',
'Spine (H36M)',
'Jaw (H36M)', 
'Head (H36M)',
'Nose', 
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

SMPL_JOINT_MAP = {
'Left Hip', 'Right Hip', 'Spine 1 (Lower)', 'Left Knee', 'Right Knee', 'Spine 2 (Middle)',
'Left Ankle', 'Right Ankle', 'Spine 3 (Upper)' 'Left Foot', 'Right Foot',
'Neck', 'Left Shoulder (Inner)', 'Right Shoulder (Inner)', 'Head',
'Left Shoulder (Outer)', 'Right Shoulder (Outer)', 'Left Elbow', 'Right Elbow', 
'Left Wrist', 'Right Wrist', 'Left Hand', 'Right Hand'
}

JOINT_NAMES_H36M = [
'Right Ankle', 'Right Knee', 'Right Hip',
'Left Hip', 'Left Knee', 'Left Ankle',
'Right Wrist', 'Right Elbow', 'Right Shoulder', 
'Left Shoulder', 'Left Elbow', 'Left Wrist', 
'Neck (LSP)', 'Head (H36M)', 'Pelvis (MPII)', 'Spine (H36M)', 'Jaw (H36M)'
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
# From here H36M
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}
