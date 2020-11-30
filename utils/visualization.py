from utils.visualization_utils import *
from cfg import constants

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torch
import numpy as np

import trimesh
import pyrender
import neural_renderer as nr
import cv2

import os


def align_two_joints(gt_joints, pred_joints, opt_joints=None, data_type='human36m'):
    
    def centering_joints(joints):
        if data_type == 'human36m':
            sacrum_center = joints[:, 14]
        elif data_type == 'cmu':
            lpelvis, rpelvis = joints[:, 6].clone(), joints[:, 12].clone()
            sacrum_center = (lpelvis + rpelvis)/2
        
        joints_ = joints - sacrum_center.unsqueeze(1)

        return joints_

    gt_joints = centering_joints(gt_joints)
    
    flip = torch.tensor([1, 1, 1], 
        device=pred_joints.device, dtype=pred_joints.dtype)

    pred_joints = centering_joints(pred_joints)

    if opt_joints is not None:
        flip = torch.tensor([1, 1, 1], 
            device=opt_joints.device, dtype=opt_joints.dtype)
        opt_joints = opt_joints * flip
        opt_joints = centering_joints(opt_joints)

        return gt_joints, pred_joints, opt_joints

    return gt_joints, pred_joints


def get_segment_idx():

    segments = dict()
    segments['larm1'] = [13, 14]
    segments['larm2'] = [14, 15]
    segments['rarm1'] = [10, 11]
    segments['rarm2'] = [11, 12]
    segments['lleg1'] = [3, 4]
    segments['lleg2'] = [4, 5]
    segments['rleg1'] = [0, 1]
    segments['rleg2'] = [1, 2]
    segments['back1'] = [6, 7]
    segments['back2'] = [7, 8]

    return segments


def set_range(ax, offset=[0, 0, 0]):
    x, y, z = offset
    ax.set_xlim(-65 + x, 65 + x)
    ax.set_ylim(-65 + y, 65 + y)
    ax.set_zlim(-65 + z, 65 + z)


def plot_skeletal_model(joints, ax=None, show=False, conf=None, kind='mpii', data_type = 'human36m', joint_type='prediction'):

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.set_axis_off()
    ax.view_init(azim=-90, elev=-50)

    if joints.shape[-1] == 4:
        joints = joints[:, :-1]

    if type(joints) == torch.Tensor:
        joints = joints.detach().cpu().numpy()

    center = joints[14]
    set_range(ax, offset=center)

    x, y, z = np.split(joints, 3, axis=-1)

    parts = CONNECTIVITY_DICT[kind]
    colors = COLOR_DICT[data_type]

    if conf is None:
        conf = np.ones(joints.shape[0])

    for idxs, color in zip(parts, colors):
        part_x = [x[idx][0] for idx in idxs if conf[idx] > 0]
        part_y = [y[idx][0] for idx in idxs if conf[idx] > 0]
        part_z = [z[idx][0] for idx in idxs if conf[idx] > 0]

        _color = [col/255 for col in color]
        ax.plot(part_x, part_y, part_z, c=_color)

    j_type = 'Prediction' if joint_type=='prediction' else 'Ground Truth'
    ax.set_title(j_type)

    if show:
        plt.show()


def compare_two_joints(keypoints_3d_gt, keypoints_3d_pred, show=False, kind='mpii', data_type='human36m'):
    fig = plt.figure()
    ax_gt = fig.add_subplot(1, 2, 1, projection='3d')
    ax_pred = fig.add_subplot(1, 2, 2, projection='3d')

    if keypoints_3d_gt.size(-1) == 4:
        keypoints_3d_gt = keypoints_3d_gt.clone()[:, :-1]

    keypoints_3d_gt = keypoints_3d_gt.clone() / 10
    if keypoints_3d_pred.shape[0] == 49:
        keypoints_3d_pred = keypoints_3d_pred[25:][constants.SMPL_TO_H36].clone() * 100
    else:
        keypoints_3d_pred = keypoints_3d_pred * 1e2
    keypoints_3d_gt, keypoints_3d_pred = align_two_joints(keypoints_3d_gt[None], keypoints_3d_pred[None], data_type=data_type)
    
    plot_skeletal_model(keypoints_3d_gt[0], ax=ax_gt, show=False, kind=kind, data_type=data_type, joint_type='Ground Truth')
    plot_skeletal_model(keypoints_3d_pred[0], ax=ax_pred, show=False, kind=kind, data_type=data_type, joint_type='prediction')
    
    if show:
        plt.show()


def view_smpl_model(smplx_model, model_output, plotting_module='pyrender', plot_joints=False):
    
    vertices = model_output.vertices.detach().cpu().numpy().squeeze()
    joints = model_output.joints.detach().cpu().numpy().squeeze()

    if plotting_module == 'pyrender':
        import pyrender
        import trimesh
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, smplx_model.faces,
                                   vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        if plot_joints:
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        pyrender.Viewer(scene, use_raymond_lighting=True)


def render_joints_on_image(joints_3d, image, K, R, t):
    
    def project2D_by_camera_info(x3d, K, R, t, dist=None):
        """
        x3d : B X N X 4 numpy array
        K : 3 X 3 numpy array
        R : 3 X 3 numpy array
        t : 3 X 1 numpy array
        dist : 5 X 1 numpy array
        """

        conf_mask = np.ones((x3d.shape[0], x3d.shape[1]))

        x2d = np.zeros_like(x3d[:, :, :2])
        R2_criterion = np.zeros_like(x3d[:, :, 0])

        for J in range(x3d.shape[1]):
            """ J is joint index """
            
            x = np.dot(R, x3d[:, J].T) + t
            xp = x[:2] / x[2]

            if dist is not None:
                X2 = xp[0] * xp[0]
                Y2 = xp[1] * xp[1]
                XY = X2 * Y2
                R2 = X2 + Y2
                R4 = R2 * R2
                R6 = R4 * R2
                R2_criterion[:, J] = R2

                radial = 1.0 + dist[0] * R2 + dist[1] * R4 + dist[4] * R6
                tan_x = 2.0 * dist[2] * XY + dist[3] * (R2 + 2.0 * X2)
                tan_y = 2.0 * dist[3] * XY + dist[2] * (R2 + 2.0 * Y2)

                xp[0, :] = radial * xp[0, :] + tan_x
                xp[1, :] = radial * xp[1, :] + tan_y

            pt = np.dot(K[:2, :2], xp) + K[:2, 2:]
            x2d[:, J, :] = pt.T
        
        x2d = x2d.astype('int32')
        x_visible = np.logical_and(x2d[:, :, 0] >= 0, x2d[:, :, 0] < image.shape[0])
        y_visible = np.logical_and(x2d[:, :, 1] >= 0, x2d[:, :, 1] < image.shape[1])
        visible = np.logical_and(x_visible, y_visible)
        vis_mask = np.logical_and(visible, R2_criterion < 1.)
        mask = np.logical_and(conf_mask, vis_mask)

        return x2d, mask

    joints_2d, mask = project2D_by_camera_info(joints_3d, K, R, t)
    joints_2d[~mask] = 0.
    
    x = joints_2d[0, :, 0].astype('int32')
    y = joints_2d[0, :, 1].astype('int32')
    
    parts = CONNECTIVITY_DICT['mpii']
    colors = COLOR_DICT['human36m']
    
    for idxs, color in zip(parts, colors):
        # _color = [col/255 for col in color]
        visible = True
        for idx in idxs:
            if (x[idx] == 0 and y[idx] == 0):
                visible = False
        
        if visible:
            start = (x[idxs[0]], y[idxs[0]])
            end = (x[idxs[1]], y[idxs[1]])
            image = cv2.line(image, start, end, color, 3)
    
    return image


def render_smpl(vertices, faces, image, intrinsics, pose, transl, 
                alpha=1.0, filename='render_sample.png'):
    
    img_size = image.shape[-2]
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

    # Generate SMPL vertices mesh
    mesh = trimesh.Trimesh(vertices, faces)

    # Default rotation of SMPL body model
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
    scene.add(mesh, 'mesh')

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = pose
    camera_pose[:3, 3] = transl
    camera = pyrender.IntrinsicsCamera(fx=intrinsics[0, 0], fy=intrinsics[1, 1],
                                       cx=intrinsics[0, 2], cy=intrinsics[1, 2])
    scene.add(camera, pose=camera_pose)

    # Light information
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
    light_pose = np.eye(4)

    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=img_size, viewport_height=img_size, point_size=1.0)

    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    valid_mask = (rend_depth > 0)[:,:,None]
    
    color = color.astype(np.float32) / 255.0
    valid_mask = (rend_depth > 0)[:,:,None]
    output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image / 255.0

    cv2.imwrite(filename, 255 * output_img)


def generate_figure(camera, pred_output, body_model, images, gt, iters, save_org=False):
    from scipy.spatial.transform import Rotation as R

    betas = pred_output.betas.clone()
    body_pose = pred_output.body_pose.clone()
    glob_ori = pred_output.global_orient.clone()

    if body_pose.shape[-1] != 3:
        body_pose = torch.from_numpy(R.from_rotvec(body_pose.cpu().detach().numpy().reshape(-1, 3)).as_matrix())[None].cuda().float()
        glob_ori = torch.from_numpy(R.from_rotvec(glob_ori.cpu().detach().numpy().reshape(-1, 3)).as_matrix())[None].cuda().float()

    faces = body_model.faces
    
    gt = gt.detach().cpu().numpy()
    
    for cam_idx in range(4):
        image = images[0, cam_idx]

        # Get camera information
        camera_info = camera[cam_idx][0]
        pose = camera_info.R
        intrinsics = camera_info.K
        transl = (camera_info.t.reshape(3)) / 1e3
        transl[0] *= -1     # Adjust x-axis translation

        # Change body orientation so that camera matrix to be identical
        rot = torch.from_numpy(pose).to(device=glob_ori.device, dtype=glob_ori.dtype)
        glob_ori_R = rot @ glob_ori
        
        pred_output_ = body_model(betas=betas, body_pose=body_pose, global_orient=glob_ori_R, pose2rot=False)

        # Match and tranform keypoints gt and pred
        loc_gt = gt @ pose.T
        loc_pred_ = pred_output_.joints[:, 25:][:, constants.SMPL_TO_H36].detach().cpu().numpy() * 1e3
        loc_diff = loc_gt[:, 14] - loc_pred_[:, 14]
        loc_pred_ = loc_pred_ + loc_diff

        vertices = pred_output_.vertices[0].detach().cpu().numpy()
        vertices = vertices + loc_diff / 1e3

        if save_org:
            cv2.imwrite('demo_figs/render_sample_%03d_org_%d.png'%(iters, cam_idx+1), image[:, :, ::-1])
        filename = 'demo_figs/render_sample_%03d_%d.png'%(iters, cam_idx+1)
        
        image_org = image[:, :, ::-1].copy()
        render_smpl(vertices, faces, image[:, :, ::-1], intrinsics, np.eye(3), transl, filename=filename)