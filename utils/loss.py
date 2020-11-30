from cfg import constants
from utils.multiview import *
from utils.conversion import *
from utils.pose_utils import *

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

DEFAULT_DTYPE = torch.float

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re


class MaxMixturePrior(nn.Module):

    def __init__(self, prior_folder='data/dataset/SPIN/data',
                 num_gaussians=8, device='cuda', dtype=DEFAULT_DTYPE, epsilon=1e-16,
                 use_merged=True,
                 **kwargs):
        super(MaxMixturePrior, self).__init__()

        if dtype == DEFAULT_DTYPE:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            print('Unknown float type {}, exiting!'.format(dtype))
            sys.exit(-1)

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)

        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            print('The path to the mixture prior "{}"'.format(full_gmm_fn) +
                  ' does not exist, exiting!')
            sys.exit(-1)

        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            print('Unknown type for the prior: {}, exiting!'.format(type(gmm)))
            sys.exit(-1)

        self.register_buffer('means', torch.tensor(means, dtype=dtype))

        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)

        self.register_buffer('precisions',
                             torch.tensor(precisions, dtype=dtype))

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const *
                                                   (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon)
                    for cov in covs]
        self.register_buffer('cov_dets',
                             torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]
        self.to(device=device)

    def get_mean(self):
        ''' Returns the mean of the mixture '''
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose, betas):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic - \
            torch.log(self.nll_weights)

        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose, betas, *args, **kwargs):
        ''' Create graph operation for negative log-likelihood calculation
        '''
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b',
                                              [curr_loglikelihood,
                                               diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term +
                                         self.random_var_dim *
                                         self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)

        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, pose, betas):
        if self.use_merged:
            return self.merged_log_likelihood(pose, betas)
        else:
            return self.log_likelihood(pose, betas)


class LossFunction():
    def __init__(self, 
                 model_type,
                 num_joints=17,
                 data_type='human36m',
                 device='cuda',
                 dtype=torch.float32,
                 rho=100,
                 lw_keypoints_3d=1,
                 lw_keypoints_2d=1,
                 lw_model_params_pose=5,
                 lw_model_params_betas=5,
                 lw_model_params_vertices=5,
                 **kwargs):
        
        self.J_regressor = torch.from_numpy(np.load(constants.JOINT_REGRESSOR_H36M)).float()
        self.num_joints = num_joints
        self.model_type = model_type
        self.data_type = data_type
        self.rho = rho
        self.device = device

        self.criterion_keypoints = nn.MSELoss(reduction='none').to(device)
        self.criterion_regr = nn.MSELoss(reduction='none').to(device)
        self.criterion_pprior = MaxMixturePrior(dtype=dtype)

        self.lw_keypoints_3d = lw_keypoints_3d
        self.lw_keypoints_2d = lw_keypoints_2d
        self.lw_model_params_pose = lw_model_params_pose
        self.lw_model_params_betas = lw_model_params_betas
        self.lw_model_params_vertices = lw_model_params_vertices


    def __call__(self, pred_output, keypoints_3d_gt, proj_matricies, print_log=False, opt_output=None, has_smpl=None, total_loss=0):
        J_regressor = self.J_regressor[None, :].expand(keypoints_3d_gt.shape[0], -1, -1).to(self.device)
        
        # Calculate 3D and 2D keypoints loss
        keypoints_3d_pred = torch.matmul(J_regressor, pred_output.vertices)
        keypoints_3d_pred = keypoints_3d_pred[:, constants.H36M_TO_J17, :]
        keypoints_3d_gt = keypoints_3d_gt / 1e3
        keypoints_3d_loss = self.keypoints_3d_loss(keypoints_3d_gt, keypoints_3d_pred)
        keypoints_2d_loss = self.keypoints_2d_loss(keypoints_3d_gt, keypoints_3d_pred, proj_matricies)
        
        loss_dict = {'3D_loss': keypoints_3d_loss, '2D_loss': keypoints_2d_loss}
        
        # Calculate SMPL parameters loss if weakly supervising
        if opt_output is not None:
            keypoints_3d_opt = torch.matmul(J_regressor, opt_output.vertices)
            keypoints_3d_opt = keypoints_3d_opt[:, constants.H36M_TO_J17, :]
            batch_error_3d_pred, batch_error_3d_opt = \
                self.keypoints_3d_error_in_batch(keypoints_3d_gt, keypoints_3d_pred, keypoints_3d_opt)
            
            has_smpl = torch.Tensor(has_smpl).to(self.device).bool()
            model_params_loss = self.model_params_loss(pred_output, opt_output)
            model_params_loss[~has_smpl] = model_params_loss[~has_smpl] * 0
            model_params_loss = model_params_loss.mean()
            prior_loss = (self.pose_prior_loss(pred_output) + 5 * self.shape_prior_loss(pred_output)) * 5e-4
            prior_loss[has_smpl] = prior_loss[has_smpl] * 0
            
            prior_loss = prior_loss.mean()
            model_params_loss = model_params_loss + prior_loss
            
            loss_dict.update({'SMPL_params_loss': model_params_loss})
        
        else:
            batch_error_3d_pred, _ = \
                self.keypoints_3d_error_in_batch(keypoints_3d_gt, keypoints_3d_pred)
        
        for key, value in loss_dict.items():
            total_loss = total_loss + value
            loss_dict[key] = value.item()

        if print_log:
            # Calculate MPJPE and Reconstruction error for log
            mpjpe = batch_error_3d_pred.mean().item() * 1e3
            recone = self.reconstruction_error_in_batch(keypoints_3d_gt, keypoints_3d_pred)  * 1e3
            loss_dict.update({'MPJPE': mpjpe, 'RECONE': recone})
            
            if opt_output is not None:
                mpjpe_opt = batch_error_3d_opt.mean().item() * 1e3
                loss_dict.update({'MPJPE_(OPT)': mpjpe_opt})
        
        return total_loss, loss_dict

    def eval(self, pred_output, keypoints_3d_gt, ):
        num_joints_ = self.num_joints
        self.num_joints = 14
        
        J_regressor = self.J_regressor[None, :].expand(keypoints_3d_gt.shape[0], -1, -1).to(self.device)
        keypoints_3d_pred = torch.matmul(J_regressor, pred_output.vertices)
        keypoints_3d_pred = keypoints_3d_pred[:, constants.H36M_TO_J17, :]
        keypoints_3d_gt = keypoints_3d_gt / 1e3
        
        mpjpe, _ = self.keypoints_3d_error_in_batch(keypoints_3d_gt, keypoints_3d_pred)
        recone = self.reconstruction_error_in_batch(keypoints_3d_gt, keypoints_3d_pred)

        self.num_joints = num_joints_

        return mpjpe.mean().item() * 1e3, recone.mean() * 1e3

    def centering_joints(self, joints):
        if self.data_type == 'cmu':
            lpelvis, rpelvis = joints[:, 6].clone(), joints[:, 12].clone()
            sacrum_center = (lpelvis + rpelvis)/2
        # if self.data_type == 'human36m':
        else:
            sacrum_center = joints[:, 14]
            # sacrum_center = joints[:, 6]
        
        joints_ = joints - sacrum_center.unsqueeze(1)

        return joints_

    def align_two_joints(self, gt_joints, pred_joints, opt_joints=None):
        
        gt_joints = self.centering_joints(gt_joints)
        pred_joints = self.centering_joints(pred_joints)

        if opt_joints is not None:
            opt_joints = self.centering_joints(opt_joints)

        return gt_joints, pred_joints, opt_joints


    def robustifier(self, value):
        dist = torch.div(value**2, value**2 + self.rho ** 2)
        return self.rho ** 2 * dist


    def keypoints_3d_loss(self, keypoints_3d_gt, keypoints_3d_pred, use_robustifier=False):
        # keypoints_3d_pred = keypoints_3d_pred[:, 25:][:, constants.SMPL_TO_H36]
        keypoints_3d_gt = keypoints_3d_gt

        if keypoints_3d_gt.size(-1) == 4:
            conf = keypoints_3d_gt[:, :, -1].unsqueeze(-1)
            keypoints_3d_gt = keypoints_3d_gt[:, :, :-1]
        else:
            conf = torch.ones(*keypoints_3d_gt.shape[:-1]).unsqueeze(-1)
            conf = conf.to(device=keypoints_3d_gt.device, dtype=keypoints_3d_gt.dtype)
        
        # align two keypoints
        keypoints_3d_gt, keypoints_3d_pred, _ = self.align_two_joints(keypoints_3d_gt, keypoints_3d_pred)
        keypoints_3d_gt = keypoints_3d_gt[:, :self.num_joints]
        keypoints_3d_pred = keypoints_3d_pred[:, :self.num_joints]
        
        if use_robustifier:
            joint_dist = self.robustifier(keypoints_3d_gt - keypoints_3d_pred)
            loss = torch.sum(joint_dist * conf ** 2) * self.lw_keypoints_3d
        else:
            loss = (self.criterion_keypoints(keypoints_3d_pred, keypoints_3d_gt)  * conf ** 2).mean() * self.lw_keypoints_3d
        
        return loss


    def keypoints_3d_error_in_batch(self, keypoints_3d_gt, keypoints_3d_pred, keypoints_3d_opt=None):
        # if self.data_type == 'human36m':
        
        # keypoints_3d_pred = 1000 * keypoints_3d_pred[:, 25:][:, constants.SMPL_TO_H36].clone().detach()
        # if keypoints_3d_opt is not None:
        #     keypoints_3d_opt = 1000 * keypoints_3d_opt[:, 25:][:, constants.SMPL_TO_H36].clone().detach()

        if keypoints_3d_gt.size(-1) == 4:
            keypoints_3d_gt = keypoints_3d_gt[:, :, :-1].detach()
        
        keypoints_3d_gt, keypoints_3d_pred, keypoints_3d_opt =\
            self.align_two_joints(keypoints_3d_gt, keypoints_3d_pred, keypoints_3d_opt)
        keypoints_3d_gt = keypoints_3d_gt[:, :self.num_joints]
        keypoints_3d_pred = keypoints_3d_pred[:, :self.num_joints]
        pred_error_in_batch = torch.sqrt(((keypoints_3d_pred - keypoints_3d_gt)**2).sum(-1)).mean(1)
        
        if keypoints_3d_opt is not None:
            keypoints_3d_opt = keypoints_3d_opt[:, :self.num_joints]
            opt_error_in_batch = torch.sqrt(((keypoints_3d_opt - keypoints_3d_gt)**2).sum(-1)).mean(1)
            return pred_error_in_batch.detach(), opt_error_in_batch.detach()
        else:
            return pred_error_in_batch.detach(), None


    def reconstruction_error_in_batch(self, keypoints_3d_gt, keypoints_3d_pred):
        # keypoints_3d_pred = keypoints_3d_pred[:, 25:][:, constants.SMPL_TO_H36].clone().detach()

        if keypoints_3d_gt.size(-1) == 4:
            conf = keypoints_3d_gt[:, :, -1:]
            keypoints_3d_pred = keypoints_3d_pred * conf
            keypoints_3d_gt = keypoints_3d_gt[:, :, :-1].clone().detach()
        
        keypoints_3d_gt, keypoints_3d_pred, _ = self.align_two_joints(keypoints_3d_gt, keypoints_3d_pred)

        gt = keypoints_3d_gt[:, :self.num_joints].cpu().detach().numpy()
        pred = keypoints_3d_pred[:, :self.num_joints].cpu().detach().numpy()

        pred_hat = np.zeros_like(pred)
        batch_size = keypoints_3d_gt.shape[0]
        for b in range(batch_size):
            pred_hat[b] = compute_similarity_transform(pred[b], gt[b])
        
        error = np.sqrt( ((pred_hat - gt)**2).sum(axis=-1)).mean(axis=-1)
        
        return error.mean()


    def keypoints_2d_loss(self, keypoints_gt, keypoints_3d_pred, proj_matricies, has_2d_gt=False):

        (batch_size, n_views), n_joints = proj_matricies.shape[:2], keypoints_gt.shape[-2]
        device, dtype = keypoints_gt.device, keypoints_gt.dtype
        
        dim = 2 if has_2d_gt else 3
        if keypoints_gt.size(-1) != dim:
            conf = keypoints_gt[:, :, -1]
            keypoints_gt = keypoints_gt[:, :, :-1]
        else:
            conf = torch.ones(*keypoints_gt.shape[:-1]).unsqueeze(-1).unsqueeze(1)
            conf = conf.to(device=device, dtype=dtype)
        
        keypoints_gt, keypoints_3d_pred, _ = self.align_two_joints(keypoints_gt, keypoints_3d_pred)
        
        keypoints_2d_gt = torch.zeros((batch_size, n_views, n_joints, 2)).to(device=device, dtype=dtype)
        keypoints_2d_pred = torch.zeros((batch_size, n_views, n_joints, 2)).to(device=device, dtype=dtype)
        
        for camera in range(n_views):
            for b in range(batch_size):
                keypoints_2d_gt[b, camera] = project_3d_points_to_image_plane_without_distortion(proj_matricies[b, camera], keypoints_gt[b])
                keypoints_2d_pred[b, camera] = project_3d_points_to_image_plane_without_distortion(proj_matricies[b, camera], keypoints_3d_pred[b])
        
        loss = (self.criterion_keypoints(keypoints_2d_pred * 1e3, keypoints_2d_gt * 1e3) * conf ** 2).mean() * self.lw_keypoints_2d
        
        return loss


    def model_params_loss(self, pred_output, opt_output):
        pred_betas, pred_pose, pred_global_orient, pred_vertices = \
            pred_output.betas, pred_output.body_pose, pred_output.global_orient, pred_output.vertices

        opt_betas, opt_pose, opt_global_orient, opt_vertices = \
            opt_output.betas, opt_output.body_pose, opt_output.global_orient, opt_output.vertices
        
        betas_loss = self.criterion_regr(pred_betas, opt_betas).mean(dim=-1) * self.lw_model_params_betas
        # vertices_loss = self.criterion_regr(pred_vertices, opt_vertices).mean(dim=(-2, -1)) * self.lw_model_params_vertices
        
        mean_dim = (-1, -2, -3)
        if pred_pose.shape != opt_pose.shape:
            batch_size = opt_pose.shape[0]
            opt_pose = batch_rodrigues(opt_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
            opt_global_orient = batch_rodrigues(opt_global_orient).unsqueeze(1)
        
        pose_loss = self.criterion_regr(pred_pose, opt_pose).mean(dim=mean_dim) * self.lw_model_params_pose
        global_orient_loss = self.criterion_regr(pred_global_orient, opt_global_orient).mean(dim=mean_dim) * self.lw_model_params_pose

        # return betas_loss + pose_loss + global_orient_loss + vertices_loss
        return betas_loss + pose_loss + global_orient_loss


    def pose_prior_loss(self, pred_output):
        pred_pose, pred_betas = pred_output.body_pose, pred_output.betas
        batch_size, device, dtype = pred_pose.shape[0], pred_pose.device, pred_pose.dtype
        transl = torch.tensor([0,0,1], dtype=dtype,device=device).reshape(1, 3, 1).expand(batch_size * 23, -1, -1)
        pred_pose_hom = torch.cat([pred_pose.reshape(-1, 3, 3), transl], dim=-1)
        pred_pose_euler = rotation_matrix_to_angle_axis(pred_pose_hom).reshape(batch_size, -1)
        is_nan = torch.isnan(pred_pose_euler)
        pred_pose_euler[is_nan] = pred_pose_euler[is_nan] * 0
        
        return self.criterion_pprior(pred_pose_euler, pred_betas)

    def shape_prior_loss(self, pred_output):
        pred_betas = pred_output.betas
        return torch.sum((pred_betas ** 2), dim=-1)

    
def build_loss_function(cfg):
    
    return LossFunction(model_type=cfg.MODEL.META_ARCH,
                        data_type=cfg.DATASET.TYPE, 
                        num_joints=cfg.LOSS.NUM_JOINTS,
                        rho=cfg.LOSS.RHO,
                        lw_keypoints_3d=cfg.LOSS.LW_KEYPOINTS_3D,
                        lw_keypoints_2d=cfg.LOSS.LW_KEYPOINTS_2D,
                        lw_model_params_betas=cfg.LOSS.LW_MODEL_PARAMS_BETAS,
                        lw_model_params_pose=cfg.LOSS.LW_MODEL_PARAMS_POSE,
                        lw_model_params_vertices=cfg.LOSS.LW_MODEL_PARAMS_VERTICES)