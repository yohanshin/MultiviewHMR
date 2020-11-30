import numpy as np
import cv2
from PIL import Image

import torch

IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

def crop_image(image, bbox):
    """Crops area from image specified as bbox. Always returns area of size as bbox filling missing parts with zeros
    Args:
        image numpy array of shape (height, width, 3): input image
        bbox tuple of size 4: input bbox (left, upper, right, lower)

    Returns:
        cropped_image numpy array of shape (height, width, 3): resulting cropped image

    """

    image_pil = Image.fromarray(image)
    image_pil = image_pil.crop(bbox)

    return np.asarray(image_pil)


def resize_image(image, shape):
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)

def normalize_image(image):
    """Normalizes image using ImageNet mean and std

    Args:
        image numpy array of shape (h, w, 3): image

    Returns normalized_image numpy array of shape (h, w, 3): normalized image
    """
    return (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def denormalize_image(image):
    """Reverse to normalize_image() function"""
    return np.clip(255.0 * (image * IMAGENET_STD + IMAGENET_MEAN), 0, 255)


# SPIN utils
def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1

def crop_SPIN(img, center, scale, res):
    """Crop image according to the supplied bounding box."""

    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1
    
    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], 
                                                        old_x[0]:old_x[1]]    
    
    return new_img


def get_square_bbox_SPIN(center, scale, res_bf, res_af):
    """Crop image according to the supplied bounding box."""

    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res_af, invert=1))-1
    # Bottom right point
    br = np.array(transform([res_af[0]+1, 
                             res_af[1]+1], center, scale, res_af, invert=1))-1

    old_x = max(0, ul[0]), min(res_bf[1], br[0])
    old_y = max(0, ul[1]), min(res_bf[0], br[1])
    
    return np.array((old_x[0], old_y[0], old_x[1], old_y[1]))


def get_square_bbox(bbox):
    """Makes square bbox from any bbox by stretching of minimal length side

    Args:
        bbox tuple of size 4: input bbox (left, upper, right, lower)

    Returns:
        bbox: tuple of size 4:  resulting square bbox (left, upper, right, lower)
    """

    left, upper, right, lower = bbox
    width, height = right - left, lower - upper

    if width > height:
        y_center = (upper + lower) // 2
        upper = y_center - width // 2
        lower = upper + width
    else:
        x_center = (left + right) // 2
        left = x_center - height // 2
        right = left + height

    return left, upper, right, lower


def scale_bbox(bbox, scale):
    left, upper, right, lower = bbox
    width, height = right - left, lower - upper

    x_center, y_center = (right + left) // 2, (lower + upper) // 2
    new_width, new_height = int(scale * width), int(scale * height)

    new_left = x_center - new_width // 2
    new_right = new_left + new_width

    new_upper = y_center - new_height // 2
    new_lower = new_upper + new_height

    return new_left, new_upper, new_right, new_lower


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def image_batch_to_numpy(image_batch):
    image_batch = to_numpy(image_batch)
    image_batch = np.transpose(image_batch, (0, 2, 3, 1)) # BxCxHxW -> BxHxWxC
    return image_batch


def image_batch_to_torch(image_batch):
    image_batch = np.transpose(image_batch, (0, 3, 1, 2)) # BxHxWxC -> BxCxHxW
    image_batch = to_torch(image_batch).float()
    return image_batch


def undistort_image(image, camera):
    h, w = image.shape[:2]
                
    fx, fy = camera.K[0, 0], camera.K[1, 1]
    cx, cy = camera.K[0, 2], camera.K[1, 2]

    grid_x = (np.arange(w, dtype=np.float32) - cx) / fx
    grid_y = (np.arange(h, dtype=np.float32) - cy) / fy
    meshgrid = np.stack(np.meshgrid(grid_x, grid_y), axis=2).reshape(-1, 2)

    # distort meshgrid points
    k = camera.dist[:3].copy(); k[2] = camera.dist[-1]
    p = camera.dist[2:4].copy()

    r2 = meshgrid[:, 0] ** 2 + meshgrid[:, 1] ** 2
    radial = meshgrid * (1 + k[0] * r2 + k[1] * r2**2 + k[2] * r2**3).reshape(-1, 1)
    tangential_1 = p.reshape(1, 2) * np.broadcast_to(meshgrid[:, 0:1] * meshgrid[:, 1:2], (len(meshgrid), 2))
    tangential_2 = p[::-1].reshape(1, 2) * (meshgrid**2 + np.broadcast_to(r2.reshape(-1, 1), (len(meshgrid), 2)))

    meshgrid = radial + tangential_1 + tangential_2

    # move back to screen coordinates
    meshgrid *= np.array([fx, fy]).reshape(1, 2)
    meshgrid += np.array([cx, cy]).reshape(1, 2)

    # cache (save) distortion maps
    meshgrid = cv2.convertMaps(meshgrid.reshape((h, w, 2)), None, cv2.CV_16SC2)

    image_undistorted = cv2.remap(image, *meshgrid, cv2.INTER_CUBIC)

    return image_undistorted