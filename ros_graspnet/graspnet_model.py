"""demo.py"""
import os
import sys
import dataclasses
from typing import Dict, Optional, Tuple
import logging
logging.getLogger().setLevel(logging.INFO)

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(ROOT_DIR, '../graspnet-baseline')
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

Array = np.ndarray


@dataclasses.dataclass
class GraspNetModel:

    checkpoint_path: str
    num_point: int = 20000
    num_view: int = 300
    collision_thresh: float = 0.01
    voxel_size: float = 0.01

    def __post_init__(self):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._net = get_net(self.checkpoint_path, self.num_view, self._device)
        self._retry = 0

    def __call__(self, obs: Dict[str, Array]):
        cloud_sampled, color_sampled, o3d_cloud = get_and_process_data(
            obs['image'], obs['depth'], obs['point_cloud'], self.num_point
        )
        pcds = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        pcds = pcds.to(self._device)
        with torch.no_grad():
            end_points = self._net({'point_clouds': pcds, 'cloud_colors': color_sampled})
            grasp_preds = pred_decode(end_points)
            gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        gg = collision_detection(gg, np.asarray(o3d_cloud.points), self.collision_thresh, self.voxel_size)
        gg.nms()
        if len(gg) == 0 and self._retry < 2:
            self._retry += 1
            logging.info(f'Found 0 grasps. Trying again {self._retry}')
            gg, o3d_cloud, action = self(obs)
        self._retry = 0
        gg.sort_by_score()
        gg, o3d_cloud = transform_fn(gg, o3d_cloud, obs['optical_frame'])
        action = infer_action(gg)
        return gg, o3d_cloud, action


def get_net(checkpoint_path, num_view, device):
    # Init the model
    net = GraspNet(
        input_feature_dim=0,
        num_view=num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False
    )
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    logging.info('-> loaded checkpoint %s (epoch: %d)' % (checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net


def get_and_process_data(color: Array,
                         depth: Array,
                         cloud: Array,
                         num_point: int,
                         workspace_mask: Optional[Array] = None
                         ) -> Tuple[Array, Array, o3d.geometry.PointCloud]:
    assert color.dtype == np.uint8
    assert depth.dtype == np.uint16
    assert color.ndim == 3
    assert depth.ndim == 2
    assert cloud.ndim == 3
    # assert color.shape[:2] == depth.shape == cloud.shape[:2]

    color = color.astype(np.float32) / 255.
    mask = (0 < depth) & (depth < 1000)
    if workspace_mask:
        mask = np.logical_and(mask, workspace_mask)
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    o3d_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    return cloud_sampled, color_sampled, o3d_cloud


def collision_detection(
        gg: GraspGroup,
        cloud: o3d.geometry.PointCloud,
        collision_thresh: float = 0.01,
        voxel_size: float = 0.01
) -> GraspGroup:
    if collision_thresh <= 0 or voxel_size <= 0:
        return gg
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    return gg[~collision_mask]


def transform_fn(gg: GraspGroup, cloud, transform: Array):
    pos, quat = np.split(transform, [3])
    rmat = Rotation.from_quat(quat).as_matrix()
    rigid = np.eye(4)
    rigid[:3, :3] = rmat
    rigid[:3, 3] = pos
    return gg.transform(rigid), cloud.transform(rigid)


# todo: handle rotation opposite rotations of z-axis
def infer_action(gg: GraspGroup) -> Array:
    # pick the best
    g = gg[0]
    trans = g.translation
    rot = g.rotation_matrix
    rot = Rotation.from_matrix(rot)
    rot_adj = Rotation.from_matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    rot = (rot * rot_adj).as_quat()
    return np.r_[trans, rot, 0, 0]
