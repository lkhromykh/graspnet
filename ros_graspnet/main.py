import argparse
import pickle

import numpy as np
import open3d as o3d
from graspnet_model import GraspNetModel
from ur_env.remote import RemoteEnvClient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='checkpoint-kn.tar', help='Model checkpoint path')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
    parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
    return parser.parse_args()


def main(cfg):
    env = RemoteEnvClient(('192.168.1.136', 5555))
    net = GraspNetModel(
        checkpoint_path=cfg.checkpoint_path,
        num_point=cfg.num_point,
        num_view=cfg.num_view,
        collision_thresh=cfg.collision_thresh,
        voxel_size=cfg.voxel_size
    )
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # mock_obs = pickle.load(open('d405_working_obs.pkl', 'rb'))
    try:
        while True:
            obs = env_reset(env).observation
            gg, cloud, action = net(obs)
            o3d.visualization.draw_geometries([coord, cloud] + gg[:1].to_open3d_geometry_list())
            print("Execute action?")
            if input() != "y":
                continue
            action[-2] = 1
            ts = env.step(action)
            print('Is object detected: ', ts.observation['gripper_is_obj_detected'])
    except KeyboardInterrupt:
        env.close()


def env_reset(env_):
    # INIT_ACTION = [-0.4, -0.06, 0.4, 0.738, 0.6744, 0, -0, 0, 0]
    # tcp = ts.observation.copy()['tcp_pose']
    # action = np.zeros(9)
    # action[:7] = tcp
    # action[2] += 0.1
    # return env_.step(action)
    return env_.reset()


if __name__ == '__main__':
    args = parse_args()
    main(args)
