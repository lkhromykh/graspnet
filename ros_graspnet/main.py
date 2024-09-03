import argparse

import open3d as o3d
from graspnet_model import GraspNetModel
from ur_env.remote import RemoteEnvClient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
    parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
    return parser.parse_args()


def main(cfg):
    env = RemoteEnvClient(('0.tcp.eu.ngrok.io', 11884))
    net = GraspNetModel(
        checkpoint_path=cfg.checkpoint_path,
        num_point=cfg.num_point,
        num_view=cfg.num_view,
        collision_thresh=cfg.collision_thresh,
        voxel_size=cfg.voxel_size
    )
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
    try:
        while True:
            input()
            ts = env.reset()
            gg, cloud, action = net(ts.observation)
            o3d.visualization.draw_geometries([coord, cloud] + gg[:10].to_open3d_geometry_list())
            breakpoint()
    except KeyboardInterrupt:
        env.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
