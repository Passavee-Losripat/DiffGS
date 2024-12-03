import trimesh
import numpy as np
import os
import traceback
from multiprocessing import Pool
from fnmatch import fnmatch
import multiprocessing as mp
import json

def sample(arg):
    path, name = arg
    mesh = trimesh.load_mesh(os.path.join(path, name))

    num_points = 100000
    points = mesh.sample(num_points)

    point_cloud = trimesh.points.PointCloud(points)

    save_path = os.path.join(path, 'points3d.ply')
    point_cloud.export(save_path)


if __name__ == '__main__':

    shapene_folder = r'path/to/your/shapenet/folder'
    

    pattern = "*.obj"
    args = []
    for path, subdirs, files in os.walk(shapene_folder):
        for name in files:
            if fnmatch(name, pattern):
                args.append((path, name))

    print(f"{len(args)} left to be processed!")

    workers = 35
    pool = mp.Pool(workers)
    pool.map(sample, args)
    