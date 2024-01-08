import open3d as o3d
import numpy as np
import tqdm
import os

root = '/home/panmingjie/gsnet_data/data_obj'

with open('filtered_obj_list.txt', 'r') as f:
    obj_names = f.read().splitlines()
    
obj_names = ['/home/panmingjie/gsnet_data/data_obj/PhoCaL/bottle/bottle_011/Scan/Simp.obj']
for obj_name in tqdm.tqdm(obj_names, total=len(obj_names)):
    obj_path = os.path.join(root, obj_name)
    ply_path = obj_path.replace('.obj', '.ply')
    
    # 将OBJ模型转换为点云
    mesh = o3d.io.read_triangle_mesh(obj_path)
    obj_pcd = mesh.sample_points_uniformly(number_of_points=10000)
    # print(obj_pcd.points.shape)
    # 保存点云为PLY文件
    o3d.io.write_point_cloud(ply_path, obj_pcd)

    # # 可视化点云
    o3d.visualization.draw_geometries([obj_pcd])
