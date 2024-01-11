from glob import glob

import os
import cv2
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

DEBUG=False


def get_model_grasps(datapath):
    ''' Author: chenxi-wang
    Load grasp labels from .npz files.
    '''
    label = np.load(datapath)
    points = label['points']
    offsets = label['offsets']
    scores = label['scores']
    collision = label['collision']
    return points, offsets, scores, collision

def generate_views(N, phi=(np.sqrt(5)-1)/2, center=np.zeros(3, dtype=np.float32), R=1):
    idxs = np.arange(N, dtype=np.float32)
    Z = (2 * idxs + 1) / N - 1
    X = np.sqrt(1 - Z**2) * np.cos(2 * idxs * np.pi * phi)
    Y = np.sqrt(1 - Z**2) * np.sin(2 * idxs * np.pi * phi)
    views = np.stack([X,Y,Z], axis=1)
    views = R * np.array(views) + center
    return views

def viewpoint_params_to_matrix(towards, angle):
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    if np.linalg.norm(axis_y) == 0:
        axis_y = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle), np.cos(angle)]])
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2.dot(R1)
    return matrix.astype(np.float32)

def get_view_angles(V=300, A=12):
    views = generate_views(V)
    angles = np.arange(0, np.pi, np.pi / A)

    points_view = np.zeros((V, A, 3, 3))
    for i, view in enumerate(views):
        for j, angle in enumerate(angles):
            points_view[i, j, :, :] = viewpoint_params_to_matrix(-view, angle)

    points_view = points_view.astype(np.float32)
    return points_view



def quaternion_wxyz_to_rotation_matrix(quaternion_wxyz):
    # Create a Rotation object from the quaternion
    r = Rotation.from_quat(quaternion_wxyz[1:]+[quaternion_wxyz[0]])
    # Convert to rotation matrix
    rotation_matrix = r.as_matrix()
    return rotation_matrix


#convert depth to point cloud
def depth2pc(depths, intrinsics, scale=4):
    fx, fy, cx, cy = intrinsics

    w, h = depths.shape[1], depths.shape[0]
    new_w, new_h = int(w/scale), int(h/scale)
    fx = fx / scale
    fy = fy / scale
    cx = cx / scale
    cy = cy / scale
    depths = cv2.resize(depths, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    mask = (depths > 0) & (depths < 2)

    points_z = depths
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # remove outlier
    points = np.stack([points_x, points_y, points_z], axis=-1)
    return points[mask]


def prase_meta(meta_file):
    with open(meta_file, "r") as f:
        meta = json.load(f)
    return meta


def prase_label_path(obj_meta):
    global labels_root 
    obj_path = os.path.dirname(obj_meta["meta"]["instance_path"])
    label_path = os.path.join(labels_root, obj_path, "Aligned.npz")
    return label_path


def process(depth_file, view_angles):
    depth_map = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    meta_file = depth_file.replace("depth.exr", "meta.json")
    meta = prase_meta(meta_file)
    collision_mask_file = depth_file.replace("depth.exr", "collision_label.npz")
    print(collision_mask_file)

    camera_intrinsic = meta["camera"]["intrinsics"]
    fx, fy, cx, cy = camera_intrinsic["fx"], camera_intrinsic["fy"], camera_intrinsic["cx"], camera_intrinsic["cy"]
    scene_points = depth2pc(depth_map[...,0], [fx, fy, cx, cy], scale=8)
    scene_points = torch.Tensor(scene_points).cuda()
    grasp_dict = {}


    # Load and preprocess all objects
    obj_pose = []
    obj_points_at_scene = []
    obj_offsets = []
    obj_scores = []
    obj_collision = []
    for obj_meta in meta["objects"].values():
        translation = obj_meta["translation"]
        rotation_matrix = quaternion_wxyz_to_rotation_matrix(obj_meta["quaternion_wxyz"])
        grasp_label_path = prase_label_path(obj_meta)
        assert os.path.exists(grasp_label_path), 'File lost: %s'%grasp_label_path

        # object pose at scene-frame
        pose = np.eye(4)
        pose[:3, :3] = rotation_matrix
        pose[:3, 3] = translation
        pose = torch.Tensor(pose).to(torch.float16).cuda()
        obj_pose.append(pose)

        points, offsets, scores, collision = get_model_grasps(grasp_label_path)

        # transform points from object-frame to scene-frame
        points_homn = torch.ones((points.shape[0], 4), dtype=torch.float16).cuda()
        points_homn[:,:3] = torch.Tensor(points)
        points_at_scene = (pose @ points_homn.T).T[:,:3].to(torch.float16)

        obj_points_at_scene.append(points_at_scene)
        obj_offsets.append(offsets)
        obj_scores.append(scores)
        obj_collision.append(collision)
        


    obj_pose = torch.stack(obj_pose)

    all_obj_points = torch.cat(obj_points_at_scene, dim=0)
    scene_points = torch.cat([scene_points, all_obj_points]).to(torch.float16)
    view_angles = view_angles.to(torch.float16)

    for i, obj_meta in tqdm(enumerate(meta["objects"].values()), desc="Object Loop", leave=False):
        pose = obj_pose[i]
        obj_points = obj_points_at_scene[i]
        offsets = torch.Tensor(obj_offsets[i]).to(torch.float16).cuda()
        scores = torch.Tensor(obj_scores[i]).to(torch.float16).cuda()
        collision = torch.Tensor(obj_collision[i]).to(torch.bool).cuda()

        View_at_scene = pose[:3,:3].unsqueeze(0).unsqueeze(0) @ view_angles

        height = 0.02
        depth_base = 0.02
        finger_width = 0.01
        finger_length = 0.06
        outlier=0.05    
        empty_thresh=10

        approach_dist=0.03


        ## 仅对物体周围区域做碰撞检测，周围区域 = 最小外接矩形 + outlier
        xmin, xmax = obj_points[:,0].min(), obj_points[:,0].max()
        ymin, ymax = obj_points[:,1].min(), obj_points[:,1].max()
        zmin, zmax = obj_points[:,2].min(), obj_points[:,2].max()
        xlim = ((scene_points[:,0] > xmin-outlier) & (scene_points[:,0] < xmax+outlier))
        ylim = ((scene_points[:,1] > ymin-outlier) & (scene_points[:,1] < ymax+outlier))
        zlim = ((scene_points[:,2] > zmin-outlier) & (scene_points[:,2] < zmax+outlier))
        workspace = scene_points[xlim & ylim & zlim]
        
        target = (workspace[None,:,:] - obj_points[:,None,:])
        targets = torch.zeros((obj_points.shape[0], 12, workspace.shape[0], 3), dtype=torch.float16).cuda()
        
        collision_label = torch.zeros((obj_points.shape[0], 300, 12, 4), dtype=torch.bool).cuda()
        mask1 = torch.zeros((obj_points.shape[0], 12,  workspace.shape[0]), dtype=torch.bool).cuda()
        mask2 = torch.zeros((obj_points.shape[0], 12,  workspace.shape[0]), dtype=torch.bool).cuda()
        mask3 = torch.zeros((obj_points.shape[0], 12,  workspace.shape[0]), dtype=torch.bool).cuda()
        mask4 = torch.zeros((obj_points.shape[0], 12,  workspace.shape[0]), dtype=torch.bool).cuda()
        mask5 = torch.zeros((obj_points.shape[0], 12,  workspace.shape[0]), dtype=torch.bool).cuda()
        mask6 = torch.zeros((obj_points.shape[0], 12,  workspace.shape[0]), dtype=torch.bool).cuda()
        mask7 = torch.zeros((obj_points.shape[0], 12,  workspace.shape[0]), dtype=torch.bool).cuda()

        grasp_depths = offsets[:,:,:,:,1].unsqueeze(-1)
        grasp_widths = offsets[:,:,:,:,2].unsqueeze(-1)/2 # half_width

        
        for v in range(view_angles.shape[0]):
            for d in range(4):
                depths = grasp_depths[:,v,:,d,:]
                widths = grasp_widths[:,v,:,d,:]
                # 将scene点云转换至gripper坐标系
                targets[...] = target[:,None,:,:] @ View_at_scene[v,None,:,:,:]  
                ## collision detection
                # height mask
                mask1[...] = ((-height/2 < targets[...,2]) & (targets[...,2]<height/2))
                
                # mask2[...] = ((targets[...,0] > -depth_base) & (targets[...,0]<depths))
                mask2[...] = ((targets[...,0] > depths - finger_length) & (targets[...,0]<depths))
                # left finger mask
                mask3[...] = (targets[...,1] > -(widths + finger_width))
                mask4[...] = (targets[...,1] < -widths)
                # right finger mask     OK
                mask5[...] = (targets[...,1] < (widths + finger_width))
                mask6[...] = (targets[...,1] > widths)
                # bottom mask with approach
                mask7[...] = ((targets[...,0] <= depths - finger_length) \
                              & (targets[...,0] > depths - finger_length - finger_width - approach_dist))
                # mask7[...] = ((targets[...,0] < -depth_base) \
                #               & (targets[...,0] > -(depth_base+widths)))
                
                left_mask = (mask1 & mask2 & mask3 & mask4)
                right_mask = (mask1 & mask2 & mask5 & mask6)
                bottom_mask = (mask1 & mask3 & mask5 & mask7)
                inner_mask = (mask1 & mask2 &(~mask4) & (~mask6)) 
         
                collision_mask = torch.any((left_mask | right_mask | bottom_mask), dim=-1)
                empty_mask = (torch.sum(inner_mask, dim=-1) < empty_thresh)
                collision_label[:,v,:,d] = (collision_mask | empty_mask)


        collision_label = collision_label | collision
        obj_name = obj_meta["meta"]["oid"]
        grasp_dict[obj_name] = collision_label.cpu().numpy().astype(bool)


                
    
    np.savez(collision_mask_file, **grasp_dict)


import argparse
parser = argparse.ArgumentParser(description="将文件列表分成8份")
parser.add_argument("part_id", type=int, choices=range(0, 8), help="选择要获取的部分(0-7)")
args = parser.parse_args()
def get_split_files(files, part_num, part_id):
    k, m = divmod(len(files), part_num)
    parts = [files[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(part_num)]
    return parts[part_id]


if __name__ == "__main__":
    labels_root = "/data/panmingjie/OmniObjectPose"
    scenes_root = "/data/panmingjie/render/v1/1/train/floor"
    depthe_files = glob(scenes_root + "/*/*depth.exr", recursive=True)
    depthe_files.sort()

    depthe_files = get_split_files(depthe_files, 8, args.part_id)

    view_angles = get_view_angles()
    view_angles = torch.Tensor(view_angles).cuda()

    with torch.no_grad():
        for depth_file in tqdm(depthe_files, desc="Frame Loop", leave=False): 
            process(depth_file, view_angles)
    
