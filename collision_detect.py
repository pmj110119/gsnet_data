from glob import glob
import cv2
import os
import numpy as np
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import taichi as ti

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


import taichi as ti


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


@ti.kernel
def ti_collision(scene:ti.template(), trans:ti.template(), va:ti.template(), 
                points:ti.template(), offsets:ti.template(), scores:ti.template(),
                collisions: ti.template(), res_field: ti.template()):
        for p in range(points.shape[0]):
            for v in range(va.shape[0]):
                for a in range(va.shape[1]):
                    for d in range(collisions.shape[3]):  # Assuming collisions has at least 4 dimensions
                        if scores[p, v, a, d] > 0.8 or scores[p, v, a, d] < 0:
                            continue
                        if collisions[p, v, a, d]:
                            continue
                        R = va[v, a]
                        R = trans[None][:3,:3] @ R
                        if angles_m_z(R) > 80:
                            continue
                        t = transform_single_point(points[p], trans[None])
                        depth = offsets[p, v, a, d, 1]
                        width = offsets[p, v, a, d, 2]
                        if detect_scene_collision(scene, R, t, width, depth) == 0:
                            res_field[p, v, a, d] = 1


@ti.func
def angles_m_z(R):
    PI = 3.141592653589793

    # Assuming R is a ti.Matrix and already normalized (since it's a rotation matrix)
    rotated_x_axis = R[:, 0]  # This is the first column of the rotation matrix

    # Original Z-axis
    original_z_axis = ti.Vector([0, 0, 1])

    # Compute the dot product between the original Z-axis and the rotated X-axis
    dot_product = rotated_x_axis.dot(original_z_axis)

    # For unit vectors, their magnitudes are 1, so we skip magnitude calculation

    # Compute the cosine of the angle between the two vectors
    cos_angle = dot_product  # Since both vectors are unit vectors, their magnitudes are 1

    # Compute the angle in radians and then convert to degrees
    angle_radians = ti.acos(ti.max(ti.min(cos_angle, 1.0), -1.0))  # Clipping for numerical stability
    angle_degrees = angle_radians * 180.0 / PI  # Converting to degrees

    return angle_degrees


@ti.func
def transform_single_point(point, trans):
    point_homogeneous = ti.Vector([point[0], point[1], point[2], 1.0])
    transformed_point_homogeneous = trans @ point_homogeneous
    return transformed_point_homogeneous.xyz


@ti.func
def subtract_point(points: ti.template(), point: ti.template()):
    for i in points:
        points[i] = points[i] - point

@ti.func
def dot_point(points: ti.template(), R: ti.template()):
    for i in points:
        points[i] = points[i] @ R

@ti.func
def detect_scene_collision(scene, R, t, width, depth):
    approach_dist=0.05
    finger_length = 0.06
    finger_width = 0.01
    voxel_size=0.005
    heights = 0.002 / 2
    
    n = scene.shape[0]
    half_width = width / 2

    collision_thresh=0.1
    inner_thresh = 20    #ensure enough points grasped
    f_thresh = 10  #ensure fingers and bottom not collided

    collision_detected = 0

    left_mask_sum = 0
    right_mask_sum = 0
    shifting_mask_sum = 0
    global_mask_sum = 0
    grasp_points_count = 0
    ymin = ti.f32(+float('inf'))
    ymax = ti.f32(-float('inf'))

    for i in range(n):
        target = scene[i] - t
        target = target @ R
        x, y, z = target
        # height mask
        mask1 = -heights < z < heights
        # depth mask
    # subtract_point(scene, t)
        mask2 = depth - finger_length < x < depth
        # left finger mask
        mask3 = y > -(width/2 + finger_width)
        mask4 = y < -width/2
        # right finger mask
        mask5 = y < width/2 + finger_width 
        mask6 = y > half_width
        # bottom mask
        mask7 = depth - finger_length - finger_width < x <= depth - finger_length
        # shifting mask
        mask8 = depth - finger_length - finger_width - approach_dist < x <= depth - finger_length - finger_width

        # get collision mask of each point
        wh_inner_mask = mask1 and mask3 and mask5
        dh_inner_mask = mask1 and mask2
        left_mask = dh_inner_mask and mask3 and mask4
        right_mask = dh_inner_mask and mask5 and mask6
        bottom_mask = wh_inner_mask and mask7
        shifting_mask = wh_inner_mask and mask8
        global_mask = left_mask or right_mask or bottom_mask or shifting_mask

        if left_mask:
            left_mask_sum += 1
        if right_mask:
            right_mask_sum += 1
        if shifting_mask:
            shifting_mask_sum += 1
        if global_mask:
            global_mask_sum += 1

        w_outer_mask = mask4 or mask6
        w_inner_mask = not w_outer_mask and dh_inner_mask

        if w_inner_mask:
            grasp_points_count += 1
            ymin = min(ymin, y)
            ymax = max(ymax, y)

    # Volume calculation (manually done as Ti doesn't support dynamic reshaping)
    left_right_volume = heights * finger_length * finger_width / (voxel_size**3)
    bottom_volume = heights * (width + 2 * finger_width) * finger_width / (voxel_size**3)
    shifting_volume = heights * (width + 2 * finger_width) * approach_dist / (voxel_size**3)
    volume = left_right_volume * 2 + bottom_volume + shifting_volume

    # Calculate equivalent volume of each part and get collision iou of each part
    global_iou = global_mask_sum / (volume + 1e-6)

    if global_iou > collision_thresh:
        collision_detected = 1

    if left_mask_sum > f_thresh or right_mask_sum > f_thresh or shifting_mask_sum > f_thresh:
        collision_detected = 1
    
    if grasp_points_count < inner_thresh:
        collision_detected = 1

    if ymin > 0 or ymax < 0:
        collision_detected = 1

    if ymax - ymin < width / 2:
        collision_detected = 1

    return collision_detected


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
    sub_name = obj_meta["meta"]["instance_path"].split("/")[-3]
    label_path = os.path.join(labels_root, sub_name, "Aligned_m.npz")
    return label_path


def process(task, view_angles):
    depth_file, meta_file = task
    
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    meta = prase_meta(meta_file)

    camera_intrinsic = meta["camera"]["intrinsics"]
    fx, fy, cx, cy = camera_intrinsic["fx"], camera_intrinsic["fy"], camera_intrinsic["cx"], camera_intrinsic["cy"]
    points_cloud = depth2pc(depth[...,0], [fx, fy, cx, cy], scale=4)

    #taichi 
    NUM_VIEWS = 300
    NUM_ANGLES = 12

    ti.init(arch=ti.gpu)
    scene_field = ti.Vector.field(3, dtype=ti.f16, shape=points_cloud.shape[0])
    scene_field.from_numpy(points_cloud.astype(np.float16))
    va_field = ti.Matrix.field(3, 3, dtype=ti.f16, shape=(NUM_VIEWS, NUM_ANGLES))
    va_field.from_numpy(view_angles.astype(np.float16))

    grasp_dict = {}

    for obj_meta in tqdm(meta["objects"].values(), desc="Object Loop", leave=False):
        translation = obj_meta["translation"]
        rotation_matrix = quaternion_wxyz_to_rotation_matrix(obj_meta["quaternion_wxyz"])
        grasp_label_path = prase_label_path(obj_meta)
        import ipdb;ipdb.set_trace()
        if os.path.exists(grasp_label_path) == False:
            print(grasp_label_path)
            continue
        
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation

        sampled_points, offsets, scores, collision = get_model_grasps(grasp_label_path)
        n = sampled_points.shape[0]
        trans_field = ti.Matrix.field(4, 4, dtype=ti.f16, shape=())
        trans_field.from_numpy(transform.astype(np.float16))
        # Assuming the shape of score is (NUM_VIEWS, NUM_ANGLES, NUM_DEPTHS)
        sampled_point_field = ti.Vector.field(3, dtype=ti.f16, shape=(n))
        offest_field = ti.field(ti.f16, shape=offsets.shape)
        score_field = ti.field(ti.f16, shape=scores.shape)
        collision_mask_field = ti.field(ti.i8, shape=collision.shape)
        res_field = ti.field(ti.i32, shape=collision.shape)

        sampled_point_field.from_numpy(sampled_points.astype(np.float16))
        offest_field.from_numpy(offsets.astype(np.float16))
        score_field.from_numpy(scores.astype(np.float16))
        collision_mask_field.from_numpy(collision.astype(np.int8))

        import ipdb;ipdb.set_trace()
        ti_collision(scene_field, trans_field, va_field, sampled_point_field, 
                    offest_field, score_field, collision_mask_field, res_field) 

        collision_result = res_field.to_numpy()
        obj_name = obj_meta["meta"]["oid"]
        grasp_dict[obj_name] = collision_result.astype(np.int8)
    
    collision_maks_path = meta_file.replace("meta.json", "collision_label.npz")
    np.savez(collision_maks_path, **grasp_dict)


if __name__ == "__main__":
    labels_root = "/home/panmingjie/gsnet_data/data_obj/models_514_obj"
    root = "/home/panmingjie/gsnet_data/render/v1/1/train/floor"
    depthes = glob(root + "/*/*depth.exr", recursive=True)
    meta_files = [depth.replace("depth.exr", "meta.json") for depth in depthes]

    depthes.sort()
    meta_files.sort()
    view_angles = get_view_angles()

    tasks = [task for task in zip(depthes, meta_files)]
    
    for task in tqdm(tasks, desc="Scene Loop", leave=False): 
        process(task, view_angles)
    
