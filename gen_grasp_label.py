


from utils.find_contact_torch import get_contacts
from utils.get_views import get_points_view, generate_views
from tqdm import tqdm
import open3d as o3d
import numpy as np
import time
import os

import torch
device=torch.device('cuda')

DEBUG=False


def get_boundbox_size(mesh):
    bbox = mesh.get_axis_aligned_bounding_box()
    return bbox.get_max_bound() - bbox.get_min_bound()




score_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
score_array = torch.Tensor([-1] + score_list).to(device)

def grasp_score_parallel(contack_p1, contack_p2, normal_p1, normal_p2, friction_coefs, nc_thresh=0.5):
    check_too_close = torch.linalg.norm(contack_p1 - contack_p2, dim=1) < 0.0001


    diff_r = contack_p2 - contack_p1 + 1e-7
    nr_cos = torch.einsum('ij,ij->i', normal_p2, diff_r) / (torch.linalg.norm(normal_p2, dim=1) * torch.linalg.norm(diff_r, dim=1))

    diff_l = -diff_r
    nl_cos = torch.einsum('ij,ij->i', normal_p1, diff_l) / (torch.linalg.norm(normal_p1, dim=1) * torch.linalg.norm(diff_l, dim=1))


    angle_r = torch.arccos(nr_cos)
    angle_l = torch.arccos(nl_cos)
    angles = torch.stack([angle_r, angle_l])

    friction_angles = torch.arctan(friction_coefs)
    valid_grasps = torch.all(angles[:, :, None] <= friction_angles, dim=0)

    # check_nr_cos_p1 = nr_cos < nc_thresh
    # check_nl_cos_p1 = nl_cos < nc_thresh
    check_mask = check_too_close #| check_nr_cos_p1 | check_nl_cos_p1

    valid_grasps[check_mask] = 0

    # Find the indices of the last 1 in each row. If no 1 is found, index will be 0
    reversed_samples = valid_grasps.flip(dims=[1])
    idx = torch.argmax(reversed_samples.to(torch.int32), dim=1) # Reverse and find the first 1

    last_met_idx = valid_grasps.shape[1] - idx # Adjust indices for reversed array
    last_met_idx[torch.all(valid_grasps == 0, dim=1)] = 0
    
    # Return the scores corresponding to the indices
    return score_array[last_met_idx]


    # return valid_grasps.to(torch.int32)









v, a = 300, 12
depth_list = torch.Tensor([0.01, 0.02, 0.03, 0.04]).to(device)
fc_list = torch.Tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]).to(device)
# depth_list.reverse()


inner_mask_all = torch.zeros((v,a,10000,4,15), dtype=bool).to(device)
outer_mask_all = torch.zeros((v,a,10000,4,15), dtype=bool).to(device)
outer_mask_final = torch.zeros((v,a,10000,4,15), dtype=bool).to(device)

width_mask = torch.zeros((v,a,10000,4,15), dtype=bool).to(device)
depth_mask = torch.zeros((v, a, 10000, len(depth_list)), dtype=bool).to(device)
depth_for_score = torch.zeros((v, a, len(depth_list)), dtype=torch.float32).to(device)
depth_for_score[:,:,0:len(depth_list)] = depth_list


grippers = torch.zeros((v, a, len(depth_list), 2, 3)).to(device)


width_list = [0.01 * x for x in range(1, 16, 1)]
width_tensor = torch.Tensor(width_list+[width_list[-1]]).to(device)

check_mask_all = torch.ones((v, a, len(depth_list), len(width_list)+1), dtype=bool).to(device)
half_width = torch.zeros((v, a, len(depth_list), len(width_list)+1), dtype=torch.float32).to(device)
check_mask = torch.ones((len(depth_list), len(width_list)+1), dtype=torch.bool).to(device).long()
check_inner_nums = torch.zeros((v,a,len(depth_list), len(width_list)), dtype=torch.bool).to(device)

def select_score(score):
    """
        score: torch.Size([10])
    """
    tmp, is_force_closure = False, False
    quality = -1
    fc_list = torch.Tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]).to(device)
    for ind_, value_fc in enumerate(fc_list):
        tmp = is_force_closure
        is_force_closure = score[ind_] == 1
        if tmp and not is_force_closure:
            quality = fc_list[ind_ - 1]
            break
        elif is_force_closure and value_fc == fc_list[-1]:
            quality = value_fc
            break
        elif value_fc == fc_list[0] and not is_force_closure:
            break
    return quality




def grasp_sample(point, point_normal, pcd, pcd_normals,
                 view_angle_matrix, views):
    
    

    V = 300
    A = 12
    H = 0.02
    depth_base = 0.02
    collision_thresh = 0.01

    angles = torch.Tensor([x for x in range(A)]) * torch.pi / A

    v, a = view_angle_matrix.shape[:2]

    curr_offset = torch.zeros((v, a, len(depth_list), 3)).to(device) # angle, depth, width
    curr_collision = torch.ones((v, a, len(depth_list)), dtype=bool).to(device) # has collision: 1, no collision:0


    points_centered = pcd - point       # 10000x3

    t0 = time.time()
    points_centered_expanded = points_centered.unsqueeze(0).unsqueeze(0).expand(v, a, -1, -1)
    points_rotated_all = torch.matmul(view_angle_matrix.transpose(-1, -2), points_centered_expanded.transpose(-1, -2))
    points_rotated_all = points_rotated_all.transpose(-1,-2)
    height_mask_all = (points_rotated_all[:,:,:,2] > (-H / 2.0)) & (points_rotated_all[:,:,:,2] < (H / 2.0))
    
    
    depth_mask[...] = 0
    for i, depth in enumerate(depth_list):
        depth_mask[:,:,:,i] = points_rotated_all[:,:,:,0]<depth

    final_mask_all = height_mask_all.unsqueeze(-1) & depth_mask

    width_mask[...] = 0
    for i, w in enumerate(width_list):
        mask3 = (points_rotated_all[:,:,:,1] > (-w / 2.0))
        mask4 = (points_rotated_all[:,:,:,1] < (w / 2.0))
        mask = mask3 & mask4
        width_mask[:,:,:,:,i] = mask.unsqueeze(3)


    
    check_mask_all[...] = 1
    half_width[...] = 0
    check_mask[...] = 1
    check_inner_nums[...] = 0




    
    

    inf_1 = torch.tensor(float('inf')).to(device)
    inf_2 = torch.tensor(-float('inf')).to(device)

    inner_mask_all[...] = final_mask_all.unsqueeze(-1) & width_mask
    outer_mask_all[...] = final_mask_all.unsqueeze(-1) & ~width_mask

    points_rotated_all = points_rotated_all.unsqueeze(-1).unsqueeze(-1)
    points_y = points_rotated_all[:,:,:,1,:,:]
    y_inner_min = torch.where(inner_mask_all, points_y, inf_1).min(dim=2).values
    y_inner_max = torch.where(inner_mask_all, points_y, inf_2).max(dim=2).values
    half_width[:,:,:,:-1] = torch.max(abs(y_inner_min), abs(y_inner_max))

    for i in range(v):
        for j in range(a):
            half_width_ = half_width[i,j]
            outer_mask1 = (points_y[i,j] >= -(half_width_[:,:-1] + collision_thresh))
            outer_mask2 = (points_y[i,j] <= (half_width_[:,:-1] + collision_thresh))
            outer_mask_final[i,j] = outer_mask1 & outer_mask2 & outer_mask_all[i,j]

            # check inner space, make sure the gripper can grasp enough points
            check_inner_nums[i,j] = torch.sum(inner_mask_all[i,j], dim=0) > 10

            curr_offset[i,j,:,0] =  angles[j]

    # check contact points, make sure each finger has same approach distance
    check_outer_nums =  torch.all(outer_mask_final == 0, dim=2)
    # check depth, make sure the gripper is not too deep
    check_too_depth =  torch.all((inner_mask_all & (points_rotated_all[:,:,:,0] < -depth_base)) == 0 , dim=2)

    check_mask_all[:,:,:,:-1] = check_outer_nums & check_too_depth & check_inner_nums

    idx = torch.argmax(check_mask_all.to(torch.int32), dim=-1)

    curr_offset[:,:,:, 1] = depth_list
    curr_offset[:,:,:,2:3] = torch.gather(half_width*2, 3, idx.unsqueeze(-1))
    curr_collision = idx==len(width_list)

    R_all = view_angle_matrix.unsqueeze(2).expand(-1, -1, len(depth_list), -1, -1)

    center = torch.zeros([V, A, len(depth_list), 3]).to(device)
    center[:,:,:,0] = depth_list
    center = torch.matmul(R_all, center.unsqueeze(-1)).squeeze(-1)
    center = center + point

    grippers[...] = 0
    grippers[:,:,:,0,:] = center - curr_offset[:,:,:,2:3]/2 * R_all[:,:,:,:,1]
    grippers[:,:,:,1,:] = center + curr_offset[:,:,:,2:3]/2 * R_all[:,:,:,:,1]


    contacts_normals = get_contacts(pcd, pcd_normals, p1=grippers[...,0,:], p2=grippers[...,1,:])

    scores = grasp_score_parallel(contacts_normals[0], contacts_normals[1], contacts_normals[2], contacts_normals[3], fc_list) 
    scores = scores.view(v, a, len(depth_list))



    if DEBUG:
        scores_ours = scores.cpu().numpy()
        # import ipdb;ipdb.set_trace()
        gsnet_result = np.load('gsnet_result.npz')
        official_label = gsnet_result['curr_label']

        true_mask = official_label>0
        false_mask = ~true_mask
        tp = (scores_ours[true_mask]>0).sum()/true_mask.sum()
        fp = (scores_ours[false_mask]<=0).sum()/false_mask.sum()
        print('2 Classication: TP:%4f, FP:%4f'%(tp.item(), fp.item()))
        tp = (scores_ours[true_mask] == official_label[true_mask]).sum()/true_mask.sum()
        fp = (scores_ours[false_mask] == official_label[false_mask]).sum()/false_mask.sum()
        print('10 Classication: TP:%4f, FP:%4f'%(tp.item(), fp.item()))
        diff = scores_ours-official_label
        diff_less_1 = abs(diff)<=0.1
        tp = diff_less_1[true_mask].sum()/true_mask.sum()
        fp = diff_less_1[false_mask].sum()/false_mask.sum()
        print('10 Classication, diff<0.1: TP:%4f, FP:%4f'%(tp.item(), fp.item()))
        diff = scores_ours-official_label
        diff_less_1 = abs(diff)<=0.2
        tp = diff_less_1[true_mask].sum()/true_mask.sum()
        fp = diff_less_1[false_mask].sum()/false_mask.sum()
        print('10 Classication, diff<0.2: TP:%4f, FP:%4f'%(tp.item(), fp.item()))
        diff = scores_ours-official_label
        diff_less_1 = abs(diff)<=0.3
        tp = diff_less_1[true_mask].sum()/true_mask.sum()
        fp = diff_less_1[false_mask].sum()/false_mask.sum()
        print('10 Classication, diff<0.3: TP:%4f, FP:%4f'%(tp.item(), fp.item()))
        exit()
        


    return curr_offset, curr_collision, scores


def process_obj(obj_name):
    label_path = obj_name.replace(".obj", ".npz")
    if os.path.exists(label_path):
        print('Skip', label_path)
        return

    mesh = o3d.io.read_triangle_mesh(obj_name)
    if 'PhoCaL' in obj_name or 'OmniObject3d' in obj_name or 'GoogleScan' in obj_name:
        mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices)/1000.) 

    mesh_size = min(get_boundbox_size(mesh))
    obj_pcd = mesh.sample_points_uniformly(number_of_points=10000, seed=1999)

    sample_voxel_size, model_voxel_size, model_num_sample = 0.004, 0.001, 10000
    # sample_voxel_size = mesh_size/10

    normal_radius = mesh_size / 8
    obj_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    pcd_normals = torch.Tensor(obj_pcd.normals).to(device)    
    obj_sampled = obj_pcd.voxel_down_sample(sample_voxel_size)
    obj_sampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    sampled_normals = torch.Tensor(obj_sampled.normals).to(device)

    views = generate_views(300)
    views_matrix = get_points_view()

    views = torch.Tensor(views).to(device)
    views_matrix = torch.Tensor(views_matrix).to(device)

    sampled_points = torch.Tensor(obj_sampled.points).to(device)
    sampled_ind = range(len(sampled_points))
    pcd_points = torch.Tensor(obj_pcd.points).to(device)
    


    saved_score = torch.zeros((len(sampled_points), 300, 12 ,4)).to(device)
    saved_offset = torch.zeros((len(sampled_points), 300, 12 ,4, 3)).to(device)
    saved_collision = torch.zeros((len(sampled_points), 300, 12 ,4), dtype=bool).to(device)

    with torch.no_grad():
        for i in tqdm(sampled_ind, total=len(sampled_ind)):
            # import ipdb;ipdb.set_trace()
            offset, collision, score = grasp_sample(
                sampled_points[i], sampled_normals[i], 
                pcd_points, pcd_normals, 
                views_matrix, views
            )
            
            saved_score[i] = score#.cpu().numpy()
            saved_offset[i] = offset#.cpu().numpy()
            saved_collision[i] = collision#.cpu().numpy()
     

    np.savez_compressed(label_path,
             points=sampled_points.cpu().numpy(),
             offsets=saved_offset.cpu().numpy(),
             collision=saved_collision.cpu().numpy(),
             scores=saved_score.cpu().numpy())



import argparse
parser = argparse.ArgumentParser(description="将文件列表分成8份")
parser.add_argument("part_id", type=int, choices=range(0, 8), help="选择要获取的部分(0-7)")
args = parser.parse_args()

def split_list_into_parts(lst, n):
    """将列表分为n个部分"""
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__  == "__main__":
    root = '/data/panmingjie/OmniObjectPose/data'
    with open("./filtered_obj_list.txt", 'r') as f:
        obj_list = f.read().splitlines()
        obj_list = [os.path.join(root, x.replace(os.path.basename(x), 'Aligned.obj')) for x in obj_list]

    parts = list(split_list_into_parts(obj_list, 8))
    selected_objs = parts[args.part_id]  


    for idx, obj_name in enumerate(selected_objs):
        print('%d/%d %s'%(idx, len(selected_objs), obj_name))
        process_obj(obj_name)
