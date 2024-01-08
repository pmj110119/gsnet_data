import re
from matplotlib.pylab import f
import numpy as np
import torch
import open3d as o3d

# get rotation matrix from normal vector
def get_rotation_matrix_from_normal(normal_vector):
    matrix = np.eye(3)
    matrix[:, 2] = normal_vector
    matrix[:, 0] = np.cross(normal_vector, np.array([0, 0, 1]))
    matrix[:, 0] /= np.linalg.norm(matrix[:, 0])
    matrix[:, 1] = np.cross(matrix[:, 2], matrix[:, 0])
    return matrix


# open3d sphere from point
def sphere_from_point(point, radius = 0.01, color = [0, 1, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(point)
    sphere.paint_uniform_color(color)
    return sphere


# open3d line from points
def line_from_point(p1, p2, color = [1, 0, 0]):
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(torch.Tensor([p1, p2]))
    line.lines = o3d.utility.Vector2iVector(torch.Tensor([[0, 1]]))
    line.colors = o3d.utility.Vector3dVector(torch.Tensor([color]))
    return line


# get bound box size of open3d mesh
def get_boundbox_size(mesh):
    bbox = mesh.get_axis_aligned_bounding_box()
    return bbox.get_max_bound() - bbox.get_min_bound()


# draw a arrow given a point and a normal
def draw_arrow(point, normal, length=0.001, color=[0, 1, 0]):
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.0001, cone_radius=0.0002, cylinder_height=length, cone_height=length*0.2)
    #get rotation matrix from normal
    rotation_matrix = get_rotation_matrix_from_normal(normal)
    arrow.rotate(rotation_matrix, center=(0,0,0))
    arrow.translate(point)
    arrow.paint_uniform_color(color)
    return arrow


# find top k smallest elements' index in a numpy array
def find_top_k_smallest_index(array, k):
    return torch.argsort(array)[:k]


# calculate the distance between a point and a line
def point_to_line_distance(points, p1, p2):
    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)
    dis_line = torch.linalg.norm(p2 - p1, dim=2)
    cross = torch.cross((p1 - points), (points - p2), dim=2)

    distance = torch.linalg.norm(cross, dim=2) / dis_line
    return distance


def point_to_line_projection_ratio(points, c1, c2):
    c1 = c1.unsqueeze(1)
    c2 = c2.unsqueeze(1)
    # 计算线段向量及其长度的平方
    segment_vector = c2 - c1
    segment_length_sq = torch.sum(segment_vector ** 2, dim=2) + 1e-7
    # 计算每个点到c1的向量
    vectors_to_c1 = points - c1
    # 计算向量在线段上的投影长度比例
    projection_ratios = torch.sum(vectors_to_c1 * segment_vector, dim=2) / segment_length_sq

    return projection_ratios




# get gripper contacts and normals points on point cloud 
def get_contacts(points, normals, p1, p2, distance_thresh=0.002):
    """
        points: Nx3
        normals: Nx3
        p1: Bx3
        p2: Bx3
    """
    # import ipdb;ipdb.set_trace()
    if len(p1.shape)>2:
        p1 = p1.reshape(-1,3)
    if len(p2.shape)>2:
        p2 = p2.reshape(-1,3)
    distance_to_line = point_to_line_distance(points, p1, p2)  
    mask_batch = distance_to_line < distance_thresh      # BxN  仅考虑连线附近的点（圆柱体）
    
    p1_contact = torch.zeros_like(p1)
    p2_contact = torch.zeros_like(p1)
    p1_normal = torch.zeros_like(p1)
    p2_normal = torch.zeros_like(p1)
    
    p1_contact[:,0] = 1e-7
    p2_contact[:,0] = 1e-6
    p1_normal[:,0] = 1
    p2_normal[:,0] = 1


    # Filter out cases with less than 10 close points
    valid_cases = torch.sum(mask_batch, dim=1) >= 4        # Bx1 


    # import ipdb;ipdb.set_trace()
    # 确保投影点在c1和c2之间
    projection_ratios = point_to_line_projection_ratio(points, p1, p2)
    mask_projection_batch = (projection_ratios >= 0) & (projection_ratios <= 1)

    mask_batch = mask_batch & mask_projection_batch



    
    points = points.unsqueeze(0).expand(p1.shape[0], -1, -1)
    normals = normals.unsqueeze(0).expand(p1.shape[0], -1, -1)

    distance_to_p1 = torch.norm(points - p1.unsqueeze(1), dim=2)
    distance_to_p2 = torch.norm(points - p2.unsqueeze(1), dim=2)
    
    distance_to_p1[~mask_batch] = torch.tensor(float('inf'))
    distance_to_p2[~mask_batch] = torch.tensor(float('inf'))
    # distance_to_p1[]

    # Find indices of minimum distances
    contact_p1_idx = torch.argmin(distance_to_p1, dim=1).view(-1,1,1).expand(-1,-1,3)
    contact_p2_idx = torch.argmin(distance_to_p2, dim=1).view(-1,1,1).expand(-1,-1,3)



    p1_contact[valid_cases] = torch.gather(points, 1, contact_p1_idx)[valid_cases][:,0,:]
    p2_contact[valid_cases] = torch.gather(points, 1, contact_p2_idx)[valid_cases][:,0,:]
    p1_normal[valid_cases] = torch.gather(normals, 1, contact_p1_idx)[valid_cases][:,0,:]
    p2_normal[valid_cases] = torch.gather(normals, 1, contact_p2_idx)[valid_cases][:,0,:]

    
    return torch.stack([p1_contact, p2_contact, p1_normal, p2_normal], dim=0)






import time

if __name__ == "__main__":
    mesh = o3d.io.read_triangle_mesh("/home/panmingjie/data_obj/PhoCaL/bottle/bottle_003/Scan/Simp.obj")
    obj_pcd = mesh.sample_points_uniformly(number_of_points=10000)
    obj_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=40))
    obj_normals = torch.Tensor(obj_pcd.normals).cuda()

    # p1 = [[-0.1, 0, 0.08], [-0.1, 0, 0.02]] * 150*12*4
    # p1 = torch.Tensor(p1)

    # p2 = [[0.1, 0, -0.05], [0.1, 0, 0.03]] * 150*12*4
    # p2 = torch.Tensor(p2)
    

    points = torch.Tensor(obj_pcd.points).cuda()

    # t0 = time.time()
    # p1 = p1.cuda()
    # p2 = p2.cuda()
    # points = points.cuda()

    


    # import ipdb;ipdb.set_trace()
    gsnet_data = np.load('gsnet_result.npz')
    c1 = gsnet_data['p1'].reshape(-1,3)
    c2 = gsnet_data['p2'].reshape(-1,3)
    n1 = gsnet_data['n1'].reshape(-1,3)
    n2 = gsnet_data['n2'].reshape(-1,3)


    gripper1 = gsnet_data['gripper1'].reshape(-1,3)
    gripper2 = gsnet_data['gripper2'].reshape(-1,3)



    ours_contacts = get_contacts(points, obj_normals, torch.Tensor(gripper1).cuda(), torch.Tensor(gripper2).cuda())

    c1_ours = ours_contacts[0].cpu().numpy()
    c2_ours = ours_contacts[1].cpu().numpy()
    n1_ours = ours_contacts[2].cpu().numpy()
    n2_ours = ours_contacts[3].cpu().numpy()

    num = 0
    for i in range(c1.shape[0]):
        if c1[i].max()==0:
            continue
        print(i)
        line_pcd = line_from_point(gripper1[i], gripper2[i])
        arrow1 = draw_arrow(c1[i], n1[i]*5)
        arrow2 = draw_arrow(c2[i], n2[i]*5)
        sphere1 = sphere_from_point(c1[i], radius=0.001)
        sphere2 = sphere_from_point(c2[i], radius=0.001)

        arrow1_ours = draw_arrow(c1_ours[i], n1_ours[i]*5, color=[1,0,0])
        arrow2_ours = draw_arrow(c2_ours[i], n2_ours[i]*5, color=[1,0,0])
        sphere1_ours = sphere_from_point(c1_ours[i], radius=0.001, color=[1,0,0])
        sphere2_ours = sphere_from_point(c2_ours[i], radius=0.001, color=[1,0,0])

        o3d.visualization.draw_geometries([obj_pcd, line_pcd, arrow1, arrow2, sphere1, sphere2, arrow1_ours, arrow2_ours, sphere1_ours, sphere2_ours])
        # o3d.visualization.draw_geometries([])

        num += 1
        if num>=2:
            break
