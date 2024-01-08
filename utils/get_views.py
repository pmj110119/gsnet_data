from turtle import pos
import numpy as np

def generate_views(N, phi=(np.sqrt(5)-1)/2, center=np.zeros(3, dtype=np.float32), R=1):
    ''' Author: chenxi-wang
    View sampling on a sphere using Febonacci lattices.

    **Input:**

    - N: int, number of viewpoints.

    - phi: float, constant angle to sample views, usually 0.618.

    - center: numpy array of (3,), sphere center.

    - R: float, sphere radius.

    **Output:**

    - numpy array of (N, 3), coordinates of viewpoints.
    '''
    idxs = np.arange(N, dtype=np.float32)
    Z = (2 * idxs + 1) / N - 1
    X = np.sqrt(1 - Z**2) * np.cos(2 * idxs * np.pi * phi)
    Y = np.sqrt(1 - Z**2) * np.sin(2 * idxs * np.pi * phi)
    views = np.stack([X,Y,Z], axis=1)
    views = R * np.array(views) + center
    return views


def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    '''
    **Input:**

    - towards: numpy array towards vectors with shape (n, 3).

    - angle: numpy array of in-plane rotations (n, ).

    **Output:**

    - numpy array of the rotation matrix with shape (n, 3, 3).
    '''
    axis_x = batch_towards
    ones = np.ones(axis_x.shape[0], dtype=axis_x.dtype)
    zeros = np.zeros(axis_x.shape[0], dtype=axis_x.dtype)
    axis_y = np.stack([-axis_x[:,1], axis_x[:,0], zeros], axis=-1)
    mask_y = (np.linalg.norm(axis_y, axis=-1) == 0)
    axis_y[mask_y] = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x, axis=-1, keepdims=True)
    axis_y = axis_y / np.linalg.norm(axis_y, axis=-1, keepdims=True)
    axis_z = np.cross(axis_x, axis_y)
    sin = np.sin(batch_angle)
    cos = np.cos(batch_angle)
    R1 = np.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], axis=-1)
    R1 = R1.reshape([-1,3,3])
    R2 = np.stack([axis_x, axis_y, axis_z], axis=-1)
    matrix = np.matmul(R2, R1)
    return matrix.astype(np.float32)

def viewpoint_params_to_matrix(towards, angle):
    '''
    **Input:**

    - towards: numpy array towards vector with shape (3,).

    - angle: float of in-plane rotation.

    **Output:**

    - numpy array of the rotation matrix with shape (3, 3).
    '''
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
    matrix = matrix.astype(np.float32)

    return matrix


def get_points_view(V=300, A=12):
    views = generate_views(V)
    angles = np.arange(0, np.pi, np.pi / A)

    points_view = np.zeros((V, A, 3, 3))
    for i, view in enumerate(views):
        for j, angle in enumerate(angles):
            points_view[i, j, :, :] = viewpoint_params_to_matrix(-view, angle)

    points_view = points_view.astype(np.float32)
    return points_view


if __name__ == "__main__":
    A = 12
    V = 300
    view_angles = get_points_view(V, A)
    views = generate_views(V)
    angles = np.arange(0, np.pi, np.pi / A)

    v = viewpoint_params_to_matrix(-views[1], angles[1])
    print(view_angles[1][1])
    print(v)

