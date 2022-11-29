import json
import numpy as np
import pdb
import torch

from ray_utils import get_rays, get_ray_directions

# tensor([[[0, 0, 0],
#          [0, 0, 1],
#          [0, 1, 0],
#          [0, 1, 1],
#          [1, 0, 0],
#          [1, 0, 1],
#          [1, 1, 0],
#          [1, 1, 1]]])
BOX_OFFSETS = torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]], device='cuda')


def hash(coords, log2_hashmap_size):
    """
    coords: this function can process upto 7 dim coordinates 八个grid的index
    log2T:  logarithm of T w.r.t 2  论文中的T那个参数的指数值
    """
    # 论文中的Π1，Π2
    # 对于3维空间点，就只能用到前三个
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]
    # [bs,8]
    xor_result = torch.zeros_like(coords)[..., 0]

    for i in range(coords.shape[-1]):
        # 在最后一个维度，就是grid 在各个方向上的index
        # coords [bs,8,3] 第一个是bs，第二个是8个点，第三个是index的值
        # 0 ^ {0 index}*1 ^ {1 index}*2654435761 ^ {2 index}*805459861
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    """
    获取体素的顶点
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    """

    box_min, box_max = bounding_box

    # 判断点位是否超出box
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        pdb.set_trace()
        # 裁剪，防止超出box
        xyz = torch.clamp(xyz, min=box_min, max=box_max)
    # 三边长除以分辨率，就是grid的边长
    grid_size = (box_max - box_min) / resolution
    # 当前的点位-bbox的最小点位值 除 网格的大小，那么就是grid在各个维度的index
    bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()

    # 上面得到的index对应的体素的坐标
    voxel_min_vertex = bottom_left_idx * grid_size + box_min
    # voxel_min_vertex 加上 一个单元的体素 后的坐标
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0, 1.0]) * grid_size

    # 当前点位，对应的四周八个点位的 网格的index
    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS

    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    # 前两个返回值是坐标，最后一个返回值是index
    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices


# ----------------------------------------------------------------------------------------------------------------------

def get_bbox3d_for_blenderobj(camera_transforms, H, W, near=2.0, far=6.0):
    """
    空间中所有的光线上near和far点位的三维坐标
    在其中找到找到xyz对应各自的最小值和最大值
    最小的xyz集合作为最小边界
    最大的xyz集合作为最大边界
    """
    camera_angle_x = float(camera_transforms['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    for frame in camera_transforms["frames"]:
        c2w = torch.FloatTensor(frame["transform_matrix"])
        rays_o, rays_d = get_rays(directions, c2w)

        def find_min_max(pt):
            for i in range(3):
                if (min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if (max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        # 图像的四个角
        for i in [0, W - 1, H * W - W, H * W - 1]:
            min_point = rays_o[i] + near * rays_d[i]  # 这条光线，near点位的坐标
            max_point = rays_o[i] + far * rays_d[i]  # 这条光线，far点位的坐标
            # 更新最小点位
            find_min_max(min_point)
            # 更新最大点位
            find_min_max(max_point)

    return (
        torch.tensor(min_bound) - torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor(max_bound) + torch.tensor([1.0, 1.0, 1.0]))


if __name__ == "__main__":
    with open("data/nerf_synthetic/chair/transforms_train.json", "r") as f:
        camera_transforms = json.load(f)

    bounding_box = get_bbox3d_for_blenderobj(camera_transforms, 800, 800)
