from utils.utils_common import DataModes, crop
from utils import stns
from skimage import measure
import vtk
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates, gaussian_filter
import numpy as np
from IPython import embed


class Sample:
    def __init__(self, x, y, atlas):
        self.x = x
        self.y = y
        self.atlas = atlas


class SamplePlus:
    def __init__(self, x, y, y_outer=None, w=None, x_super_res=None, y_super_res=None, y_outer_super_res=None,
                 shape=None):
        self.x = x
        self.y = y
        self.y_outer = y_outer
        self.x_super_res = x_super_res
        self.y_super_res = y_super_res
        self.w = w
        self.shape = shape


class DatasetAndSupport(object):

    def quick_load_data(self, patch_shape): raise NotImplementedError

    def load_data(self, patch_shape): raise NotImplementedError

    def evaluate(self, target, pred, cfg): raise NotImplementedError

    def save_results(self, target, pred, cfg): raise NotImplementedError

    def update_checkpoint(self, best_so_far, new_value): raise NotImplementedError


def get_item(item, mode, config):
    x = item.x.cuda()[None]
    y = item.y.cuda()
    y_outer = item.y_outer.cuda()
    shape = item.shape

    # augmentation done only during training
    if mode == DataModes.TRAINING:  # if training do augmentation

        x = elastic_deformation(x)
        x = add_gaussian_noise(x)

        if torch.rand(1)[0] > 0.5:  # 维度置换, 可以模拟从不同角度对物体的观察
            x = x.permute([0, 1, 3, 2])
            y = y.permute([0, 2, 1])
            y_outer = y_outer.permute([0, 2, 1])

        if torch.rand(1)[0] > 0.5:  # 翻转
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[0])
            y_outer = torch.flip(y_outer, dims=[0])

        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
            y_outer = torch.flip(y_outer, dims=[1])

        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, dims=[3])
            y = torch.flip(y, dims=[2])
            y_outer = torch.flip(y_outer, dims=[2])
        # 机器人学空间变换
        orientation = torch.tensor([0, -1, 0]).float()  # 旋转变换
        new_orientation = (torch.rand(3) - 0.5) * 2 * np.pi
        new_orientation = F.normalize(new_orientation, dim=0)
        q = orientation + new_orientation
        q = F.normalize(q, dim=0)
        theta_rotate = stns.stn_quaternion_rotations(q)

        shift = torch.tensor(
            [d / (D // 2) for d, D in zip(2 * (torch.rand(3) - 0.5) * config.augmentation_shift_range, y.shape)])
        theta_shift = stns.shift(shift)  # 平移变换

        f = 0.1
        scale = 1.0 - 2 * f * (torch.rand(1) - 0.5)  # 缩放变换
        theta_scale = stns.scale(scale)

        theta = theta_rotate @ theta_shift @ theta_scale

        x, y, y_outer = stns.transform(theta, x, y, y_outer)

    surface_points_normalized_all = []
    vertices_mc_all = []
    faces_mc_all = []
    vertices_vtk_all = []
    faces_vtk_all = []
    for i in range(1, config.num_classes):
        shape = torch.tensor(y.shape)[None].float()

        gap = 1
        y_ = clean_border_pixels((y == i).long(), gap=gap)  # 降噪， 降低边缘效应
        vertices_mc, faces_mc = voxel2mesh(y_, 1, shape)  # MARCH CUBES 进行网格化处理
        vertices_mc_all += [vertices_mc]
        faces_mc_all += [faces_mc]

        y_outer = sample_outer_surface_in_voxel((y == i).long())  # 输入仍保持原来维度，与y没区别
        surface_points = torch.nonzero(y_outer)  # 找出非零的序号
        surface_points = torch.flip(surface_points, dims=[1]).float()  # convert z,y,x -> x, y, z
        surface_points_normalized = normalize_vertices(surface_points, shape)  # -1，1
        # surface_points_normalized = y_outer 


        perm = torch.randperm(len(surface_points_normalized))
        point_count = 3000
        surface_points_normalized_all += [surface_points_normalized[perm[:len(perm)]].cuda()]  # randomly pick 3000 points    乱序查找，至多3000点

    if mode == DataModes.TRAINING:
        return {'x': x,
                'y_voxels': y,
                'vertices_mc': vertices_mc_all,
                'surface_points': surface_points_normalized_all,  # 只有一个元素的一个列表
                'unpool': [0, 1, 0, 1, 1]
                }
    else:
        return {'x': x,
                'y_voxels': y,
                'vertices_mc': vertices_mc_all,
                'faces_mc': faces_mc_all,
                'surface_points': surface_points_normalized_all,
                'unpool': [0, 1, 0, 1, 1]}


def sample_outer_surface_in_voxel(volume):

    # outer surface
    a = F.max_pool3d(volume[None, None].float(), kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))[0]
    b = F.max_pool3d(volume[None, None].float(), kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))[0]
    c = F.max_pool3d(volume[None, None].float(), kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))[0]
    border, _ = torch.max(torch.cat([a, b, c], dim=0), dim=0)
    surface = border - volume.float()
    return surface.long()


def normalize_vertices(vertices, shape):
    assert len(vertices.shape) == 2 and len(shape.shape) == 2, "Inputs must be 2 dim"
    assert shape.shape[0] == 1, "first dim of shape should be length 1"

    return 2 * (vertices / (torch.max(shape) - 1) - 0.5)  # 转换到-1到1   这个地方注意三轴是不是都要一样


def sample_to_sample_plus(samples, cfg, datamode):
    new_samples = []
    # surface_point_count = 100
    for sample in samples:
        x = sample.x
        y = sample.y

        y = (y > 0).long()

        shape = torch.tensor(y.shape)[None].float()
        y_outer = sample_outer_surface_in_voxel(y)  # 返回的是tensor

        new_samples += [SamplePlus(x.cpu(), y.cpu(), y_outer.cpu(), shape=shape)]  # # 全是tensor

    return new_samples


def voxel2mesh(volume, gap, shape):  # gap值越小，产生的网格细节越丰富；值越大，结果的网格就越粗糙。
    '''
    :param volume:
    :param gap:
    :param shape:
    :return:
    '''
    vertices_mc, faces_mc, _, _ = measure.marching_cubes_lewiner(volume.cpu().data.numpy(), 0.5, step_size=gap,
                                                         allow_degenerate=False)  # 表示不允许产生退化的三角形（即面积为0的三角形）
    #
    # normals 包含与vertices中的每个顶点相对应的表面法向量。法向量对于计算光照和渲染表面非常有用。
    # values 包含与vertices中的每个顶点相对应的标量值

    vertices_mc = torch.flip(torch.from_numpy(vertices_mc), dims=[1]).float()  # convert z,y,x -> x, y, z
    vertices_mc = normalize_vertices(vertices_mc, shape)
    faces_mc = torch.from_numpy(faces_mc).long()
    return vertices_mc, faces_mc

def voxel2mesh_smooth(volume, gap, shape, sigma=0.1):
    '''
    :param volume: 3D体积数据
    :param gap: Marching Cubes 算法的步长参数，控制输出网格的精细度
    :param shape: 3D体积数据的形状
    :param sigma: 高斯平滑滤波器的标准差，控制平滑程度
    :return: 网格的顶点和面
    '''

    # 使用 Marching Cubes 提取表面
    vertices_mc, faces_mc, _, _ = measure.marching_cubes_lewiner(volume.cpu().data.numpy(), 0.5, step_size=gap,
                                                         allow_degenerate=False)

    # 对顶点数据进行高斯平滑处理
    vertices_mc_smoothed = gaussian_filter(vertices_mc, sigma=sigma)

    # 调整顶点顺序并归一化
    vertices_mc_smoothed = torch.flip(torch.from_numpy(vertices_mc_smoothed), dims=[1]).float()  # convert z,y,x -> x, y, z
    vertices_mc_smoothed = normalize_vertices(vertices_mc_smoothed, shape)

    # 将面数据转换为张量
    faces_mc = torch.from_numpy(faces_mc).long()

    return vertices_mc_smoothed, faces_mc


def clean_border_pixels(image, gap):
    '''
    :param image:
    :param gap:
    :return:
    '''
    assert len(image.shape) == 3, "input should be 3 dim"

    D, H, W = image.shape
    y_ = image.clone()
    y_[:gap] = 0;
    y_[:, :gap] = 0;
    y_[:, :, :gap] = 0;
    y_[D - gap:] = 0;
    y_[:, H - gap] = 0;
    y_[:, :, W - gap] = 0;

    return y_




def elastic_deformation(volume, alpha=1000, sigma=50, random_state=None):
    """Apply elastic deformation on a 4D volume where the first dimension is singleton.

    Args:
        volume (torch.Tensor): The input volume tensor of shape (1, D, H, W).
        alpha (float): The scaling factor for deformation intensities.
        sigma (float): The standard deviation of the Gaussian kernel used for smoothing the random displacement fields.
        random_state (int, optional): Random state for reproducibility.

    Returns:
        torch.Tensor: The deformed volume.
    """
    # Remove the singleton dimension and convert to NumPy array
    volume_np = volume.squeeze(0).cpu().numpy()  # This changes shape from (1, D, H, W) to (D, H, W)

    if random_state is not None:
        np.random.seed(random_state)

    # Generate random displacement fields (vectors) for each axis
    displacement_field = np.random.uniform(-1, 1, (3, *volume_np.shape))

    # Smooth the random displacement fields
    for i in range(3):
        displacement_field[i, ...] = gaussian_filter(displacement_field[i, ...], sigma, mode="constant", cval=0)

    # Scale the displacement fields
    displacement_field *= alpha

    # Create a grid of indices
    D, H, W = volume_np.shape
    coords = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')

    # Apply the displacements
    indices = [coords[0] + displacement_field[0], coords[1] + displacement_field[1], coords[2] + displacement_field[2]]

    # Map the coordinates from the displaced grid to the original grid
    deformed_volume = map_coordinates(volume_np, indices, order=1, mode='reflect').astype(np.float32)

    # Convert the deformed volume back to a torch tensor, re-adding the singleton dimension
    return torch.from_numpy(deformed_volume).unsqueeze(0).to(volume.device)


def add_gaussian_noise(x, mean=0, std=0.1):
    noise = torch.randn_like(x) * std + mean
    noisy_x = x + noise
    return noisy_x
