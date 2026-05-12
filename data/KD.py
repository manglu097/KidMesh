import sys
import numpy as np
import json
from .data import DatasetAndSupport, get_item, sample_to_sample_plus

from utils.metrics import jaccard_index, chamfer_weighted_symmetric, chamfer_directed, calculate_dice_iou_voe, \
    chamfer_symmetric, ASSD, HD90, evaluate_mesh_topology, p2s_distance
from utils.utils_common import crop, DataModes, crop_indices, blend

from torch.utils.data import Dataset
import pickle
import torch.nn.functional as F
import torch
import os
import h5py
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom


class Sample:
    def __init__(self, x, y, atlas):
        self.x = x
        self.y = y
        self.atlas = atlas


class KD_Dataset(Dataset):

    def __init__(self, data, cfg, mode):
        self.data = data

        self.cfg = cfg
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return get_item(item, self.mode, self.cfg)


class KD(DatasetAndSupport):

    def quick_load_data(self, cfg, trial_id):

        data_root = cfg.dataset_path
        data = {}
        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):
            with open(data_root + '/final_data_' + datamode + '.pickle', 'rb') as handle:
                samples = pickle.load(handle)
                new_samples = sample_to_sample_plus(samples, cfg, datamode)
                data[datamode] = KD_Dataset(new_samples, cfg, datamode)

        return data

    def pre_process_dataset(self, cfg):

        print('Data pre-processing - KD Dataset')

        data_root = cfg.dataset_path
        out_shape = cfg.output_shape
        pad_shape = cfg.pad_shape
        dataset_init(data_root, "./data/dataset/positions.json")

        samples = [dir for dir in os.listdir('{}/data'.format(data_root))]

        print('Load data...')

        inputs = []
        labels = []
        for itr, sample in enumerate(samples):

            if '.npy' in sample:
                x, y = self.read_sample(data_root, sample, pad_shape)
                inputs += [x.cpu()]
                labels += [y.cpu()]

        inputs_ = [i[None].data.numpy() for i in inputs]
        labels_ = [i[None].data.numpy() for i in labels]
        inputs_ = np.concatenate(inputs_, axis=0)
        labels_ = np.concatenate(labels_, axis=0)

        hf = h5py.File(data_root + '/data.h5', 'w')
        hf.create_dataset('inputs', data=inputs_)
        hf.create_dataset('labels', data=labels_)
        hf.close()

        print('Saving data...')

        np.random.seed(0)
        perm = np.random.permutation(len(inputs))
        counts = [perm[:40], perm[40:]]  # 8:2

        data = {}

        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):

            samples = []

            for j in counts[i]:
                x = inputs[j]
                y = labels[j]

                samples.append(Sample(x, y, None))  # 一对数据就是一个对象

            with open(data_root + '/final_data_' + datamode + '.pickle', 'wb') as handle:
                pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # data[datamode] = KD_Dataset(samples, cfg, datamode)

        print('\nPre-processing complete')
        return data

    def evaluate(self, target, pred, cfg):
        results = {}

        if target.voxel is not None:  # 栅格化的评价指标
            val_dice, val_jaccard, val_VOE = calculate_dice_iou_voe(target.voxel, pred.voxel, cfg.num_classes)
            results['Dice'] = val_dice
            results['Jaccard'] = val_jaccard
            results['VOE'] = val_VOE

            a = target.voxel
            b = pred.voxel

        if target.points is not None:
            target_points = target.points
            pred_points = pred.mesh
            target_points_MC = target.mesh[0]['vertices']
            # val_chamfer_weighted_symmetric = np.zeros(len(target_points))  # 就一个数字

            # for i in range(len(target_points)):
            #     val_chamfer_weighted_symmetric[i] = chamfer_symmetric(target_points[i].cpu(),
            #                                                                    pred_points[i]['vertices'])
            CF = chamfer_symmetric(target_points[0].cpu(), pred_points[0]['vertices'])
            CF_MC = chamfer_symmetric(target_points_MC, pred_points[0]['vertices'])
            AD = ASSD(target_points[0].cpu(), pred_points[0]['vertices'])
            AD_MC = ASSD(target_points_MC, pred_points[0]['vertices'])
            HD = HD90(target_points[0].cpu(), pred_points[0]['vertices'])
            HD_MC = HD90(target_points_MC, pred_points[0]['vertices'])
            P2D_MC = p2s_distance(target_points_MC, pred_points[0]['vertices'])

            c = target_points_MC

            results['CF'] = CF
            results['CF_MC'] = CF_MC
            results['AD'] = AD
            results['AD_MC'] = AD_MC
            results['HD'] = HD
            results['HD_MC'] = HD_MC
            results['P2D_MC'] = P2D_MC

        if pred.mesh is not None:
            pred_mesh = pred.mesh
            SI, Hole = evaluate_mesh_topology(pred_mesh[0]['vertices'], pred_mesh[0]['faces'])
            results['SI'] = SI
            results['Hole'] = Hole

            d = pred_mesh[0]['vertices']
            e = pred_mesh[0]['faces']

        return results

    def update_checkpoint(self, best_so_far, new_value):

        if 'CF' in new_value[DataModes.TESTING]:
            key1 = 'CF'
            key2 = 'CF_MC'
            new_value1 = new_value[DataModes.TESTING][key1]
            new_value2 = new_value[DataModes.TESTING][key2]

            if best_so_far is None:
                return True
            else:
                best_so_far1 = best_so_far[DataModes.TESTING][key1]
                best_so_far2 = best_so_far[DataModes.TESTING][key2]
                new_value = np.mean(new_value1) + np.mean(new_value2)
                best_so_far = np.mean(best_so_far1) + np.mean(best_so_far2)
                return True if new_value < best_so_far else False  # 优先级更高

        elif 'Jaccard' in new_value[DataModes.TESTING]:
            key1 = 'Jaccard'
            key2 = 'Dice'
            new_value1 = new_value[DataModes.TESTING][key1]
            new_value2 = new_value[DataModes.TESTING][key2]

            if best_so_far is None:
                return True
            else:
                best_so_far1 = best_so_far[DataModes.TESTING][key1]
                best_so_far2 = best_so_far[DataModes.TESTING][key2]
                new_value = np.mean(new_value1) + np.mean(new_value2)
                best_so_far = np.mean(best_so_far1) + np.mean(best_so_far2)
                return True if new_value > best_so_far else False  # 优先级更高


    def read_sample(self, data_root, sample, pad_shape):  # pad_shape假定为ZYX
        x = np.load(f'{data_root}/data/{sample}')
        y = np.load(f'{data_root}/seg/{sample}')

        # 等比重采样
        x_resampled = resample_to_max_dim(x, max(pad_shape), mode='image')
        y_resampled = resample_to_max_dim(y, max(pad_shape), mode='label')

        # 对称填充
        x_padded = symmetric_pad(x_resampled, pad_shape)
        y_padded = symmetric_pad(y_resampled, pad_shape)

        # 标准化
        x_padded_normalized = (x_padded - np.mean(x_padded)) / np.std(x_padded)

        # 转换为Tensor
        x_tensor = torch.from_numpy(x_padded_normalized).float()
        y_tensor = torch.from_numpy(y_padded).long()

        # 这里不再需要使用F.interpolate调整尺寸
        return x_tensor, y_tensor



def dataset_init(data_root, json_file):
    samples = [dir for dir in os.listdir(f'{data_root}/data')]

    for itr, sample in enumerate(tqdm(samples)):
        if '.nii.gz' in sample:
            x_path = os.path.join(data_root, 'data', sample)
            y_path = os.path.join(data_root, 'seg', sample)

            x_nii = nib.load(x_path)
            y_nii = nib.load(y_path)
            original_spacing = x_nii.header.get_zooms()[:3]

            x_data = x_nii.get_fdata()
            y_data = y_nii.get_fdata() > 0  # 二值化并获取数据

            # 先裁剪
            x_cropped = crop_3d(x_data, json_file, sample)
            y_cropped = crop_3d(y_data, json_file, sample)

            # 再统一spacing
            x_resized = resize_image_to_target_spacing(x_cropped, original_spacing, mode='image')
            y_resized = resize_image_to_target_spacing(y_cropped, original_spacing, mode='label')

            x = torch.from_numpy(x_resized).permute([2, 1, 0]).float()
            y = torch.from_numpy(y_resized).permute([2, 1, 0]).float()

            x = x.numpy()
            y = y.long().numpy()

            np.save('{}/data/{}'.format(data_root, sample), x)
            np.save('{}/seg/{}'.format(data_root, sample), y)


def resample_to_max_dim(image, max_dim=128, mode='image'):
    # 计算重采样因子
    scaling_factor = max_dim / max(image.shape)
    new_shape = np.round(np.array(image.shape) * scaling_factor).astype(int)
    # 重采样图像
    if mode == 'image':
        # 对图像使用三次或线性插值
        resampled_image = zoom(image, scaling_factor, order=3, mode='nearest')
    elif mode == 'label':
        # 对标签使用最邻近插值
        resampled_image = zoom(image, scaling_factor, order=0, mode='nearest')


    return resampled_image

def resize_image_to_target_spacing(img_data, original_spacing, target_spacing=(1, 1, 1), mode='image'):
    """
    根据指定的目标spacing重采样图像，对图像和标签使用不同的插值方法。

    参数:
    - img_data: 输入图像数据，numpy数组。
    - original_spacing: 原始图像的spacing。
    - target_spacing: 目标spacing。
    - mode: 插值模式，'image'表示图像数据，'label'表示标签数据。
    """
    # 计算重采样的比例
    resize_factor = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.round(img_data.shape * resize_factor)
    real_resize_factor = new_shape / img_data.shape
    new_spacing = original_spacing / real_resize_factor

    if mode == 'image':
        # 对图像使用三次或线性插值
        resized_img = zoom(img_data, real_resize_factor, order=3, mode='nearest')
    elif mode == 'label':
        # 对标签使用最邻近插值
        resized_img = zoom(img_data, real_resize_factor, order=0, mode='nearest')

    return resized_img

def expand_range(orig_range, total_length, expansion_ratio=0.05):
    """
    对原始裁剪范围进行扩展。

    参数:
    - orig_range: 原始裁剪范围，一个包含两个元素的列表或元组，表示起始和结束索引。
    - total_length: 在该维度上的总长度。
    - expansion_ratio: 扩展的比例，默认为5%。

    返回:
    - 扩展后的裁剪范围。
    """
    length = orig_range[1] - orig_range[0] + 1
    expand_length = int(length * expansion_ratio) // 2  # 对称扩展，因此除以2

    # 计算新的起始和结束索引
    new_start = max(orig_range[0] - expand_length, 0)
    new_end = min(orig_range[1] + expand_length, total_length - 1)

    return [new_start, new_end]


def crop_3d(image, json_file, sample):
    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        crop_ranges = json.load(f)

    # 提取原始裁剪范围并进行扩展
    x_range = expand_range(crop_ranges[sample]["X"], image.shape[0])
    y_range = expand_range(crop_ranges[sample]["Y"], image.shape[1])
    z_range = expand_range(crop_ranges[sample]["Z"], image.shape[2])

    # 进行裁剪
    cropped_image = image[x_range[0]:x_range[1] + 1, y_range[0]:y_range[1] + 1, z_range[0]:z_range[1] + 1]

    return cropped_image




def symmetric_pad(image, target_shape):
    padding = [(0, 0)] * 3
    for i in range(3):
        total_pad = target_shape[i] - image.shape[i]
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        padding[i] = (pad_before, pad_after)
    padded_image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
    return padded_image
