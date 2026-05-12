import numpy as np
import nibabel as nib
from scipy import ndimage
import os
import json

def find_largest_region(mask):
    """
    找到给定掩码中的最大连通区域。

    参数:
    - mask: 二维或三维的numpy数组，其中0表示背景，1表示目标区域。

    返回值:
    - largest_region_mask: 最大连通区域的二值化掩码。
    - bbox: 最大连通区域的边界框信息，格式为((min_x, max_x), (min_y, max_y), (min_z, max_z))。
    """
    # 使用label函数标记连通区域
    labeled_mask, regions_count = ndimage.label(mask)

    # 计算每个区域的面积
    region_sizes = np.bincount(labeled_mask.flatten())

    # 找到最大连通区域的标签
    largest_region_label = np.argmax(region_sizes[1:]) + 1

    # 提取最大连通区域的二值化掩码
    largest_region_mask = labeled_mask == largest_region_label

    # 找到最大连通区域的边界框
    bbox = ndimage.find_objects(labeled_mask == largest_region_label)[0]

    return largest_region_mask, bbox

samples_dir = './data_final/seg'
samples = [sample for sample in os.listdir(samples_dir)]

# 保存位置信息的列表
positions = {}

# 找到每个样本的最大连通区域
for sample in samples:
    # 读取NIfTI文件
    nii_image = nib.load(os.path.join(samples_dir, sample))
    image_data = nii_image.get_fdata()

    # 将影像数据转换为二值掩码
    binary_mask = np.where(image_data > 0, 1, 0)

    # 找到最大连通区域和边界框
    largest_region_mask, bbox = find_largest_region(binary_mask)

    # 计算边界框的XYZ范围
    x_range = (bbox[0].start, bbox[0].stop)
    y_range = (bbox[1].start, bbox[1].stop)
    z_range = (bbox[2].start, bbox[2].stop)

    # 将范围信息保存到字典中
    positions[sample] = {'X': x_range, 'Y': y_range, 'Z': z_range}

# 保存为JSON文件
json_file = './positions.json'
with open(json_file, 'w') as f:
    json.dump(positions, f)

print(f"Positions saved to {json_file}")

# 输出能包裹所有数据最大连通区域的XYZ范围
min_x = min(pos["X"][0] for pos in positions.values())
max_x = max(pos["X"][1] for pos in positions.values())
min_y = min(pos["Y"][0] for pos in positions.values())
max_y = max(pos["Y"][1] for pos in positions.values())
min_z = min(pos["Z"][0] for pos in positions.values())
max_z = max(pos["Z"][1] for pos in positions.values())


max1 = max(pos["X"][1] - pos["X"][0]for pos in positions.values())

max2 = max(pos["Y"][1] - pos["Y"][0] for pos in positions.values())

max3 = max(pos["Z"][1] - pos["Z"][0] for pos in positions.values())

min1 = min(pos["X"][1] - pos["X"][0]for pos in positions.values())

min2 = min(pos["Y"][1] - pos["Y"][0] for pos in positions.values())

min3 = min(pos["Z"][1] - pos["Z"][0] for pos in positions.values())

x_lengths = [pos["X"][1] - pos["X"][0] for pos in positions.values()]

average_x_length = np.mean(x_lengths)

y_lengths = [pos["Y"][1] - pos["Y"][0] for pos in positions.values()]

average_y_length = np.mean(y_lengths)

z_lengths = [pos["Z"][1] - pos["Z"][0] for pos in positions.values()]

average_z_length = np.mean(z_lengths)

print(f"All data: X: ({min_x}, {max_x}), Y: ({min_y}, {max_y}), Z: ({min_z}, {max_z})")


print(f"EVERY data: XYZ: ({max1}, {max2}, {max3})")
print(f"EVERY data: XYZ_min: ({min1}, {min2}, {min3})")
print(f"EVERY data: XYZ_mean: ({average_x_length}, {average_y_length}, {average_z_length})")