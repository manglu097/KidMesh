import os
import nibabel as nib

def print_nii_spacing(directory):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.nii.gz'):
            filepath = os.path.join(directory, filename)
            try:
                # 使用nibabel读取nii.gz文件
                img = nib.load(filepath)
                # 获取体素的尺寸，即spacing
                spacing = img.header.get_zooms()
                print(f'{filename}: Spacing = {spacing}')
            except Exception as e:
                print(f'Error reading {filename}: {e}')

# 替换为你的文件夹路径
directory_path = './data_final/data'
print_nii_spacing(directory_path)
