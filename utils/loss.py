import torch
import torch.nn.functional as F
import trimesh
import numpy as np



def Dice_loss(input, target):

    smooth = 1.0
    # 对input应用sigmoid函数以转换为概率值
    input_prob = torch.sigmoid(input)

    # 由于是二分类，我们可以只取其中一个类别的概率进行计算（一般取第一个类别）
    # 因为对于二分类问题，一个类别的概率为p，则另一个类别的概率为1-p，且Dice损失相同
    input_flat = input_prob[:, 1, ...].contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    # 计算交集
    intersection = torch.sum(input_flat * target_flat)

    # 计算并集
    union = torch.sum(input_flat) + torch.sum(target_flat)

    # 计算Dice系数
    dice_coeff = (2. * intersection + smooth) / (union + smooth)

    # 计算Dice损失
    dice_loss = 1.0 - dice_coeff

    return dice_loss

# 假设 pred[0][-1][3] 是模型的预测结果，data['y_voxels'] 是标签
# 使用 Dice 损失计算


def intersection_loss(mesh):
    if mesh.is_watertight:
        # 如果网格是密封的，则假定没有自相交
        return 0
    else:
        return 1



def area_loss(meshes):
    # 提取所有的顶点和面
    verts = meshes.verts_packed()  # [N, 3] N个顶点
    faces = meshes.faces_packed()  # [M, 3] M个面，每个面由三个顶点索引组成

    # 根据面的索引提取顶点坐标
    verts_faces = verts[faces]  # [M, 3, 3]

    # 计算每个三角形面片的两个向量
    v1 = verts_faces[:, 1] - verts_faces[:, 0]  # [M, 3]
    v2 = verts_faces[:, 2] - verts_faces[:, 0]  # [M, 3]

    # 计算叉积的长度（面积的两倍）
    cross_prod = torch.cross(v1, v2, dim=1)  # [M, 3]
    area = torch.norm(cross_prod, dim=1) / 2.0  # [M]

    # 计算平均面积
    average_area = area.mean()
    return average_area