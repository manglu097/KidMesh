import numpy as np
import trimesh
import torch.nn.functional as F
import torch


def rmse_all(target, pred, num_classes):
    rmse = []
    pred = pred
    target = target

    for cls in range(1, num_classes):
        rmse.append(torch.sqrt(torch.mean((pred[:, cls] - target[:, cls])**2)).data.cpu().numpy())

    return np.array(rmse)

def jaccard_index(target, pred, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, num_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = pred_inds[target_inds].long().sum().data.cpu()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu() + target_inds.long().sum().data.cpu() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(union))
    return np.array(ious)





def calculate_dice_iou_voe(target, pred, num_classes):
    dice_scores = []
    ious = []
    voes = []  # 初始化VOE列表
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore metrics for background class ("0")
    for cls in range(1, num_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = pred_inds[target_inds].long().sum().item()  # Use item() to get a Python number

        # Calculate union for IoU
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection

        # Calculate Dice, IoU, and VOE, handle division by zero
        if union == 0:
            dice_scores.append(float('nan'))
            ious.append(float('nan'))
            voes.append(float('nan'))  # VOE也设为NaN
        else:
            dice_score = 2 * float(intersection) / (pred_inds.long().sum().item() + target_inds.long().sum().item())
            iou = float(intersection) / float(union)
            voe = (1 - float(intersection) / float(union)) * 100  # 计算VOE

            dice_scores.append(dice_score)
            ious.append(iou)
            voes.append(voe)  # 将VOE添加到列表中

    return np.array(dice_scores), np.array(ious), np.array(voes)  # 返回包含VOE的结果



def chamfer_directed(A, B):


    N1 = A.shape[1]
    N2 = B.shape[1]

    if N1 > 0 and N2 > 0:
        y1 = A[:, :, None].repeat(1, 1, N2, 1)
        y2 = B[:, None].repeat(1, N1, 1, 1)

        diff = torch.sum((y1 - y2) ** 2, dim=3)

        loss, _ = torch.min(diff, dim=2)

        loss = torch.mean(loss)
    else:
        loss = torch.Tensor([float("Inf")]).cuda() if A.is_cuda else torch.Tensor([float("Inf")])



    return loss

def chamfer_symmetric(A, B):
    losses = []
    diff = calculate_distances(A, B)

    loss1, _ = torch.min(diff, dim=1)
    loss2, _ = torch.min(diff, dim=2)

    loss = torch.sum(loss1) + torch.sum(loss2)
    num_points = A.shape[1] + B.shape[1]
    mean_loss = loss / num_points
    losses.append(mean_loss)
    
    return np.array(losses)


def p2s_distance(A, B):
    # 计算B到A的每个点的距离，这里B是预测点集，A是真实点集

    p2ses = []
    diff = calculate_distances(B, A)  # 注意这里首先传入B，然后是A

    # 对于B中的每个点，找到距离A中最近点的距离
    loss, _ = torch.min(diff, dim=2)  # 此处沿着dim=2寻找最小值意味着对每个预测点寻找最近的真实点

    # 计算平均损失
    mean_loss = torch.mean(loss)
    p2ses.append(mean_loss.item())

    return p2ses


def calculate_distances(A, B):


    N1 = A.shape[1]
    N2 = B.shape[1]
    y1 = A[:, :, None].repeat(1, 1, N2, 1)
    y2 = B[:, None].repeat(1, N1, 1, 1)

    diff = torch.sqrt(torch.sum((y1 - y2) ** 2, dim=3))  # 计算欧几里得距离

    return diff


def ASSD(A, B):
    assdes = []
    diff = calculate_distances(A, B)

    min_dist_A_to_B, _ = torch.min(diff, dim=2)  # A到B的最小距离
    min_dist_B_to_A, _ = torch.min(diff, dim=1)  # B到A的最小距离

    assd = (torch.sum(min_dist_A_to_B) + torch.sum(min_dist_B_to_A)) / (
                min_dist_A_to_B.numel() + min_dist_B_to_A.numel())
    assdes.append(assd)
    return np.array(assdes)


def HD90(A, B):
    HDes = []
    diff = calculate_distances(A, B)

    min_dist_A_to_B, _ = torch.min(diff, dim=2)
    min_dist_B_to_A, _ = torch.min(diff, dim=1)

    all_distances = torch.cat((min_dist_A_to_B.flatten(), min_dist_B_to_A.flatten()))
    hd90 = torch.quantile(all_distances, 0.9)  # 计算90%分位数
    HDes.append(hd90)
    return np.array(HDes)


def chamfer_weighted_symmetric(A, B):   # 还没有加入权重

    N1 = A.shape[1]
    N2 = B.shape[1]
    y1 = A[:, :, None].repeat(1, 1, N2, 1)
    y2 = B[:, None].repeat(1, N1, 1, 1)

    diff = torch.sum((y1 - y2) ** 2, dim=3)

    loss1, _ = torch.min(diff, dim=1)
    loss2, _ = torch.min(diff, dim=2)
    loss = torch.mean(loss1) + torch.mean(loss2)
    return loss

def chamfer_weighted_symmetric_with_dtf(A, B, B_dtf):

    N1 = A.shape[1]
    N2 = B.shape[1]
    y1 = A[:, :, None].repeat(1, 1, N2, 1)
    y2 = B[:, None].repeat(1, N1, 1, 1)

    diff = torch.sum((y1 - y2) ** 2, dim=3) 
    loss1, _ = torch.min(diff, dim=1)
    # loss2, _ = torch.min(diff, dim=2)
    A_ = A[:, :, None, None]  
    loss2 = F.grid_sample(B_dtf, A_, mode='bilinear', padding_mode='border', align_corners=True)

    loss = torch.mean(loss1) + torch.mean(loss2)
    return loss

def rmse(target, pred):
    return torch.sqrt(torch.sum((target - pred)**2))

def angle_error(target, pred):
    target = target.data.cpu().numpy()
    pred = pred.data.cpu().numpy()
    angle = np.arccos(np.dot(target, pred) / (np.linalg.norm(target) * np.linalg.norm(pred)))
    return angle


def evaluate_mesh_topology(vertices, faces):
    ISes = []
    Holes = []
    # 将顶点和面集转换成trimesh中的Mesh对象
    vertices_np = vertices.squeeze(0).numpy()
    faces_np = faces.squeeze(0).numpy()  # .cpu().detach()
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)

    # 检查网格是否有自相交
    has_self_intersections = mesh.is_watertight

    # 计算自相交的面的比例
    intersecting_faces = trimesh.repair.broken_faces(mesh, color=True)
    intersecting_ratio = 100 * len(intersecting_faces) / len(mesh.faces) if mesh.faces.size > 0 else 0

    # # 检查网格的空洞数
    # holes = mesh.facets_boundary
    #
    # number_of_holes = len(holes)

    mesh_filled = mesh.copy()
    mesh_filled.fill_holes()

    # 通过比较填充前后的面的数量来估计空洞的数量
    number_of_holes = len(mesh_filled.faces) - len(mesh.faces)

    ISes.append(intersecting_ratio)
    Holes.append(number_of_holes)

    # 返回评估结果
    return  np.array(ISes), np.array(Holes)