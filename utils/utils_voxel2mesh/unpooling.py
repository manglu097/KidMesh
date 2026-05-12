import torch
from scipy.spatial import ConvexHull
from itertools import combinations
from sklearn.cluster import KMeans
import numpy as np
def get_commont_vertex(edge_pair):
    a = edge_pair[:, 0] == edge_pair[:, 1]
    b = edge_pair[:, 0] == torch.flip(edge_pair[:, 1], dims=[1])

    return edge_pair[:, 0][a + b]

def uniform_unpool(vertices_, faces_, identical_face_batch=True):
    if vertices_ is None:
        return None, None
    batch_size , _, _ = vertices_.shape
    new_faces_all = []
    new_vertices_all = []

    for vertices, faces in zip(vertices_, faces_):
        face_count, _ = faces.shape
        vertices_count = len(vertices)
        edge_combinations_3 = torch.tensor(list(combinations(range(3), 2))) # 这行代码的目的是获取三个顶点中任意两个顶点的组合，以便进一步处理边的信息。
        edges = faces[:, edge_combinations_3]      # 提取出每个面片的边。   320*3*2
        unique_edges = edges.view(-1, 2)           # 960*2
        unique_edges, _ = torch.sort(unique_edges, dim=1)   # 由小索引数顶点指向大索引
        unique_edges, unique_edge_indices = torch.unique(unique_edges, return_inverse=True, dim=0)  # 一般来说共享效应，//2
        face_edges = vertices[unique_edges]   # 480*2*3

        ''' Computer new vertices '''
        new_vertices = torch.mean(face_edges, dim=1) #  找边上的中点，480*3
        new_vertices = torch.cat([vertices, new_vertices], dim=0)  # <----------------------- new vertices + old vertices   （162+480，3）
        new_vertices_all += [new_vertices[None]]

        ''' Compute new faces '''
        corner_faces = []  # 储存角面
        middle_face = []   # 储存中心面
        for j, combination in enumerate(edge_combinations_3):
            edge_pair = edges[:, combination]    # 320*2*2
            common_vertex = get_commont_vertex(edge_pair)    # 320个共同顶点

            new_vertex_1 = unique_edge_indices[torch.arange(0, 3 * face_count, 3) + combination[0]] + vertices_count
            new_vertex_2 = unique_edge_indices[torch.arange(0, 3 * face_count, 3) + combination[1]] + vertices_count

            middle_face += [new_vertex_1[:, None], new_vertex_2[:, None]]
            corner_faces += [torch.cat([common_vertex[:, None], new_vertex_1[:, None], new_vertex_2[:, None]], dim=1)]

        corner_faces = torch.cat(corner_faces, dim=0)
        middle_face = torch.cat(middle_face, dim=1)
        middle_face = torch.unique(middle_face, dim=1)
        new_faces_all += [torch.cat([corner_faces, middle_face], dim=0)[None]]  # new faces-3

        if identical_face_batch:
            new_vertices_all = new_vertices_all[0].repeat(batch_size, 1, 1)
            new_faces_all = new_faces_all[0].repeat(batch_size, 1, 1)
            break

    return new_vertices_all, new_faces_all


def adoptive_unpool(vertices, faces_prev, sphere_vertices, latent_features, N_prev):
    vertices_primary = vertices[0,:N_prev, :]    # unpooling之前的点
    vertices_secondary = vertices[0,N_prev:, :]  # unpooling之后的点
    faces_primary = faces_prev[0]

    sphere_vertices_primary = sphere_vertices[0,:N_prev]
    sphere_vertices_secondary = sphere_vertices[0,N_prev:]

    if latent_features is not None:
        latent_features_primary = latent_features[0,:N_prev]
        latent_features_secondary = latent_features[0,N_prev:]

    face_count, _ = faces_primary.shape
    vertices_count = len(vertices_primary)
    edge_combinations_3 = torch.tensor(list(combinations(range(3), 2))).cuda()
    edges = faces_primary[:, edge_combinations_3]   # 320*3*2  序号顺序
    unique_edges = edges.view(-1, 2)
    unique_edges, _ = torch.sort(unique_edges, dim=1)
    unique_edges, unique_edge_indices = torch.unique(unique_edges, return_inverse=True, dim=0)
    face_edges_primary = vertices_primary[unique_edges]    # 480*2*3

    a = face_edges_primary[:,0]    # 480*3
    b = face_edges_primary[:,1]    # 480*3
    v = vertices_secondary

    va = v - a    # 末 减 初
    vb = v - b
    ba = b - a

    cond1 = (va * ba).sum(1)    # 点集运算  480
    norm1 = torch.norm(va, dim=1)   # 计算范数

    cond2 = (vb * ba).sum(1)
    norm2 = torch.norm(vb, dim=1)

    dist = torch.norm(torch.cross(va, ba), dim=1)/torch.norm(ba, dim=1)  # 其模长等于va和ba构成的平行四边形的面积。然后，将这个面积除以ba的长度（即边缘的长度），得到v到边缘ab的最短距离。  480
    dist[cond1 < 0] = norm1[cond1 < 0]    #  角度大于90的，其到ab的最短距离
    dist[cond2 < 0] = norm2[cond2 < 0]    #  原来dist[cond2 < 0] = norm2[cond2 < 0]

    # threshold = find_threshold(dist)
        # 挑阈值，可以进行消融
    sorted_, _ = torch.sort(dist)
    threshold = sorted_[int(0.3*len(sorted_))] 

    vertices_needed = vertices_secondary[dist > threshold]
    
    sphere_vertices_needed = sphere_vertices_secondary[dist > threshold] 
    if latent_features is not None:
        latent_features_needed = latent_features_secondary[dist > threshold]

    vertices = torch.cat([vertices_primary,vertices_needed],dim=0)[None]
    if latent_features is not None:
        latent_features = torch.cat([latent_features_primary,latent_features_needed],dim=0)[None]
    sphere_vertices = torch.cat([sphere_vertices_primary,sphere_vertices_needed],dim=0)

    sphere_vertices = sphere_vertices/torch.sqrt(torch.sum(sphere_vertices**2,dim=1)[:,None])

    hull = ConvexHull(sphere_vertices.data.cpu().numpy())    # sphere_vertices 和 vertices 在这里是一样的
    faces = torch.from_numpy(hull.simplices).long().cuda()[None]   # 即凸包中的简单形状（通常是三角形）的顶点索引。
    # 点集Q的凸包（convex hull）是指一个最小凸多边形，满足Q中的点或者在多边形边上或者在其内。图1中由红色线段表示的多边形就是点集Q={p0,p1,...p12}的凸包。
    # 如果把一个多边形的所有边中，有一条边向两方无限延长成为一直线时，其他各边都在此直线的同旁，那么这个多边形就叫做凸多边形
    sphere_vertices = sphere_vertices[None]  

    return vertices, faces, latent_features, sphere_vertices




def find_threshold(dist):
    # 将一维张量转换为 numpy 数组，以适配 KMeans
    dist_reshaped = dist.view(-1, 1).cpu().detach().numpy()

    # 使用 KMeans 分成三类
    kmeans = KMeans(n_clusters=3, random_state=0).fit(dist_reshaped)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_.flatten()

    # 排序聚类中心，找到小中大的边界
    sorted_centers = np.sort(cluster_centers)
    small_medium_threshold = np.mean(sorted_centers[:2])

    # 计算小类数量占总数的比例
    small_class_ratio = np.sum(labels == np.argmin(cluster_centers)) / len(dist)

    # 根据小类数量选择阈值
    if small_class_ratio > 0.3:
        # 如果小类超过30%，使用最小的30%的值作为阈值
        threshold_value = np.percentile(dist_reshaped, 30)
    else:
        # 否则使用小类和中类之间的分界作为阈值
        threshold_value = small_medium_threshold

    # 将阈值转换回0维张量
    threshold_tensor = torch.tensor(threshold_value, dtype=dist.dtype)

    return threshold_tensor

# 假设 `dist` 是你的一维张量输入
# threshold = find_threshold(dist)
# print(threshold)

