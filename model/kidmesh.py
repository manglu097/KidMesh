from itertools import chain
import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency)
import trimesh
from utils.utils_common import crop_and_merge
from utils.utils_voxel2mesh.graph_conv import adjacency_matrix, Features2Features, Feature2VertexLayer, \
    Features2FeaturesWithResidual
from utils.utils_voxel2mesh.feature_sampling import LearntNeighbourhoodSampling
from utils.utils_voxel2mesh.file_handle import read_obj
from utils.utils_voxel2mesh.fusion_feature import FeatureFusionTransformer
from utils.utils_voxel2mesh.unpooling import uniform_unpool, adoptive_unpool

from utils.utils_unet import UNetLayer, ResUNetLayer
from utils.loss import intersection_loss, Dice_loss, area_loss


class Kidmesh(nn.Module):
    """ Voxel2Mesh  """

    def __init__(self, config):
        super(Kidmesh, self).__init__()

        self.config = config

        self.max_pool = nn.MaxPool3d(2) if config.ndims == 3 else nn.MaxPool2d(2)

        ConvLayer = nn.Conv3d if config.ndims == 3 else nn.Conv2d
        ConvTransposeLayer = nn.ConvTranspose3d if config.ndims == 3 else nn.ConvTranspose2d
        batch_size = config.batch_size

        '''  Down layers '''
        down_layers = [ResUNetLayer(config.num_input_channels, config.first_layer_channels, config.ndims)]
        for i in range(1, config.steps + 1):
            graph_conv_layer = ResUNetLayer(config.first_layer_channels * 2 ** (i - 1),
                                            config.first_layer_channels * 2 ** i, config.ndims)
            down_layers.append(graph_conv_layer)
        self.down_layers = down_layers
        self.encoder = nn.Sequential(*down_layers)

        ''' Up layers '''
        self.skip_count = []
        self.latent_features_coount = []
        for i in range(config.steps + 1):
            self.skip_count += [config.first_layer_channels * 2 ** (config.steps - i)]  # 256
            self.latent_features_coount += [32]

        dim = 3

        up_std_conv_layers = []
        up_f2f_layers = []
        up_f2v_layers = []
        for i in range(config.steps + 1):
            feature2feature_layers = []
            feature2vertex_layers = []
            skip = LearntNeighbourhoodSampling(config, self.skip_count[i], i)
            fusion = FeatureFusionTransformer(self.skip_count[i])

            # lyr = Feature2VertexLayer(self.skip_count[i])
            if i == 0:
                grid_upconv_layer = None
                grid_unet_layer = None
                for k in range(config.num_classes - 1):
                    feature2feature_layers += [
                        Features2FeaturesWithResidual(self.skip_count[i] + dim + self.latent_features_coount[0] ,  #
                                                      self.latent_features_coount[i],  # batch_norm=False  注意修改
                                                      hidden_layer_count=config.graph_conv_layer_count)]  # , graph_conv=GraphConv

            else:
                grid_upconv_layer = ConvTransposeLayer(
                    in_channels=config.first_layer_channels * 2 ** (config.steps - i + 1),
                    out_channels=config.first_layer_channels * 2 ** (config.steps - i), kernel_size=2, stride=2)
                grid_unet_layer = ResUNetLayer(config.first_layer_channels * 2 ** (config.steps - i + 1),
                                               config.first_layer_channels * 2 ** (config.steps - i), config.ndims,
                                               config.batch_norm)
                for k in range(config.num_classes - 1):
                    feature2feature_layers += [
                        Features2FeaturesWithResidual(self.skip_count[i] + self.latent_features_coount[i - 1] + dim,
                                                      # ！！！！！！！！
                                                      self.latent_features_coount[i],
                                                      hidden_layer_count=config.graph_conv_layer_count)]  # , graph_conv=GraphConv if i < config.steps else GraphConvNoNeighbours

            for k in range(config.num_classes - 1):
                feature2vertex_layers += [
                    Feature2VertexLayer(self.latent_features_coount[i], 3)]  # batch_norm =False  注意修改

            up_std_conv_layers.append((skip, grid_upconv_layer, grid_unet_layer, fusion))
            up_f2f_layers.append(feature2feature_layers)
            up_f2v_layers.append(feature2vertex_layers)

        self.up_std_conv_layers = up_std_conv_layers
        self.up_f2f_layers = up_f2f_layers
        self.up_f2v_layers = up_f2v_layers

        self.decoder_std_conv = nn.Sequential(*chain(*up_std_conv_layers))   # 不能删除！！！！！！！！！！！！！！！
        self.decoder_f2f = nn.Sequential(*chain(*up_f2f_layers))
        self.decoder_f2v = nn.Sequential(*chain(*up_f2v_layers))

        ''' Final layer (for voxel decoder)'''
        self.final_layer = ConvLayer(in_channels=config.first_layer_channels, out_channels=config.num_classes,
                                     kernel_size=1)
        self.deep_supervised_layer1 = ConvLayer(in_channels=config.first_layer_channels * 2,
                                                out_channels=config.num_classes,
                                                kernel_size=1)
        self.deep_supervised_layer2 = ConvLayer(in_channels=config.first_layer_channels * 4,
                                                out_channels=config.num_classes,
                                                kernel_size=1)
        self.first_GNN = Features2Features(self.config.ndims, self.latent_features_coount[0],
                                           hidden_layer_count=config.graph_conv_layer_count)   # 注意定义的位置

        sphere_path = './spheres/kidney_left_{}_N.obj'.format(162)
        sphere_vertices, sphere_faces = read_obj(sphere_path)
        sphere_vertices = torch.from_numpy(
            sphere_vertices).cuda().float()  # 其形状为(V, 3)，其中V是顶点的数量。每一行代表一个顶点，包含三个浮点数，分别是该顶点的x、y和z坐标。
        self.sphere_vertices = sphere_vertices / torch.sqrt(torch.sum(sphere_vertices ** 2, dim=1)[:, None])[
            None]  # 顶点规范化到单位球面上的坐标。这个过程确保了每个顶点到原点的距离都为1，即所有顶点都位于半径为1的球面上。
        self.sphere_faces = torch.from_numpy(sphere_faces).cuda().long()[
            None]  # 其形状为(F, 3)，其中F是面的数量。每一行代表一个面，包含三个整数，这三个整数是构成该面的三个顶点在sphere_vertices数组中的索引。

    def forward(self, data):

        x = data['x']  # 数据是字典
        unpool_indices = data['unpool']
        #unpool_indices = [0, 1, 0, 1, 1]

        sphere_vertices = self.sphere_vertices.clone()
        vertices = sphere_vertices.clone()
        faces = self.sphere_faces.clone()
        batch_size = self.config.batch_size

        # first layer
        x = self.down_layers[0](x)
        down_outputs = [x]  # 将每次下采样的结果保存至列表

        # down layers
        for unet_layer in self.down_layers[1:]:
            x = self.max_pool(x)
            x = unet_layer(x)
            down_outputs.append(x)

        A, D = adjacency_matrix(vertices, faces)
        latent_features = self.first_GNN(vertices, A, D, vertices, faces)

        pred = [None] * self.config.num_classes
        for k in range(self.config.num_classes - 1):
            pred[k] = [[vertices.clone(), faces.clone(), latent_features.clone(), None, sphere_vertices.clone()]]

        for i, (
                (skip_connection, grid_upconv_layer, grid_unet_layer, fusion_2), up_f2f_layers, up_f2v_layers,
                down_output, skip_amount,
                do_unpool) in enumerate(
            zip(self.up_std_conv_layers, self.up_f2f_layers, self.up_f2v_layers, down_outputs[::-1],
                self.skip_count, unpool_indices)):
            if grid_upconv_layer is not None and i > 0:
                x = grid_upconv_layer(x)
                x = crop_and_merge(down_output, x)  # 真正的跳跃连接
                x = grid_unet_layer(x)  # 普通 conv block
            elif grid_upconv_layer is None:
                x = down_output

            for k in range(self.config.num_classes - 1):

                # load mesh information from previous iteratioin for class k
                vertices = pred[k][i][0]
                faces = pred[k][i][1]
                latent_features = pred[k][i][2]
                sphere_vertices = pred[k][i][4]
                feature2feature = up_f2f_layers[k]
                feature2vertex = up_f2v_layers[k]

                if do_unpool[0] == 1: #do_unpool.item() ==1:      # 先均匀采样在进行一系列操作
                    faces_prev = faces
                    _, N_prev, _ = vertices.shape

                    # Get candidate vertices using uniform unpool
                    vertices, faces_ = uniform_unpool(vertices, faces)
                    latent_features, _ = uniform_unpool(latent_features,
                                                        faces)  # 其实跟顶点应该是同一原理   1*642*32 ；向量维度不变，数量要紧跟顶点数
                    sphere_vertices, _ = uniform_unpool(sphere_vertices, faces)
                    faces = faces_

                A, D = adjacency_matrix(vertices, faces)

                skipped_features1 = skip_connection(x[:, :skip_amount], vertices)  # 将维度控制一下
                skipped_features2 = skip_connection(down_output[:, :skip_amount], vertices)
                # skipped_features = (skipped_features1 + skipped_features2) / 2.0
                skipped_features = fusion_2([skipped_features1, skipped_features2])

                latent_features = torch.cat([latent_features, skipped_features, vertices],
                                            dim=2) if latent_features is not None else torch.cat(
                    [skipped_features, vertices], dim=2)

                latent_features = feature2feature(latent_features, A, D, vertices, faces)
                deltaV = feature2vertex(latent_features, A, D, vertices, faces)
                vertices = vertices + deltaV

                if do_unpool[0] == 1:  #do_unpool.item() ==1:  # 摒弃均匀采样中变形不大的点
                    vertices, faces, latent_features, sphere_vertices = adoptive_unpool(vertices, faces_prev,
                                                                                        # 仔细再看！！！！
                                                                                        sphere_vertices,
                                                                                        latent_features, N_prev)

                voxel_pred = self.final_layer(x) if i == len(self.up_std_conv_layers) - 1 else None

                # 深度分支
                if i == len(self.up_std_conv_layers) - 2:
                    voxel_pred = F.interpolate(x, size=self.config.output_shape, mode='trilinear', align_corners=True)
                    voxel_pred = self.deep_supervised_layer1(voxel_pred)

                if i == len(self.up_std_conv_layers) - 3:
                    voxel_pred = F.interpolate(x, size=self.config.output_shape, mode='trilinear', align_corners=True)
                    voxel_pred = self.deep_supervised_layer2(voxel_pred)

                pred[k] += [[vertices, faces, latent_features, voxel_pred, sphere_vertices]]


        return pred

    def loss(self, data, epoch):

        pred = self.forward(data)
        # embed()

        CE_Loss = nn.CrossEntropyLoss()  # 自带softmax
        dice_loss_main = Dice_loss(pred[0][-1][3], data['y_voxels'])
        ce_loss_main = CE_Loss(pred[0][-1][3], data['y_voxels'])
        # dice_loss_aux1 = Dice_loss(pred[0][-2][3], data['y_voxels'])
        # ce_loss_aux1 = CE_Loss(pred[0][-2][3], data['y_voxels'])
        # dice_loss_aux2 = Dice_loss(pred[0][-3][3], data['y_voxels'])
        # ce_loss_aux2 = CE_Loss(pred[0][-3][3], data['y_voxels'])
        # ce_loss = ((0.8 + (epoch / (self.config.numb_of_itrs // 40)) * 0.2) * ce_loss_main + (
        #             (1 - epoch / (self.config.numb_of_itrs // 40)) * 0.1) * ce_loss_aux1
        #            + ((1 - epoch / (self.config.numb_of_itrs // 40)) * 0.1) * ce_loss_aux2)
        #
        # dice_loss = ((0.8 + (epoch / (self.config.numb_of_itrs // 40)) * 0.2) * dice_loss_main + (
        #             (1 - epoch / (self.config.numb_of_itrs // 40)) * 0.1) * dice_loss_aux1
        #              + ((1 - epoch / (self.config.numb_of_itrs // 40)) * 0.1) * dice_loss_aux2)



        chamfer_loss1 = torch.tensor(0).float().cuda()
        chamfer_loss2 = torch.tensor(0).float().cuda()
        edge_loss = torch.tensor(0).float().cuda()
        laplacian_loss = torch.tensor(0).float().cuda()
        normal_consistency_loss = torch.tensor(0).float().cuda()
        is_loss = torch.tensor(0).float().cuda()
        ar_loss = torch.tensor(0).float().cuda()

        for c in range(self.config.num_classes - 1):
            target1 = data['vertices_mc'][c].cuda()
            # target2 = data['surface_points'][c].cuda()
            for k, (vertices, faces, _, _, _) in enumerate(pred[c][1:]):  #  Ablation： 不是逐步计算损失
                pred_mesh = Meshes(verts=list(vertices), faces=list(faces))
                pred_points = sample_points_from_meshes(pred_mesh)  #  target1.shape[0]

                vertices_np = vertices.squeeze(0).cpu().detach().numpy()
                faces_np = faces.squeeze(0).cpu().detach().numpy()
                tri_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)

                chamfer_loss1 += chamfer_distance(pred_points, target1)[0]
                # chamfer_loss2 += chamfer_distance(pred_points, target2)[0]
                laplacian_loss += mesh_laplacian_smoothing(pred_mesh, method="uniform")
                normal_consistency_loss += mesh_normal_consistency(pred_mesh)
                edge_loss += mesh_edge_loss(pred_mesh)
                is_loss += intersection_loss(tri_mesh)
                ar_loss += area_loss(pred_mesh)

        loss = (1 * chamfer_loss1 +  0.5 * dice_loss_main +  0.5 * ce_loss_main  # 1 * chamfer_loss2   +
                + 0.1 * laplacian_loss + 0.1 * edge_loss + 0.1 * normal_consistency_loss + 0.1 * is_loss + 1 * ar_loss)

        log = {"loss": loss.detach(),
               # "chamfer_loss_sp": chamfer_loss2.detach(),
               "chamfer_loss_mc": chamfer_loss1.detach(),
               "ce_loss": ce_loss_main.detach(),
               "dice_loss": dice_loss_main.detach(),
               "normal_consistency_loss": normal_consistency_loss.detach(),
               "edge_loss": edge_loss.detach(),
               "laplacian_loss": laplacian_loss.detach(),
               "is_loss": is_loss.detach(),
               "area_loss": ar_loss.detach()}
        if epoch % 100 == 0:
            print(log)
        return loss, log
