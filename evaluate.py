from utils.utils_common import DataModes, mkdir, blend, crop_indices, blend_cpu, append_line, write_lines
from utils.utils_voxel2mesh.file_handle import save_to_obj
from torch.utils.data import DataLoader
import numpy as np
import torch
import time
import torch.nn.functional as F
import nibabel as nib
import wandb
from utils.rasterize.rasterize import Rasterize


class Structure(object):

    def __init__(self, voxel=None, mesh=None, points=None):
        self.voxel = voxel
        self.mesh = mesh
        self.points = points


def write_to_wandb(writer, epoch, split, performences, num_classes):
    log_vals = {}
    for key, value in performences[split].items():
        log_vals[key + '/mean'] = np.mean(performences[split][key])  # 记录这次测试，所有测试集的平均指标
    try:
        wandb.log(log_vals)
    except:
        print('')


class Evaluator(object):
    def \
            __init__(self, net, optimizer, data, save_path, config, support):
        self.data = data
        self.net = net
        self.current_best = None
        self.save_path = save_path + '/best_performance'
        self.latest = save_path + '/latest'
        self.optimizer = optimizer
        self.config = config
        self.support = support
        self.count = 0
        self.net = self.net.eval()

    def save_model(self, epoch):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.save_path + '/model.pth')

    def evaluate(self, epoch, writer=None, backup_writer=None):
        # self.net = self.net.eval()
        performences = {}
        predictions = {}

        for split in [DataModes.TESTING]:
            dataloader = DataLoader(self.data[split], batch_size=1, shuffle=False, num_workers=8) # , num_workers=24
            performences[split], predictions[split], Time = self.evaluate_set(dataloader)  # 评价测试集
            print("AVG_time: {} ".format(np.mean(Time)))
            write_to_wandb(writer, epoch, split, performences, self.config.num_classes)

        if self.support.update_checkpoint(best_so_far=self.current_best, new_value=performences):  # 查看是不是目前最好的模型

            mkdir(self.save_path)
            mkdir(self.save_path + '/mesh_cfd')
            mkdir(self.save_path + '/voxels_cfd')

            self.save_model(epoch)  # 会进行替代
            self.save_results(predictions[DataModes.TESTING], epoch, performences[DataModes.TESTING], self.save_path,
                              '/testing_')
            self.current_best = performences  # 每一次评级验证，次变量不会消失，因为没有新建对象

    def predict(self, data, config):  # 其实就是预测完结果再准备数据
        name = config.name
        if name == 'unet':
            y_hat = self.net(data)
            y_hat = torch.argmax(y_hat, dim=1).cpu()

            x = data['x']
            y = Structure(voxel=data['y_voxels'].cpu())
            y_hat = Structure(voxel=y_hat)

        elif name == 'voxel2mesh':

            x = data['x']
            start_time = time.time()
            pred = self.net(data)
            elapsed_time = time.time() - start_time


            pred_meshes = []
            true_meshes = []
            true_points = []
            pred_voxels = torch.zeros_like(x)[:, 0].long()  # 用于存储栅格化（即转换为体素表示）后的预测。用于存储栅格化（即转换为体素表示）后的预测。

            for c in range(self.config.num_classes - 1):

                pred_vertices = pred[c][-1][0].detach().data.cpu()  # 预测的最终顶点
                pred_faces = pred[c][-1][1].detach().data.cpu()
                true_vertices = data['vertices_mc'][c].data.cpu()
                true_faces = data['faces_mc'][c].data.cpu()

                pred_meshes += [{'vertices': pred_vertices, 'faces': pred_faces, 'normals': None}]
                true_meshes += [{'vertices': true_vertices, 'faces': true_faces, 'normals': None}]
                true_points += [data['surface_points'][c].data.cpu()]

                _, _, D, H, W = x.shape
                shape = torch.tensor([D, H, W]).int().cuda()
                rasterizer = Rasterize(shape)
                pred_voxels_rasterized = rasterizer(pred_vertices, pred_faces).long()

                pred_voxels[pred_voxels_rasterized == 1] = c + 1

            true_voxels = data['y_voxels'].data.cpu()

            x = x.detach().data.cpu()
            y = Structure(mesh=true_meshes, voxel=true_voxels, points=true_points)
            y_hat = Structure(mesh=pred_meshes, voxel=pred_voxels_rasterized)

        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))  # 为什么再来一次标准化
        return x, y, y_hat, elapsed_time

    def evaluate_set(self, dataloader):  # 拿到结果。然后进行评价
        performance = {}
        predictions = []
        Time = []

        for i, data in enumerate(dataloader):  # 真正的数据迭代

            x, y, y_hat, elapsed_time = self.predict(data, self.config)  # 预测结果
            Time.append(elapsed_time)
            result = self.support.evaluate(y, y_hat, self.config)

            predictions.append((x, y, y_hat))

            for key, value in result.items():
                if key not in performance:
                    performance[key] = []
                performance[key].append(result[key])

        for key, value in performance.items():
            performance[key] = np.array(performance[key])
        return performance, predictions, Time

    def save_results(self, predictions, epoch, performence, save_path, mode):

        xs = []
        ys_voxels = []
        y_hats_voxels = []

        for i, data in enumerate(predictions):
            x, y, y_hat = data

            xs.append(x[0, 0])

            if y_hat.points is not None:
                for p, (true_points, pred_points) in enumerate(zip(y.points, y_hat.points)):
                    save_to_obj(save_path + '/points/' + mode + 'true_' + str(i) + '_part_' + str(p) + '.obj',
                                true_points, [])
                    if pred_points.shape[1] > 0:
                        save_to_obj(save_path + '/points/' + mode + 'pred_' + str(i) + '_part_' + str(p) + '.obj',
                                    pred_points, [])

            if y_hat.mesh is not None:
                for p, (true_mesh, pred_mesh) in enumerate(zip(y.mesh, y_hat.mesh)):  # p=0

                    save_to_obj(save_path + '/mesh_cfd/' + 'true_' + str(i) + '.obj',   #MC 处理的
                                true_mesh['vertices'], true_mesh['faces'], true_mesh['normals'])
                    save_to_obj(save_path + '/mesh_cfd/' + 'pred_' + str(i) + '.obj',
                                pred_mesh['vertices'], pred_mesh['faces'], pred_mesh['normals'])

            if y_hat.voxel is not None:

                for p, (y_real, y_pred) in enumerate(zip(y.voxel,y_hat.voxel)):  # p=0

                    y_real = F.upsample(y_real[None, None].float(), size=x[0, 0].shape, mode='nearest')[0, 0].type(torch.int16)
                    y_pred = F.upsample(y_pred[None, None].float(), size=x[0, 0].shape, mode='nearest')[0, 0].type(torch.int16)


                    # 创建一个NIfTI图像。这里没有显式指定仿射矩阵，所以使用单位矩阵
                    nifti_img1 = nib.Nifti1Image(y_real.cpu().detach().numpy(), affine=np.eye(4))
                    nifti_img2 = nib.Nifti1Image(y_pred.cpu().detach().numpy(), affine=np.eye(4))

                    # 保存NIfTI图像为 nii.gz 格式
                    nib.save(nifti_img1, save_path + '/voxels_cfd/' + 'true_' + str(i) + '.nii.gz')
                    nib.save(nifti_img2, save_path + '/voxels_cfd/' + 'pred_' + str(i) + '.nii.gz')


        if performence is not None:  # 性能记录
            for key, value in performence.items():
                performence_mean = np.mean(performence[key], axis=0)
                performence_variance = np.var(performence[key], axis=0)
                performence_std = np.std(performence[key], axis=0)

                summary = ('{}: ' + ', '.join(['{:.8f} +- var: {:.8f} +- std: {:.8f}' for _ in range(self.config.num_classes - 1)])).format(epoch,
                                                                                                              *performence_mean, *performence_variance, *performence_std)
                append_line(save_path + mode + 'summary' + key + '.txt', summary)
                print(('{} {}: ' + ', '.join(['{:.8f}' for _ in range(self.config.num_classes - 1)])).format(epoch, key,
                                                                                                             *performence_mean))

