import os
import warnings
warnings.filterwarnings("ignore")
GPU_index = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_index
import logging
import torch
import numpy as np
from train import Trainer
from evaluate import Evaluator
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils_common import DataModes
import wandb
import multiprocessing
from utils.utils_common import mkdir
from config import load_config
from model.kidmesh import Kidmesh as network

logger = logging.getLogger(__name__)


def init(cfg):
    save_path = cfg.save_path + cfg.save_dir_prefix + str(cfg.experiment_idx).zfill(3)

    mkdir(save_path)

    trial_id = (len([dir for dir in os.listdir(save_path) if
                     'trial' in dir]) + 1) if cfg.trial_id is None else cfg.trial_id
    trial_save_path = save_path + '/trial_' + str(trial_id)

    if not os.path.isdir(trial_save_path):
        mkdir(trial_save_path)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True  # speeds up the computation

    return trial_save_path, trial_id


def main():
    exp_id = 100 # 实验id，方便区分不同实验

    # Initialize
    cfg = load_config(exp_id)
    trial_path, trial_id = init(cfg)

    print('Experiment ID: {}, Trial ID: {}'.format(cfg.experiment_idx, trial_id))

    print("Create network")
    model = network(cfg)
    model.cuda()
    model = torch.nn.DataParallel(model)  # , device_ids=[1][0,1]

    wandb.init(name='Experiment_{}/trial_{}'.format(cfg.experiment_idx, trial_id), project="vm-net", dir=trial_path)

    print("Initialize optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate)

    print("Load pre-processed data")
    data_obj = cfg.data_obj
    data = data_obj.quick_load_data(cfg, trial_id)

    loader = DataLoader(data[DataModes.TRAINING], batch_size=model.module.config.batch_size, shuffle=True, num_workers=12)  # batch设置为1， 因为每个提取网格的顶点数不同  , num_workers=

    print("Trainset length: {}".format(loader.__len__()))

    print("Initialize evaluator")
    evaluator = Evaluator(model, optimizer, data, trial_path, cfg, data_obj)

    print("Initialize trainer")
    trainer = Trainer(model, loader, optimizer, cfg.numb_of_itrs, cfg.eval_every, trial_path, evaluator)

    if cfg.trial_id is not None:
        print("Loading pretrained network")
        save_path = trial_path + '/best_performance/model.pth'
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0

    trainer.train(start_iteration=epoch)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
