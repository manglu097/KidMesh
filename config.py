from data.KD import KD

class Config():
    def __init__(self):
        super(Config, self).__init__()


def load_config(exp_id):
    cfg = Config()
    ''' Experiment '''
    cfg.experiment_idx = exp_id
    cfg.trial_id = None

    cfg.save_dir_prefix = 'Experiment_'
    cfg.name = 'voxel2mesh'

   # 数据集参数设置
    cfg.save_path = "./data/result"
    cfg.dataset_path = "./data/dataset/data_final"
    cfg.data_obj = KD()
    assert cfg.save_path != None, "Set cfg.save_path in config.py"
    assert cfg.dataset_path != None, "Set cfg.dataset_path in config.py"
    assert cfg.data_obj != None, "Set cfg.data_obj in config.py"


    # 数据参数、网络参数、训练参数设置
    # input should be cubic. Otherwise, input should be padded accordingly.
    cfg.output_shape = (128, 128, 128)
    cfg.pad_shape = (128, 128, 128) #(192, 192, 192)

    cfg.ndims = 3
    cfg.augmentation_shift_range = 10

    ''' Model '''
    cfg.first_layer_channels = 16
    cfg.num_input_channels = 1
    cfg.steps = 4

    # Only supports batch size 1 at the moment. 
    cfg.batch_size = 1

    cfg.num_classes = 2
    cfg.batch_norm = True
    cfg.graph_conv_layer_count = 4

    ''' Optimizer '''
    cfg.learning_rate = 1e-4
    ''' Training '''
    cfg.numb_of_itrs = 20500
    cfg.eval_every = 1

    return cfg
