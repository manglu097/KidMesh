import os

GPU_index = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_index

from config import load_config


def main():
    # Initialize
    cfg = load_config(None)

    print("Pre-process data")
    data_obj = cfg.data_obj

    # Run pre-processing
    data = data_obj.pre_process_dataset(cfg)


if __name__ == "__main__":
    main()
