# KidMesh: Computational Mesh Reconstruction for Pediatric Congenital Hydronephrosis Using Deep Neural Networks

<p align="center">
  <img src="assets/architecture.png" alt="KidMesh Architecture" width="800"/>
</p>

> **KidMesh: Computational Mesh Reconstruction for Pediatric Congenital Hydronephrosis Using Deep Neural Networks**  
> Haoran Sun, Zhanpeng Zhu, Anguo Zhang, Bo Liu, Zhaohua Lin, Liqin Huang, Mingjing Yang, Lei Liu, Shan Lin, and Wangbin Ding  
> *IEEE Journal of Biomedical and Health Informatics*, 2026  
> DOI: [10.1109/JBHI.2026.3679524](https://doi.org/10.1109/JBHI.2026.3679524)

---

## Abstract

Pediatric congenital hydronephrosis (CH) is a common urinary tract disorder, primarily caused by obstruction at the renal pelvis-ureter junction. Magnetic resonance urography (MRU) can visualize hydronephrosis by utilizing the natural contrast provided by water. Existing voxel-based segmentation approaches extract CH regions from MRU but require complex post-processing to convert results into mesh representations suitable for functional assessments such as urodynamic simulations.

We propose **KidMesh**, an end-to-end deep learning framework that directly reconstructs CH meshes from MRU images without requiring mesh-level annotations. KidMesh extracts hierarchical feature maps from MRU volumes, samples vertex-wise features via grid sampling, and progressively deforms a kidney-shaped template mesh into patient-specific CH geometries. The resulting meshes are watertight, topologically regularized, and CFD-ready — enabling urodynamic simulations without manual post-processing.

---

## Architecture

KidMesh consists of three main modules:

- **Feature Extraction Module (FEM):** A 3D ResUNet encoder-decoder that extracts multi-scale feature maps `{F1, ..., F5}` at resolutions 8³–128³ with 256–16 channels.
- **Feature Sampling Module (FSM):** Samples vertex-wise features from voxel feature maps via trilinear interpolation, with learned neighborhood aggregation and a parameter-free self-attention mechanism for global coherence.
- **Mesh Deformation Module (MDM):** Progressively deforms the template mesh through 5 Displacement Estimation (DE) steps in a coarse-to-fine manner, with 3 Vertex Upsampling (VU) steps to increase mesh resolution.

<p align="center">
  <img src="assets/overview.png" alt="KidMesh Overview" width="800"/>
</p>

---

## Requirements

- Python 3.8+
- CUDA 11.8
- PyTorch 2.0

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Version |
|---|---|
| torch | 2.0.0+cu118 |
| pytorch3d | 0.7.5 |
| nibabel | 5.3.2 |
| pydicom | 3.0.1 |
| scipy | 1.15.2 |
| trimesh | 4.2.2 |
| vtk | 9.3.0 |
| scikit-image | 0.16.2 |
| h5py | 3.10.0 |

Build the custom CUDA rasterization kernel:

```bash
cd utils/rasterize
python setup.py build_ext --inplace
```

---

## Dataset

The dataset consists of MRU images from 160 pediatric CH patients (pre- and post-operative scans), acquired on a 3.0T Siemens MAGNETOM Skyra scanner (T2-weighted 3D turbo spin-echo, 1.0 mm slice thickness, 256×256 matrix). The split is 112/32/16 for train/validation/test.

**Preprocessing steps:**

1. Reorient volumes to RAS convention using ITK.
2. Center-crop a ROI around the CH region using nnU-Net segmentation centroids.
3. Resample to isotropic 1 mm³ resolution.
4. Resize to (128, 128, 128) and apply Z-score normalization.

Organize your data as follows:

```
data/
└── dataset/
    └── data_final/
        ├── <case_id>/
        │   ├── image.nii.gz
        │   └── mask.nii.gz
        └── positions.json
```

Run preprocessing:

```bash
python data_preprocess.py
```

---

## Configuration

All experiment parameters are set in `config.py`:

```python
cfg.output_shape = (128, 128, 128)
cfg.pad_shape    = (128, 128, 128)
cfg.steps        = 4                  # encoder-decoder depth
cfg.first_layer_channels = 16
cfg.num_classes  = 2
cfg.graph_conv_layer_count = 4
cfg.learning_rate = 1e-4
cfg.numb_of_itrs  = 20500
cfg.batch_size    = 1
```

Set `cfg.dataset_path` and `cfg.save_path` to your local paths before running.

---

## Training

```bash
python main.py
```

Training uses the Adam optimizer (lr=1e-4) and logs metrics to [Weights & Biases](https://wandb.ai). Set the experiment ID in `main.py`:

```python
exp_id = 100
```

To resume from a checkpoint, set `cfg.trial_id` to the trial number in `config.py`. The best model is saved to:

```
data/result/Experiment_<exp_id>/trial_<trial_id>/best_performance/model.pth
```

---

## Evaluation

Evaluation runs automatically during training at every iteration (controlled by `cfg.eval_every`). To run standalone evaluation on the test set, set `cfg.trial_id` to the target trial and run:

```bash
python main.py
```

**Metrics reported:**

| Metric | Description |
|---|---|
| Dice (%) | Voxel overlap after mesh rasterization |
| Jaccard (%) | Intersection over union |
| ASSD (mm) | Average Symmetric Surface Distance |
| HD (mm) | 90th-percentile Hausdorff Distance |
| P2SD (mm) | Point-to-surface distance |

---

## Results

### Comparison with State-of-the-Art (Table III)

| Method | ASSD_outer↓ | ASSD_mesh↓ | HD_outer↓ | HD_mesh↓ | Dice(%)↑ | Jaccard(%)↑ | P2SD↓ | Infer Time |
|---|---|---|---|---|---|---|---|---|
| nnU-Net + Post | 1.60±.384 | 1.41±.640 | 29.51±1.66 | 28.16±1.53 | 88.3±3.92 | 81.9±6.63 | 1.98±1.15 | 71.2s |
| FCN + Post | 2.05±.512 | 1.86±.576 | 25.28±2.76 | 24.96±1.21 | 83.0±5.21 | 73.4±8.23 | 1.60±.576 | 61.1s |
| SCU + Post | 2.56±.646 | 2.43±.428 | 37.50±2.87 | 36.48±2.79 | 78.3±4.56 | 67.4±7.23 | 3.52±2.30 | 88.3s |
| RES50 + Post | 1.79±.448 | 1.60±.640 | 31.80±1.77 | 31.36±1.54 | 87.5±3.56 | 80.5±6.03 | 2.18±1.02 | 86.5s |
| MeshDeformNet | 2.30±.640 | 2.18±.512 | 10.42±.640 | 9.31±.576 | 84.3±3.14 | 75.3±3.95 | 1.54±.768 | 3.14s |
| **KidMesh (ours)** | **2.11±.256** | **2.04±.320** | **8.22±.576** | **8.16±.640** | **86.0±3.58** | **78.3±2.09** | **1.15±.192** | **0.36s** |

KidMesh achieves the lowest HD and P2SD among all methods, with inference time **197× faster** than nnU-Net + post-processing, and requires **no mesh-level annotations** for training.

### Configuration Study (Table I, VU=3, Mesh_NO=162)

| VU | Mesh_NO | ASSD_mesh↓ | HD_mesh↓ | Dice(%)↑ | Jaccard(%)↑ | P2SD↓ |
|---|---|---|---|---|---|---|
| 3 | 56 | 2.56±.210 | 8.80±.239 | 84.0±2.51 | 72.9±1.85 | 1.28±.213 |
| 3 | 162 | 2.04±.320 | 8.16±.640 | 86.0±3.58 | 78.3±2.09 | 1.15±.192 |
| 3 | 642 | 1.66±.123 | 8.00±.320 | 87.6±2.46 | 79.2±2.41 | 1.14±.256 |
| 2 | 162 | 2.62±.159 | 8.68±.571 | 84.6±1.56 | 73.7±1.84 | 1.21±.145 |
| **3** | **162** | **2.04±.320** | **8.16±.640** | **86.0±3.58** | **78.3±2.09** | **1.15±.192** |
| 4 | 162 | 1.53±.115 | 7.82±.331 | 88.8±1.50 | 79.5±1.81 | 1.02±.123 |

The default configuration (VU=3, Mesh_NO=162) balances performance and resource consumption (5.89 GB GPU memory, 0.36s inference).

---

## Project Structure

```
KidMesh/
├── main.py                  # Entry point: training and evaluation
├── train.py                 # Trainer class
├── evaluate.py              # Evaluator class
├── config.py                # Experiment configuration
├── data_preprocess.py       # Data preprocessing pipeline
├── data/
│   └── KD.py                # Dataset loader (KD class)
├── model/
│   └── kidmesh.py           # KidMesh network definition
├── utils/
│   ├── loss.py              # Dice, Chamfer, Edge, Normal, Laplacian, Area, Seal losses
│   ├── metrics.py           # IoU, Dice, ASSD, HD, P2SD
│   ├── utils_common.py      # Shared utilities
│   ├── utils_unet.py        # U-Net building blocks
│   ├── rasterize/           # Custom CUDA mesh rasterization kernel
│   └── utils_voxel2mesh/
│       ├── graph_conv.py    # Graph convolutional layers
│       └── feature_sampling.py  # Trilinear vertex feature sampling
└── spheres/                 # Mesh templates (icosahedron, kidney-shaped ellipsoid)
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{sun2026kidmesh,
  title     = {KidMesh: Computational Mesh Reconstruction for Pediatric Congenital Hydronephrosis Using Deep Neural Networks},
  author    = {Sun, Haoran and Zhu, Zhanpeng and Zhang, Anguo and Liu, Bo and Lin, Zhaohua and Huang, Liqin and Yang, Mingjing and Liu, Lei and Lin, Shan and Ding, Wangbin},
  journal   = {IEEE Journal of Biomedical and Health Informatics},
  year      = {2026},
  doi       = {10.1109/JBHI.2026.3679524}
}
```

---

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (62401148, 62271149), Fuzhou Science and Technology Planning Project (2023-P-001), and Fujian Science and Technology Funds (2025J01072, 2023Y9308, 2024J01353, 2022Y0056, 2022Y4014, 2020Y9091).

KidMesh builds on ideas from [Voxel2Mesh](https://github.com/cvlab-epfl/voxel2mesh) and [MeshDeformNet](https://github.com/fkong7/MeshDeformNet). We thank the authors for their open-source contributions.
