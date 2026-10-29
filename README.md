# LiDAR mobility-aid classification and person re-identification

Code for two studies on a Velodyne Alpha Prime VLS-128 dataset collected at the CUBIC Lab, Carleton University:

1. **Mobility-aid classification** from a single LiDAR frame (Point Transformer V3, built on Pointcept).
2. **Person re-identification** from LiDAR (ReID3D), preceded by a preprocessing pipeline that isolates one person per frame.

Author: Sean Kirkby. Supervisor: Prof. James R. Green.
<!-- verify: authorship line, license, and how to cite the thesis and the MeMeA 2025 paper -->

This repository contains **code only**. The participant data, trained weights and training outputs are not included and must not be added (see `.gitignore`).

## Pipeline

```
CSV frames --> improved_processing.py --> one person cluster per frame (.bin) --> ReID3D --> embeddings, top-k / mAP
     |
     +-----> classification.py (PTv3, one whole frame in, mobility-aid class out)
```

The classification model takes the whole region-of-interest frame as input. The ReID model takes the person cluster produced by the preprocessing step, resampled to 256 points by farthest point sampling inside the dataloader.

## Repository layout

| Folder | Contents |
|---|---|
| `preprocessing/` | `improved_processing.py` (CSV to person clusters in LReID format), `cv_run_fullset.sh` (5 seeds, frames 200 to the end) |
| `classification/` | `classification.py`, `custom_models.py`, `count_pointclouds.py`, `patches/hilbert.py`, `results/` (confusion matrices) |
| `ReID3D/` | `reidnet/` (training, tuning, evaluation scripts), `util/utils.py`, `database/create_reid_database.py`, `run.sh`, `requirements*`, `results/` |
| `experiments/classification/` | `multiframe_classification.py`, `recognition.py`, `VideoRecognition.py`. Exploratory scripts. <!-- verify with Sean: whether these were used for any reported result --> |

## Data format (not included)

- One CSV per frame, with a header: `intensity, laser_id, azimuth, distance_m, adjustedtime, timestamp, vertical_angle, Points_m_XYZ:0, Points_m_XYZ:1, Points_m_XYZ:2`. Exports made directly from VeloView may carry three extra columns (`vtkOriginalIndices`, `__vtkIsSelected__`, `vtkCompositeIndexArray`); the code reads columns by name, so they are ignored. `azimuth` is stored in hundredths of a degree.
- Layout: `<root>/{train,val,test}/<participant id>/<class id>/<recording>_(Frame N).csv`.
- Frames were clipped to a region of interest at recording time, so they still contain walls, floor and a pillar. The thesis describes the region as about 12 m ahead, +-45 degrees and 2.5 m high. Measured on sample frames, the clip is closer to: distance up to 15 m, vertical angle from -25 to 5.73 degrees (119 of the 128 lasers), an azimuth sector about 63 degrees wide (roughly +-30 degrees), and about 24,000 points per frame. Axes: y forward, x lateral, z up, with the floor near z = -1 m and the clip at z = 1.5 m.
- 30 participants, 7 mobility-aid classes, 8 travel directions, fixed sensor pose, empty indoor room. Participants walk back and forth along four lines at 0, 45, 90 and 135 degrees to the sensor.
- In the eight-direction experiments, frames from 200 to the end of each recording are used (the participant stands still before that). Sample data with small frame numbers therefore holds standing frames only.

Class ids:

| Id | Class | Id | Class |
|---|---|---|---|
| 0 | Walking | 4 | Rollator walking |
| 1 | Crutches | 5 | Rollator shuffling |
| 2 | Cane | 6 | Wheelchair |
| 3 | Walker | | |

The ReID scripts loop over modes named `walking wheelchair rollatorshuffling rollatorwalking walker cane crutches`; they map to class ids 0, 6, 5, 4, 3, 2, 1 in that order.

## Recording and export (VeloView)

Frames come from recorded `.pcap` files opened in VeloView. To export one frame, open the Spreadsheet view and use Export spreadsheet with "Show only selected points" switched off. (Exporting a "Select points" selection gives a partial frame with different column names.)

The original crop settings are not documented in the thesis. The settings below were found by trial and reproduce the measured properties of a dataset frame. They are set in the Crop Returns dialog (Tools menu):

| Setting | Value |
|---|---|
| Mode | spherical |
| Inverse Cropping | off |
| Distance | 0 to 15 |
| Vertical angle | -25 to 5.8 (the highest kept laser is at 5.73 degrees; a larger limit keeps extra rings) |
| Rotational angle | a range about 63 degrees wide, centred on the walking track |

The dialog's rotational angle is offset from the `azimuth` column of the CSV, so check the exported sector in that column and adjust. The walking track is up to about +-3 m wide, which a +-30 degree sector covers from about 6 m range outward.

Comparison of one dataset frame with one frame exported with these settings:

| Property | Dataset frame | Reproduced frame |
|---|---|---|
| Points | 23,871 | 24,087 |
| Lasers (rings) | 119 | 119 |
| Vertical angle, max | 5.73 | 5.73 |
| Distance | 2.27 to 15.00 | 2.24 to 15.00 |
| z range | -1.01 to 1.50 | -1.02 to 1.50 |
| Azimuth sector (degrees) | about 327 to 30.5 | about 322 to 23 |

<!-- verify with Sean: exact VeloView crop values and export menu he used. A trial with rotational angle -104 to 290, vertical -25 to 7 and distance 0 to 16 gave 122 rings and a 15.6 m maximum, so those are not his values -->

## Environments

Three conda environments exist on the original server: `base`, `pointcept` and `reid3d`. The package lists below come from the `import` statements in this repository. A version is given only where it is written down in the repository or was tested; everything else is left open and should be replaced by an exported environment file (`conda env export -n <name> --from-history`, or `conda list -n <name> --export`).

### `base` (Anaconda default)

Created by the Anaconda installer (Anaconda3-2024.10-1). Nothing in this repository has to be installed here; it only provides the `conda` command. Do not install project packages into it. <!-- verify: which environment runs preprocessing on the original server -->

### `pointcept` (classification)

Used by `classification/classification.py`, `custom_models.py`, `count_pointclouds.py` and the files in `experiments/classification/`.

| Package | Why |
|---|---|
| Python, PyTorch 2.0 or newer, CUDA build | `torch._dynamo` is imported (exists from PyTorch 2.0). Training used a CUDA GPU (A100 40 GB). <!-- verify: exact versions --> |
| numpy, pandas | CSV loading and data handling |
| scikit-learn | `confusion_matrix`, metrics |
| matplotlib | confusion matrix figure |
| Pointcept (clone, not pip) | `pointcept.models.utils.structure.Point` and `PointTransformerV3` |
| Pointcept's own requirements | spconv, torch-scatter, timm, einops, addict and others. Follow the install section of the Pointcept README for the version you clone. <!-- verify: list from the pointcept environment --> |
| flash-attn | The model config sets `enable_flash: True`. Set it to `False` if flash-attn cannot be installed. <!-- verify: effect on results --> |
| CUDA toolkit 11.8 | An installer for it sits next to the code on the server; needed to build CUDA extensions. <!-- verify --> |

Pretrained weights: the PTv3 checkpoint trained on nuScenes, saved as `model_best.pth`.

### `reid3d` (person re-identification)

Used by everything in `ReID3D/` (training, tuning, evaluation, database lists). Python 3.9 or 3.10 (cache files from both versions exist on the server). <!-- verify -->

| Package | Why |
|---|---|
| numpy 1.26.0 | `requirements.txt` |
| torch 1.12.1, torchvision 0.13.1 | `requirements.txt`; `reidnet/main.py`, `util/utils.py` |
| pytorch3d 0.7.4 | `requirements.txt` (upstream). Must match the torch and CUDA versions. |
| scipy 1.11.1 | `requirements.txt`; `database/create_reid_database.py` imports `scipy.io` |
| scikit-learn 1.4.2 | `requirements.txt` |
| tqdm 4.66.1 | `requirements.txt`; progress bars |
| pandas | imported in `util/utils.py`, **not listed in `requirements.txt`** |
| plyfile | imported in `util/utils.py` (`.ply` reader), **not listed in `requirements.txt`** |
| Pillow | imported in `util/utils.py`; normally installed together with torchvision |

`requirements.local.txt` lists only numpy, scikit-learn, scipy and tqdm. <!-- verify: what it is for --> The imports of the upstream `model/` package were not checked. <!-- verify -->

### Preprocessing (`preprocessing/improved_processing.py`)

Needs numpy, scikit-learn (DBSCAN) and open3d. pandas is optional (the script falls back to the `csv` module). **open3d is optional in the code, but without it ground removal, wall removal and outlier removal are skipped silently**, so the output is worse without any error message.

Tested setup (a CPU-only machine without internet): Python 3.12.7, numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1, scipy 1.13.1 and open3d 0.19.0, in a clone of an Anaconda base environment. The wheels were downloaded on another machine for Linux x86_64 and Python 3.12 (`pip download open3d --only-binary=:all: --platform manylinux_2_31_x86_64 --python-version 3.12 --implementation cp --abi cp312`) and installed with `pip install --no-index --find-links <folder> open3d`.

<!-- verify: which environment ran preprocessing for the thesis results -->

## Usage

All scripts use absolute paths of the original server (`/data/<user>/...`). Change them to your own paths before running.

### 1. Preprocessing (person clusters for ReID)

`preprocessing/cv_run_fullset.sh` calls `improved_processing.py` for seeds 0 to 4. Main parameters: `--include-classes` (one class per run), `--force-split train_test --train-persons 28 --test-persons 2`, `--dbscan-eps 0.22 --dbscan-min-samples 40`, `--person-h 0.8 2.0`, `--person-w/--person-d 0.30 1.5`, `--use-scoring`, `--max-person-per-frame 1`, `--frame-start 200`. The script sets `CLASS=0` and an output name containing `walking`; edit both for another mode. The step reads XYZ only and writes XYZ only (intensity is dropped). It uses Open3D RANSAC for ground and wall removal and scikit-learn DBSCAN.

Output goes to `<out-root>/reid3d/` in LReID format (`bbox_train`, `bbox_test`, `info`) plus `manifest.json`. The messages about `debug_steps` are printed even though that folder was not written in a test run.

To try it on a small sample whose frame numbers start near 0, drop `--frame-start 200` (it would skip every frame) and `--force-split` (it needs at least train plus test persons). A test run of this kind on 100 frames from four participants produced one person cluster for 95 of the frames.

`python improved_processing.py --help` fails on Python 3.12 because the help text of `--dims-gate-slack` contains a bare percent sign. Normal runs are not affected.

### 2. Classification

1. Clone Pointcept and copy `classification.py`, `custom_models.py` (and `count_pointclouds.py`) into its root.
2. Replace `pointcept/models/utils/serialization/hilbert.py` with `classification/patches/hilbert.py`. <!-- verify: what Sean changed, diff against upstream -->
3. Put the PTv3 weights pretrained on nuScenes at `model_best.pth` in the working directory.
4. Set the `train`, `val`, `test` roots in `main()` and run `python classification.py`.

Per the thesis, the model is trained with a linear-probing phase followed by full fine-tuning. In the current `classification.py` the linear-probing phase is commented out: the script loads `best_model_phase_Phase1.pth` if that file exists and then fine-tunes the whole model. It reports per-person accuracy, accuracy per direction segment, and a confusion matrix.

### 3. Person re-identification

1. Clone ReID3D (`https://github.com/GWxuan/ReID3D`). The `model/` folder comes from upstream and is not copied here. Overlay the files of this repository.
2. **Stage 1:** train the base model on the public LReID dataset with `reidnet/train.sh`. The upstream checkpoints are not released, which is why this step is needed. The result is expected at `reidnet/checkpoints/PRETRAINED_ARCHIVE_ckpt_best.pth`. `train.sh` uses two GPUs (`--gpu_id '2,3'`) and resumes from `~/ReID3D/reidnet/log/ckpt_last.pth`; change the GPU ids and remove the `--load_ckpt` option for a fresh run.
3. Build the database lists: `bash run.sh` (calls `database/create_reid_database.py` for seeds 0 to 4).
4. **Fine-tune:** `reidnet/cv_full_train.sh` (7 modes x 5 seeds, loads the base checkpoint, AdamW, lr 5e-4).
5. **Evaluate:** `reidnet/cv_test_multiquery.sh` (multi-query top-1, top-3, mAP).
6. **Summarise:** `python multiquery_results_extraction.py` writes the per-seed table and mean +/- std.

Other scripts in `reidnet/`: `tune_rrs.py` and `tune.sh` (hyperparameter search on a subset, frames 400 to 650), `full_train.py` and `full_train.sh` (two-direction scenario, frames 400 to 650), `test_multiquery.sh` (single-mode evaluation).

Optional notification hooks at the end of some scripts run only if `notification.py` or `training_notify.py` exists next to the script. Those files are not part of this repository.

## Results

`classification/results/` holds the confusion matrices and `ReID3D/results/` holds the multi-query summaries (top-1, top-3, mAP per mode and seed). <!-- verify: which result file matches which table in the thesis -->

## Known limitations

- Hyperparameters of the preprocessing steps need re-tuning for each room.
- The preprocessing keeps one cluster per frame, so it does not handle several people in the same frame.
- Data come from one empty room with a fixed sensor pose and a young, mostly male group of participants.

## Provenance

This repository was assembled from a copy of the original code on the lab server. The original files were not changed. Differences from the original:

- `notification.py` and `training_notify.py` are left out because they held notification keys. The shell scripts that called them (`preprocessing/cv_run_fullset.sh`, `ReID3D/reidnet/cv_full_train.sh`) now run them only if the file exists.
- Files were moved into the folders above. Older or unused files (`ARCHIVE_classification.py`, `ARCHIVE_main.py`, `datasets.py`, `util/loss.py`, `run_fullset.sh` and the upstream ReID3D README) were left out; they still exist on the server.
- The contents of all other files are unchanged. Many Python files use Windows line endings.
- The README was extended in October 2026 with the environment notes, the recording and export section, and the preprocessing test run.

## Credits

- [Pointcept](https://github.com/Pointcept/Pointcept) and Point Transformer V3.
- [ReID3D / LReID](https://github.com/GWxuan/ReID3D), Guo et al., "LiDAR-based Person Re-identification".
- nuScenes, for the pretrained PTv3 weights.

Each upstream project keeps its own license.
<!-- verify: license of each upstream project before this repository is made public -->
