# Kaggle Setup

> This guide explains how to run the exercises on [Kaggle](https://www.kaggle.com/), entirely in your browser. **Kaggle is the go-to standard environment for this course** — nothing to install, works on any machine, and a free GPU included. If you are not sure where to start, start here.
>
> Once you are up and running, do also give the [local setup](local.md) a try when you get the chance: it uses **VS Code, which is the same environment as the exam**, so it is worth getting comfortable with.

## Table of Contents
- [Why Kaggle](#why-kaggle)
- [1. Create and Verify Your Account](#1-create-and-verify-your-account)
- [2. Get the Files](#2-get-the-files)
- [3. The General Procedure](#3-the-general-procedure)
    - [Import the Notebook](#import-the-notebook)
    - [Create the Dataset](#create-the-dataset)
    - [Attach the Dataset to the Notebook](#attach-the-dataset-to-the-notebook)
    - [Enable the GPU](#enable-the-gpu)
- [4. Per-Exercise Instructions](#4-per-exercise-instructions)
    - [Exercise 1 — Quadrotor Flight](#exercise-1--quadrotor-flight)
    - [Exercise 2 — Solar Compass](#exercise-2--solar-compass)
    - [Exercise 3 — Variational Autoencoders](#exercise-3--variational-autoencoders)
- [5. Sessions, Quotas and Saving Your Work](#5-sessions-quotas-and-saving-your-work)
- [Troubleshooting](#troubleshooting)

## Why Kaggle
Kaggle notebooks come with PyTorch, NumPy, matplotlib and friends pre-installed, and give every verified user a weekly quota of free GPU time. The trade-off is that Kaggle does not see your local files: every script and data file the notebook needs has to be uploaded as a **Kaggle Dataset** and attached to the notebook as an *input*.

That is the one idea behind this whole guide:

```
your machine                 Kaggle
------------                 ---------------------------------
ex_1/ex_1_kaggle.ipynb  -->  a Notebook
ex_1/additional/*.py    \
data/*.npz              /-->  a Dataset, attached as an input
                              and mounted read-only at
                              /kaggle/input/<dataset-slug>/
```

## 1. Create and Verify Your Account
1. Sign up at [kaggle.com](https://www.kaggle.com/) — a personal email is fine.

2. **Phone-verify your account.** Go to `Settings` → `Phone Verification` and complete it.

    > ⚠️ Without phone verification Kaggle will **not** give you GPU access or internet access inside notebooks. Exercise 3 needs internet to download MNIST, so do this before you start.

## 2. Get the Files
Even when working on Kaggle you need a local copy of the repository to upload from. Either:

- clone it with Git (see [local.md](local.md#2-git)), or
- download it as a ZIP from the GitHub page (`Code` → `Download ZIP`) and unzip it.

You also need the course data, from [this link](https://surfdrive.surf.nl/files/index.php/s/uStySKYBKHBXcjP) with password `Ae4353`. Extract it somewhere you can find it — you will be dragging these files into the browser.

## 3. The General Procedure
You repeat this once per exercise. The per-exercise file lists are in [section 4](#4-per-exercise-instructions).

### Import the Notebook
1. On the Kaggle home page, press the `+ Create` button in the top-left corner and choose **`New Notebook`**.

2. In the new notebook, go to `File` → **`Import Notebook`**.

3. Browse to your local repository and select the ***`_kaggle` variant*** of the exercise, e.g. `ex_1/ex_1_kaggle.ipynb` — **not** `ex_1/ex_1.ipynb`. The `_kaggle` notebooks are the same exercises with the file paths adapted to Kaggle.

### Create the Dataset
1. Press `+ Create` again and choose **`New Dataset`**.

2. Give it exactly the name listed for your exercise (`AE4353_0`, `AE4353_1`, `AE4353_2` or `AE4353_3`). The notebooks refer to these names, so a typo means broken paths.

3. Upload the files listed for that exercise.

    > ⚠️ **Upload the whole `additional` folder, not the individual files inside it.** Drag and drop the folder itself into the upload box so the folder structure is preserved — the notebooks import from `additional.<something>`.

4. Press `Create`.

    > 💡 Kaggle turns your title into a lowercase, hyphenated **slug**: `AE4353_1` becomes `ae4353-1`, which is why the notebook paths read `/kaggle/input/ae4353-1/`. You can see the real slug in the dataset's URL.

### Attach the Dataset to the Notebook
1. Go back to your notebook. In the panel on the right there is an **`Input`** section.

2. Press `+ Add Input`, switch to the `Your Datasets` / `Datasets` tab, find the dataset you just created and add it.

3. It now appears under `/kaggle/input/<slug>/` and is visible in the file browser on the right. Expand it and confirm the paths match what the notebook expects.

### Enable the GPU
In the right-hand panel, open `Session options` (or the `⋮` menu) → **`Accelerator`** and pick a **GPU** (e.g. `GPU T4 x2`). Turn `Internet` **on** in the same panel if the exercise needs it.

Then tell the notebook to actually use it — set `cuda = True` / `DEVICE = torch.device("cuda")` where the notebook defines the device, and confirm with:

```python
import torch
print(torch.cuda.is_available())  # should print True
```

> 💡 Only switch the accelerator on when you are about to train. GPU time is billed against your weekly quota even while the session sits idle.

## 4. Per-Exercise Instructions

Summary of what goes where:

| Exercise | Notebook to import | Dataset name | Mounted at | Internet |
| --- | --- | --- | --- | --- |
| 0 | `ex_0/ex_0_kaggle.ipynb` | `AE4353_0` | `/kaggle/input/ae4353-0/` | not needed |
| 1 | `ex_1/ex_1_kaggle.ipynb` | `AE4353_1` | `/kaggle/input/ae4353-1/` | not needed |
| 2 | `ex_2/ex_2 kaggle.ipynb` | `AE4353_2` | `/kaggle/input/ae4353-2/` | not needed |
| 3 | `ex_3/ex_3_kaggle.ipynb` | `AE4353_3` | `/kaggle/input/ae4353-3/` | **required** (MNIST download) |

### Exercise 0 — Python and ML Warm-Up
**Dataset `AE4353_0`** — the `additional` folder from `ex_0/` only, containing:

- `data.py`
- `plots.py`

There is no data file to upload: Exercise 0 generates its own data. No GPU and no internet are needed.

### Exercise 1 — Quadrotor Flight
**Dataset `AE4353_1`** — 6 files in total:

1. The two data files from the SURFdrive download:
    - `2D_QUAD_HOVER.npz`
    - `3D_QUAD_HOVER.npz`
2. The `additional` folder from `ex_1/`, containing:
    - `dataloader.py`
    - `plot_utils.py`
    - `system_dynamics.py`
    - `trajectory_simulation.py`

**In the notebook**, check these two things:

- The dataset path in the configuration cell points at your input, and matches the `trajectory_dim` you are working on:
    ```python
    trajectory_dim = 2
    dataset_file = "/kaggle/input/ae4353-1/2D_QUAD_HOVER.npz"   # 3D_QUAD_HOVER.npz for trajectory_dim = 3
    ```
- The cell that copies the helper scripts into the writable working directory uses the **absolute, slugified** path:
    ```python
    shutil.copytree('/kaggle/input/ae4353-1/additional', '/kaggle/working/additional', dirs_exist_ok=True)
    ```
    See [Troubleshooting](#troubleshooting) — the shipped notebook may have this line written with the un-slugified name or without the leading `/`.

### Exercise 2 — Solar Compass
**Dataset `AE4353_2`**:

1. The polarization data from the SURFdrive download — the `polarization_dataset` folder containing `dataset.h5`.
2. `ex_2/dataset.py` (the notebook does `import dataset`, so this file must sit next to the notebook at runtime).
3. The `additional` folder from `ex_2/`, containing `plots.py`.

**In the notebook**:

- Copy both the helper folder and `dataset.py` into the working directory before importing them:
    ```python
    import shutil
    shutil.copytree('/kaggle/input/ae4353-2/additional', '/kaggle/working/additional', dirs_exist_ok=True)
    shutil.copy('/kaggle/input/ae4353-2/dataset.py', '/kaggle/working/dataset.py')
    ```
- Define the dataset path before the dataset is constructed (the notebook has it commented out with a local path):
    ```python
    dataset_path = "/kaggle/input/ae4353-2/polarization_dataset"
    ```
    `PolImgDataset` is called with `h5=True`, so it looks for `dataset.h5` inside that folder.
- This exercise trains a CNN on images — [enable the GPU](#enable-the-gpu).

### Exercise 3 — Variational Autoencoders
**Dataset `AE4353_3`**:

1. The `additional` folder from `ex_3/`, containing `plots.py`.

That is all — there is no SURFdrive data for this exercise. MNIST is downloaded by `torchvision` at runtime, which is why **`Internet` must be enabled** in the session options.

**In the notebook**:

- Copy the helper folder:
    ```python
    shutil.copytree('/kaggle/input/ae4353-3/additional', '/kaggle/working/additional', dirs_exist_ok=True)
    ```
- Point the MNIST download at a **writable** location. `/kaggle/input/` is read-only, and the default `'~/datasets'` will not survive the session:
    ```python
    dataset_path = '/kaggle/working/datasets'
    ```

## 5. Sessions, Quotas and Saving Your Work
- **Sessions are temporary.** A session shuts down after ~20 minutes of inactivity, and there is a hard runtime limit (currently 12 hours CPU / 9 hours GPU per session). Everything written to `/kaggle/working/` is lost when the session ends unless you save.
- **GPU quota is weekly** (around 30 hours) and resets every week. Prototype on CPU with a small number of epochs, then switch to GPU for the real training run.
- **Save your notebook** with `Save Version` (top right):
    - *Quick Save* stores the notebook as it is on screen, without re-running it.
    - *Save & Run All* re-executes everything from a clean session — a good final check that your notebook actually runs top to bottom.
- **Back up your work locally.** `File` → `Download Notebook` gives you the `.ipynb`, which you can keep alongside the repository. Do not rely on Kaggle as your only copy.
- **Checkpoint long trainings** by saving model weights into `/kaggle/working/` and downloading them from the `Output` section of the right-hand panel before the session ends.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `FileNotFoundError: kaggle/input/...` | The path is missing its leading `/`. Use the absolute path `/kaggle/input/...`. |
| `FileNotFoundError: /kaggle/input/AE4353_1/...` | Use the **slug**, not the title: `/kaggle/input/ae4353-1/`. Check the exact slug in the file browser on the right. |
| `FileExistsError: /kaggle/working/additional` | You re-ran the `copytree` cell. Add `dirs_exist_ok=True`, or run `shutil.rmtree('/kaggle/working/additional')` first. |
| `ModuleNotFoundError: No module named 'additional'` | The `copytree` cell has not run, or you uploaded the loose `.py` files instead of the `additional` folder. Re-upload the folder itself. |
| `ModuleNotFoundError: No module named 'dataset'` (Ex 2) | `dataset.py` was not copied into `/kaggle/working/`. See [Exercise 2](#exercise-2--solar-compass). |
| `NameError: name 'dataset_path' is not defined` (Ex 2) | Define it as shown in [Exercise 2](#exercise-2--solar-compass). |
| `OSError: [Errno 30] Read-only file system` | You are writing into `/kaggle/input/`. Write to `/kaggle/working/` instead. |
| MNIST download hangs or fails (Ex 3) | `Internet` is off, or your account is not phone-verified. See [section 1](#1-create-and-verify-your-account). |
| `torch.cuda.is_available()` is `False` | No accelerator selected for the session, or the notebook's device flag is still set to CPU. See [Enable the GPU](#enable-the-gpu). |
| Session died mid-training | Idle timeout or quota. Reduce epochs, save checkpoints to `/kaggle/working/`, and keep the browser tab active while training. |

Still stuck? Come to an exercise session or reach out — see the [contact information](README.md#contact-information). We are happy to help. 🙂
