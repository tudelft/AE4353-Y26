# Local Setup (VS Code + Miniconda)

> This guide takes you from a clean machine to running `ex_1/ex_1.ipynb` locally. If you would rather work in the browser without installing anything, see [kaggle.md](kaggle.md) instead.

## Table of Contents
- [Overview](#overview)
- [1. Visual Studio Code](#1-visual-studio-code)
- [2. Git](#2-git)
    - [Windows](#windows)
    - [Linux](#linux)
    - [macOS](#macos)
- [3. Miniconda and the Course Environment](#3-miniconda-and-the-course-environment)
- [4. Data](#4-data)
- [5. Running the Exercises](#5-running-the-exercises)
- [6. GPU Acceleration (Optional)](#6-gpu-acceleration-optional)
    - [Apple Silicon (MPS)](#apple-silicon-mps)
- [Troubleshooting](#troubleshooting)

## Overview
The local setup consists of four pieces:

| Piece | Why you need it |
| --- | --- |
| Visual Studio Code | The editor you write and run the notebooks in |
| Git | To clone the course repository and pull our updates |
| Miniconda + `env.yml` | Creates the `AE4353` environment with all pinned dependencies |
| The dataset | Downloaded separately and placed in `data/` |

You only do steps 1–3 once. Step 4 is repeated whenever a new exercise needs new data.

## 1. Visual Studio Code
[Visual Studio Code](https://code.visualstudio.com/) (VS Code) is a lightweight and powerful code editor. We use it as the interface for the whole course.

- [Install](https://code.visualstudio.com/download)
- [Documentation](https://code.visualstudio.com/docs)

After installing, also install the **Python** and **Jupyter** extensions from the Extensions panel (`Ctrl + Shift + X`, or `Cmd + Shift + X` on macOS). They are what let VS Code run notebooks.

## 2. Git

### Windows
1. Install Git for Windows from the [installation link](https://git-scm.com/downloads), selecting the Windows distribution. Run the downloaded installer and accept the recommended installation location and permissions.

2. Once the installation wizard finishes, open **Git Bash** by pressing the `Windows key` and typing `Git Bash`.

3. Inside the Git Bash terminal, clone the repository (this creates a local copy):
    ```bash
    git clone https://github.com/tudelft/AE4353-Y26.git
    ```

4. You now have a local version of the repository that can be updated with one command whenever we upload changes. Navigate into the repository folder and pull:
    ```bash
    cd AE4353-Y26/
    git pull
    ```

### Linux
Installations on Linux are done directly through the terminal. Open it from the applications page or with `Ctrl + Alt + T`.

1. Install Git (see the [official installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) if you hit problems):
    ```bash
    sudo apt-get install git-all
    ```

2. Clone the repository:
    ```bash
    git clone https://github.com/tudelft/AE4353-Y26.git
    ```

3. To pick up any future updates we make, navigate into the repository and pull:
    ```bash
    cd AE4353-Y26/
    git pull
    ```

### macOS
Installations on macOS are done through the Terminal. Open it from Launchpad, or by pressing `Cmd + Space` and typing `Terminal`.

> ⚠️ First check which chip your Mac has: click the Apple logo in the top-left corner and select `About This Mac`. If it mentions `Apple M1`/`M2`/`M3`/`M4` you have an ***Apple Silicon*** Mac; if it mentions `Intel` you have an ***Intel*** Mac. You will need this in step 3.

1. Install Git. macOS does not ship with Git, but it offers to install it for you. Run:
    ```bash
    git --version
    ```

    If a pop-up appears asking to install the `command line developer tools`, click `Install` and let it finish. It is a sizeable download — roughly 1 GB, taking about 2 GB once installed — and takes 5-10 minutes on a decent connection. This is Apple's Xcode Command Line Tools package; besides Git it provides the compilers that some Python packages need to build. If you instead see a version number, Git is already installed and you can move on.

    > 💡 If you already use [Homebrew](https://brew.sh/), `brew install git` works just as well. You do not need Homebrew for this course otherwise — everything else below uses graphical installers.

2. Clone the repository:
    ```bash
    git clone https://github.com/tudelft/AE4353-Y26.git
    ```

3. To pick up any future updates we make, navigate into the repository and pull:
    ```bash
    cd AE4353-Y26/
    git pull
    ```

> 💡 `git pull` will refuse to run if you have edited a file we also changed. Commit your own work first (`git add -A && git commit -m "my work"`), then pull.

## 3. Miniconda and the Course Environment
Conda downloads all packages required for the exercises in a controlled, reproducible way.

1. **Install Miniconda.**
    - **Windows:** use the [download page](https://www.anaconda.com/download). Press the green `Get Started` button; it will ask you to create an account, so use a ***valid email*** as you need to verify it. When offered the choice between the `Distribution` and `Miniconda`, choose **`Miniconda`**.
    - **Linux:** follow the [Linux installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2).
    - **macOS:** follow the [macOS installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install#macos). Download the installer matching your chip — `Apple Silicon` (arm64) or `Intel` (x86_64). Picking the wrong one causes confusing package errors later.

2. **Open a conda-enabled terminal.**
    - **Windows:** open `Anaconda Prompt` (search for it the same way you found Git Bash).
    - **Linux:** your normal terminal now has the `conda` command available.
    - **macOS:** close and re-open the Terminal so the installer's changes take effect, then check that `conda --version` works.

3. **Create the environment.** Navigate to the repository folder and run:
    ```bash
    cd AE4353-Y26/
    conda env create -f env.yml
    ```

    This takes a few minutes.

4. **Verify.** Run:
    ```bash
    conda env list
    ```

    If `AE4353` shows up in the list, the environment installed correctly and you are ready to go.

> 💡 If we update `env.yml` during the course, refresh your environment with
> `conda env update -f env.yml --prune`.

## 4. Data
The data for this course can be found at this [link](https://surfdrive.surf.nl/files/index.php/s/uStySKYBKHBXcjP), using the password `Ae4353`.

Download it, then extract or unzip it and place the resulting `AE4353-Datasets-2026` folder inside the repository's `data/` directory at `/your/workspace/path/AE4353-Y26/data/`. You should end up with something like:

```
AE4353-Y26/
└── data/
    └── AE4353-Datasets-2026/
        ├── 2D_QUAD_HOVER.npz          # Exercise 1
        ├── 3D_QUAD_HOVER.npz          # Exercise 1
        └── polarization_dataset/
            └── dataset.h5             # Exercise 2
```

Exercise 3 uses MNIST, which is downloaded automatically by `torchvision` the first time you run the notebook — nothing to place by hand.

> ⚠️ Please do not commit the dataset — the repository ships only an empty `data/` folder.

## 5. Running the Exercises
> 💡 This section is also covered in the `Practical session 1` slides, with a screenshot of each step.

1. Open VS Code. From the welcome screen select `Open Folder...` and choose the `AE4353-Y26` repository you just cloned.

2. Open the first notebook, `ex_1/ex_1.ipynb`. On the top right you will see a `Select Kernel` option. Click it, choose `Python Environments...`, and select the **`AE4353`** environment.

3. Point the notebook at your data by editing the dataset path near the top of the notebook so it matches where you put the files in step 4.

4. You are ready to get coding! Run the cells with `Shift + Enter`.

Each exercise folder also contains a `*_kaggle.ipynb` variant. **Ignore those when working locally** — they are the same exercise adapted for the Kaggle paths and are covered in [kaggle.md](kaggle.md).

## 6. GPU Acceleration (Optional)
Everything in this course runs on CPU, just more slowly. If you have an **NVIDIA** GPU you can enable CUDA to speed up training.

1. **Check that your GPU is visible.** In your terminal run:
    ```bash
    nvidia-smi
    ```

    You should see a table of GPU statistics. If the command is not found, install the NVIDIA driver for your GPU first.

    > ⚠️ **Windows users:** the smoothest route is WSL2 with an Ubuntu distribution. Verify `nvidia-smi` works inside the **Ubuntu** terminal, then do the rest of these steps there.

2. **Enable the CUDA build of PyTorch.** In `env.yml`, uncomment the `nvidia` channel and the `pytorch-cuda` dependency:

    ```yaml
    channels:
      - pytorch
      - nvidia          # <- uncomment
      - conda-forge
    dependencies:
      ...
      - pytorch-cuda=12.4   # <- uncomment
    ```

3. **Rebuild the environment** so conda resolves the CUDA-enabled packages:
    ```bash
    conda env remove -n AE4353
    conda env create -f env.yml
    ```

4. **Verify that PyTorch sees the GPU.** With the `AE4353` environment active, run:
    ```python
    import torch
    print(torch.cuda.is_available())  # should print True
    ```

    If it prints `True`, PyTorch is correctly set up to use CUDA. If it prints `False`, CUDA support is not properly configured — re-check steps 1–3.

5. In the notebooks, set the device flag (e.g. `cuda = True` / `DEVICE = torch.device("cuda")`) so the training loop actually uses the GPU.

### Apple Silicon (MPS)
Macs have no NVIDIA GPU, so the CUDA steps above do not apply. On **Apple Silicon**, PyTorch can use the built-in GPU through the **MPS** backend instead — this needs no changes to `env.yml`. With the `AE4353` environment active, check:

```python
import torch
print(torch.backends.mps.is_available())  # should print True
```

If it prints `True`, set the notebook's device to `torch.device("mps")` wherever the instructions mention `cuda`. It is clearly faster than CPU for Exercises 2 and 3, though still slower than a Kaggle GPU.

On **Intel** Macs there is no GPU option — run on CPU, or use [Kaggle](kaggle.md).

> 💡 No NVIDIA GPU? Don't fight it — use [Kaggle](kaggle.md), which gives you a free GPU in the browser.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `conda: command not found` | You are not in a conda-enabled terminal. Use `Anaconda Prompt` on Windows, or re-open your terminal after installing Miniconda on Linux/macOS. |
| `AE4353` does not appear in VS Code's kernel list | Reload the VS Code window (`Ctrl + Shift + P`, or `Cmd + Shift + P` on macOS → `Developer: Reload Window`), and make sure the Python and Jupyter extensions are installed. |
| **macOS:** `PackagesNotFoundError` for `pytorch=2.5.1` | You likely have strict channel priority set. The `pytorch` channel's last Intel-Mac build is 2.2.2; the pinned 2.5.1 comes from `conda-forge`. Run `conda config --set channel_priority flexible` and retry. |
| `ModuleNotFoundError` for a course package | You selected the wrong kernel. Check the top-right kernel name says `AE4353`. |
| `FileNotFoundError` on the dataset | The dataset path in the notebook does not match where you extracted the data. See [Data](#4-data). |
| `conda env create` fails to solve | Delete any partial environment (`conda env remove -n AE4353`) and retry. If it persists, contact us — see the [contact information](README.md#contact-information). |
| Everything is very slow | Expected on CPU. Reduce the number of epochs while developing, then either enable [GPU acceleration](#6-gpu-acceleration-optional) or move to [Kaggle](kaggle.md) for the full runs. |
