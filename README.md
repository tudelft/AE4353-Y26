# [AE4353] Artificial Intelligence for Aerospace Control and Operations
> Welcome to the repository of the 2026/2027 [AE4353] Artificial Intelligence for Aerospace Control and Operations course! 🚀

## Table of Contents
- [About](#about)
- [Exercises](#exercises)
- [Environments](#environments)
    - [Which Should I Pick?](#which-should-i-pick)
    - [Getting Started](#getting-started)
- [Data](#data)
- [GitHub Copilot](#github-copilot)
- [Repository Layout](#repository-layout)
- [License](#license)
- [Contact Information](#contact-information)

## About
This repository contains the course materials for the exercises and the separate competition associated with the course. The exercise notebooks are kept in the repository, and the competition materials will be added separately as needed.

The exercises walk you through applying deep learning to aerospace control and perception problems: you will train a network to fly a quadrotor, build a solar compass from polarized images, and generate data with a variational autoencoder. Each one is a Jupyter notebook with sections marked `TODO` for you to complete.

If you have any questions or need assistance, feel free to reach out — see [contact information](#contact-information). Happy coding and learning! 🌟

## Exercises

| # | Notebook | Topic | Data |
| --- | --- | --- | --- |
| 1 | [ex_1/ex_1.ipynb](ex_1/ex_1.ipynb) | **Quadrotor Flight with Deep Learning** — learn a control policy from simulated hover trajectories and evaluate it closed-loop | `2D_QUAD_HOVER.npz`, `3D_QUAD_HOVER.npz` |
| 2 | [ex_2/ex_2.ipynb](ex_2/ex_2.ipynb) | **Solar Compass from Polarization Images** — CNN regression of heading angle, with representation selection and data augmentation | `polarization_dataset/dataset.h5` |
| 3 | [ex_3/ex_3.ipynb](ex_3/ex_3.ipynb) | **Variational Autoencoders** — build and train a VAE, and explore its latent space | MNIST (downloaded automatically) |

Each exercise folder also contains a `*_kaggle.ipynb` variant. It is the **same exercise**, with file paths adapted to the Kaggle environment. Work through whichever one matches the environment you chose below — not both.

Supporting code lives in each exercise's `additional/` folder (plotting helpers, dataloaders, simulation code). We keep it out of the notebooks to keep them readable; you generally do not need to edit it, but you are welcome to read it.

## Environments
We officially support **two** environments this year. Both cover every exercise, and you can switch between them at any point.

### Which Should I Pick?

| | **Local** ([local.md](local.md)) | **Kaggle** ([kaggle.md](kaggle.md)) |
| --- | --- | --- |
| Runs on | Your own machine | Kaggle's servers, in your browser |
| Interface | VS Code | Kaggle notebook editor |
| Setup effort | One-time install of VS Code, Git and Miniconda | A Kaggle account and a per-exercise file upload |
| Dependencies | Pinned by [env.yml](env.yml) — identical for everyone | Kaggle's pre-installed stack |
| GPU | Only if you own an NVIDIA GPU | Free GPU quota (~30 h/week) |
| Files | Just clone the repo | Each exercise's files must be uploaded as a Kaggle Dataset |
| Works offline | Yes | No |

**Our recommendation:** start with the **local** setup. It is the more comfortable way to work, it uses the exact package versions we pin, and your files stay where you put them. Move to **Kaggle** if the local installation gives you trouble, or when you want a GPU for the heavier training runs in Exercises 2 and 3.

> ⚠️ These two are what we support. You are of course free to use something else (Google Colab, your own cluster, a bare `pip` install), but we cannot help you debug it.

### Getting Started
- 💻 **[local.md](local.md)** — VS Code + Git + Miniconda, on Windows, Linux or macOS, including optional GPU setup.
- ☁️ **[kaggle.md](kaggle.md)** — Kaggle account, importing the notebooks, uploading datasets, and enabling the GPU.

## Data
The data for this course can be found at this [link](https://surfdrive.surf.nl/files/index.php/s/uStySKYBKHBXcjP), using the password `Ae4353`.

Once downloaded, extract or unzip the folder. Where the files go depends on your environment:

- **Local:** place them inside this repository's `data/` directory, at `/your/workspace/path/AE4353-Y26/data/`. See [local.md](local.md#4-data).
- **Kaggle:** upload them as a Kaggle Dataset and attach it to your notebook. See [kaggle.md](kaggle.md#create-the-dataset).

> ⚠️ Please do not commit the dataset back to the repository.

## GitHub Copilot
GitHub Copilot is an AI-powered assistant that helps you write code faster and more efficiently. It provides intelligent code suggestions and completions based on your context, enhancing your coding experience and boosting productivity.

> ⚠️ We encourage you to use tools like this to aid in learning concepts and practicing coding. However, it's important not to rely solely on these tools — ensure you put in the effort to understand and practice the material yourself! Please note that such tools will ***NOT*** be permitted during the final exam.

If you do not have it yet, please sign up for the Student Developer Pack on GitHub using this [link](https://education.github.com/pack). Once you have signed up, wait for GitHub to authenticate your request. Once authenticated, you will have access to GitHub Copilot, which you can enable in VS Code by installing the GitHub Copilot extension and signing in with your GitHub account.

## Repository Layout

```
AE4353-Y26/
├── README.md          # you are here — course and environment overview
├── local.md           # setup guide: VS Code + Miniconda on your machine
├── kaggle.md          # setup guide: running the exercises on Kaggle
├── env.yml            # conda environment definition (local setup)
├── data/              # place the downloaded dataset here (not committed)
├── ex_1/              # Exercise 1 — quadrotor flight
├── ex_2/              # Exercise 2 — solar compass
└── ex_3/              # Exercise 3 — variational autoencoders
```

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for more details.

## Contact Information
For any questions or inquiries, please contact us at:

- Quentin Missinne: [Q.Missinne@tudelft.nl](mailto:Q.Missinne@tudelft.nl)
- Dequan Ou: [D.Ou@tudelft.nl](mailto:D.Ou@tudelft.nl)
- Reinier Vos: [R.W.Vos@tudelft.nl](mailto:R.W.Vos@tudelft.nl)

We will be happy to answer your questions and assist you! 🙂
