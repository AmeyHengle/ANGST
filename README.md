This repository accompanies the EMNLP 2024 paper:

**“Still Not Quite There! Evaluating Large Language Models for Comorbid Mental Health Diagnosis”**

* 📄 Paper: [https://aclanthology.org/2024.emnlp-main.931/](https://aclanthology.org/2024.emnlp-main.931/)
* 🤗 Dataset (ANGST): [https://huggingface.co/datasets/ameyhengle/ANGST](https://huggingface.co/datasets/ameyhengle/ANGST)

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Reproducing Paper Results](#reproducing-paper-results)
3. [Support / Issues](#support--issues)
4. [Citation](#citation)

---

## Environment Setup

We use **Miniconda** to manage dependencies.

* Python version: `3.12.2`

### 1. Install Miniconda

If you do not already have Miniconda installed, follow the official installation guide:
[https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

### 2. Create the Environment

```bash
conda env create -f environment.yaml -n mentallm_env
```

### 3. Activate the Environment

```bash
conda activate mentallm_env
```

---

## Reproducing Paper Results

The `predictions/` folder contains model outputs across different hyperparameter configurations used in the paper.

To reproduce results:

1. Ensure the environment is activated.
2. Download the ANGST dataset from Hugging Face.
3. Run the corresponding scripts provided in the repository (see individual script documentation for details).

---

## Support / Issues

If you encounter any issues with the code or while reproducing results, please open a GitHub issue with:

* A clear description of the problem
* Steps to reproduce the issue
* Relevant logs or error messages

We’ll do our best to help.

---

## Citation

If you use this dataset or code in your work, please cite:

```bibtex
@inproceedings{hengle-etal-2024-still,
    title = "Still Not Quite There! Evaluating Large Language Models for Comorbid Mental Health Diagnosis",
    author = "Hengle, Amey and
              Kulkarni, Atharva and
              Patankar, Shantanu Deepak and
              Chandrasekaran, Madhumitha and
              D'silva, Sneha and
              Jacob, Jemima S. and
              Gupta, Rashmi",
    editor = "Al-Onaizan, Yaser and
              Bansal, Mohit and
              Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.931/",
    doi = "10.18653/v1/2024.emnlp-main.931",
    pages = "16698--16721"
}
```