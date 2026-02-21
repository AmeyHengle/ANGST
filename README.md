# _Still Not Quite There!_ Evaluating Large Language Models for Comorbid Mental Health Diagnosis

[![acl]([https://img.shields.io/badge/arxiv-paper
](https://aclanthology.org/2024.emnlp-main.931/))]()


## Table of Contents
1. [Environment](#environment)
2. [Reproducing Paper Results](#reproducing-paper-results)
3. [Additional Support/Issues?](#additional-supportissues)
4. [Citation](#citation)


## Environment
We use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage the environment. Our Python version is <code>3.12.2</code>. To create the environment, run the following command:

```
conda env create -f environment.yaml -n mentallm_env
```

To activate the environment, run the following command:

```
conda activate mentallm_env
```

## Reproducing Paper Results

The <code>predictions</code> folder contains the prediction results from all models across various hyperparameter configurations.

## Additional Support/Issues?

If you face any issues in our code / reproducing our results, raise a GitHub issue.


## Citation 

```
@inproceedings{hengle-etal-2024-still,
    title = "Still Not Quite There! Evaluating Large Language Models for Comorbid Mental Health Diagnosis",
    author = "Hengle, Amey  and
      Kulkarni, Atharva  and
      Patankar, Shantanu Deepak  and
      Chandrasekaran, Madhumitha  and
      D{'}silva, Sneha  and
      Jacob, Jemima S.  and
      Gupta, Rashmi",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.931/",
    doi = "10.18653/v1/2024.emnlp-main.931",
    pages = "16698--16721"
```
