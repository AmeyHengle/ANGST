# mental-health-comorbitidy-classification

## Getting Started
1. Make sure you have `git`, `python(>=3.8, <3.10)`, [`poetry`](https://python-poetry.org/docs/#installation) installed. Preferably within a virtual environment.

2. Install dependencies
```shell
cd mental-health-comorbitidy-classification
poetry install
git init
git add .
git commit -m "add: initial commit."
```

## Directory Structure

| File                                      | Description                                                                  |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| **project**                               | Main directory containing all the code            |
| **project/data**                          | Data directory containing the train, test and annotation files |
| **project/creds**                         | Directory containing all API access credentials ( aws / open-ai )|
| **project/runs**                              | Directory to keep track of all model runs (train / eval). For each run, we store the best_model, classfication args, eval results, metrics, etc.  |
| **project/utils**                             | Program containing utility functions              |
| **project/constants**                         | Program for accessing costant variables, shared variables or default configs   |
| **CHANGELOG.md**                          | Track changes in the code, datasets, etc.                                    |
| **LICENSE**                               | Depending on your usage choose the correct copy, don't keep the default!     |
| **pyproject.toml**                        | Track dependencies here. Also, this means you would be using poetry.         |
| **README.md**                             | This must ring a bell.                                                       |