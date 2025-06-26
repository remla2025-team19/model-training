# model-training

![Pylint Score](reports/badges/pylint.svg)
![Coverage](reports/badges/coverage.svg)
![Adequacy](reports/badges/adequacy.svg)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Training pipeline for restaurant review sentiment analysis ML model

Repository Link: https://github.com/remla2025-team19/model-training

## Installation & Setup

**Requirements:**

-   Python 3.12.9
-   `make`
-   `dvc` (with Google Cloud support)
-   (Recommended) `virtualenv` or `venv`

**Setup Environment:**

```bash
# Clone the repository
git clone https://github.com/remla2025-team19/model-training.git
cd model-training


# Create the environment with venv, download requirements, and activate
make create_environment
source .venv/bin/activate

# Install Python dependencies
make requirements

# Install DVC with Google Cloud support
pip install 'dvc[gs]'
```

**DVC Remote Setup (Google Cloud):**

In order to run these run the pipelines you will need access to `remla_secret.json`. For people not a part of Team-19, please send a request to "sidsharma620@gmail.com".

```bash
dvc remote add -d sentiment_remote gs://remla2025-team19-bucket -f
dvc remote modify --local sentiment_remote credentialpath /path/to/remla_secret.json
```

## Usage

Run full pipeline with DVC

```bash
dvc repro
```

or run the individual steps

### 1. Download Raw Data

-   **With DVC (Recommended):**

    ```bash
    dvc repro download
    ```

-   **With Make:**

    ```bash
    make download
    ```

-   **With Python:**

    ```bash
    python model_training/dataset.py download
    ```

### 2. Preprocess Data

-   **With DVC (Recommended):**

    ```bash
    dvc repro preprocess
    ```

-   **With Make:**

    ```bash
    make preprocess
    ```

-   **With Python:**

    ```bash
    python model_training/dataset.py preprocess
    ```

### 3. Split Data

-   **With DVC (Recommended):**

    ```bash
    dvc repro split
    ```

-   **With Make:**

    ```bash
    make split
    ```

-   **With Python:**

    ```bash
    python model_training/dataset.py split
    ```

### 4. Train Model

-   **With DVC (Recommended):**

    ```bash
    dvc repro train
    ```

-   **With Make:**

    ```bash
    make train
    ```

-   **With Python:**

    ```bash
    python model_training/modeling/train.py --version 1.0.0
    ```

### 5. Evaluate Model

-   **With DVC (Recommended):**

    ```bash
    dvc repro evaluate
    ```

-   **With Make:**

    ```bash
    make evaluate
    ```

-   **With Python:**

    ```bash
    python model_training/modeling/evaluate.py
    ```

### 6. Push Data/Models to Remote

```bash
dvc push
```

### 7. Custom Experiments

-   Edit `params.yaml` and run:

    ```bash
    dvc exp run -S <stage>.<parameter>=<value>
    ```

## Example Project Organization

```
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── processed
│   └── raw
│
├── models
│   └── sentiment_model_v1.0.0.pkl
│
├── pyproject.toml
├── reports
│   ├── evaluation_metrics.json
│   ├── evaluation_report.txt
│   └── badges/
│       ├── adequacy.svg
│       ├── coverage.svg
│       └── pylint.svg
│
├── requirements.txt
├── dvc.yaml / dvc.lock / params.yaml
├── model_training
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── pipeline.py
│   ├── utils.py
│   └── modeling
│       ├── __init__.py
│       ├── train.py
│       ├── evaluate.py
│       └── predict.py
│
├── tests
│   ├── conftest.py
│   ├── test_data_integrity.py
│   ├── test_infrastructure.py
│   ├── test_metamorphic.py
│   ├── test_model_development.py
│   ├── test_monitoring.py
│   └── test_training.py
└── ...
```

---
