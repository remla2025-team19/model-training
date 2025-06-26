from pathlib import Path

import yaml


def load_params(params_file: Path = Path("params.yaml")):
    """Load parameters from YAML file"""
    if params_file.exists():
        with open(params_file, "r") as f:
            return yaml.safe_load(f)
    return {}
