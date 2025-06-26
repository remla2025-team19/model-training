from pathlib import Path

import yaml


def load_params(params_file: Path = Path("params.yaml")):
    """Load parameters from YAML file"""
    if params_file.exists():
        with open(params_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def resolve_model_params(
    model_name: str | None = None,
    model_version: str | None = None,
    params_file: Path = Path("params.yaml")
):
    """Resolve model name and version from CLI args or params.yaml defaults.
    
    Args:
        model_name: CLI-provided model name (takes priority)
        model_version: CLI-provided model version (takes priority)  
        params_file: Path to parameters YAML file
        
    Returns:
        tuple: (final_model_name, final_version)
    """
    # Load parameters from YAML file
    params = load_params(params_file)
    model_params = params.get("model", {})

    # Use CLI arguments if provided, otherwise use params.yaml values, otherwise use defaults
    final_model_name = (
        model_name if model_name is not None else model_params.get("name", "sentiment_model")
    )
    final_version = (
        model_version if model_version is not None else model_params.get("version", "1.0.0")
    )
    
    return final_model_name, final_version
