"""Get configurable parameters."""

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
from warnings import warn

from omegaconf import DictConfig, ListConfig, OmegaConf

from etudelib.data.utils import TestSplitMode, ValSplitMode

def _get_now_str(timestamp: float) -> str:
    """Standard format for datetimes is defined here."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")

def get_configurable_parameters(
    model_name: Optional[str] = None,
    config_path: Optional[Union[Path, str]] = None,
    weight_file: Optional[str] = None,
    config_filename: Optional[str] = "config",
    config_file_extension: Optional[str] = "yaml",
) -> Union[DictConfig, ListConfig]:
    """Get configurable parameters.
    Args:
        model_name: Optional[str]:  (Default value = None)
        config_path: Optional[Union[Path, str]]:  (Default value = None)
        weight_file: Path to the weight file
        config_filename: Optional[str]:  (Default value = "config")
        config_file_extension: Optional[str]:  (Default value = "yaml")
    Returns:
        Union[DictConfig, ListConfig]: Configurable parameters in DictConfig object.
    """
    if model_name is None and config_path is None:
        raise ValueError(
            "Both model_name and model config path cannot be None! "
            "Please provide a model name or path to a config file!"
        )

    if config_path is None:
        config_path = Path(f"etudelib/models/{model_name}/{config_filename}.{config_file_extension}")

    config = OmegaConf.load(config_path)

    # keep track of the original config file because it will be modified
    config_original: DictConfig = config.copy()

    # if the seed value is 0, notify a user that the behavior of the seed value zero has been changed.
    if config.project.get("seed") == 0:
        warn(
            "The seed value is now fixed to 0. "
            "If you want to use the random seed, please select `None` for the seed value "
            "(`null` in the YAML file) or remove the `seed` key from the YAML file."
        )

    config = update_datasets_config(config)
    config = update_input_size_config(config)

    # Project Configs
    project_path = Path(config.project.path) / config.model.name / config.dataset.name

    # set to False by default for backward compatibility
    config.project.setdefault("unique_dir", False)

    if config.project.unique_dir:
        project_path = project_path / f"run.{_get_now_str(time.time())}"

    else:
        project_path = project_path / "run"
        warn(
            "config.project.unique_dir is set to False. "
            "This does not ensure that your results will be written in an empty directory and you may overwrite files."
        )

    (project_path / "weights").mkdir(parents=True, exist_ok=True)
    # write the original config for eventual debug (modified config at the end of the function)
    (project_path / "config_original.yaml").write_text(OmegaConf.to_yaml(config_original))

    config.project.path = str(project_path)

    # loggers should write to results/model/dataset/category/ folder
    config.trainer.default_root_dir = str(project_path)

    if weight_file:
        config.trainer.resume_from_checkpoint = weight_file

    (project_path / "config.yaml").write_text(OmegaConf.to_yaml(config))

    return config