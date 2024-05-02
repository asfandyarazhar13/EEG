import torch
import glob
import os
from typing import List
from omegaconf import OmegaConf, DictConfig

class OutputVersioning:
    """ 
    Manages output versioning by creating uniquely named directories within a base directory 
    based on an incremental versioning scheme.
    """

    def __init__(self, base: str, prefix: str = "version") -> None:
        """
        Initialize the OutputVersioning instance.

        Args:
            base (str): The base directory where the version directories will be created.
            prefix (str): The prefix to use for version directories. Default is "version".

        Raises:
            ValueError: If the specified base directory does not exist.
        """
        self.base = base
        self.prefix = prefix

        if not os.path.exists(self.base):
            raise ValueError(f"The input base directory '{base}' must already exist.")

    def create_new_output(self) -> str:
        """
        Create a new output directory with the next version number.

        Returns:
            str: The path to the newly created directory.

        Raises:
            AssertionError: If the output directory already exists or the base directory does not exist.
        """
        output_dir = os.path.join(self.base, self._get_next_dir())
        assert not os.path.exists(output_dir), "Output directory should not already exist"
        assert os.path.exists(self.base), "Base directory must exist"
        os.makedirs(output_dir)
        return output_dir

    def _get_next_dir(self) -> str:
        """
        Determine the next directory name by finding the highest existing version number and incrementing it.

        Returns:
            str: The next directory name using the prefix and next version number.
        """
        version_numbers: List[int] = []
        pattern = os.path.join(self.base, f"{self.prefix}_*")
        for file in glob.glob(pattern):
            if os.path.isdir(file):
                try:
                    version = int(file.split("_")[-1])
                    version_numbers.append(version)
                except ValueError:
                    continue

        next_version = max(version_numbers, default=-1) + 1
        return f"{self.prefix}_{next_version}"

def get_default_device() -> str:
    """
    Determine the default device based on CUDA availability.

    Returns:
        str: Returns "cuda" if CUDA is available, otherwise "cpu".
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_pretrained(path: str, model: torch.nn.Module) -> torch.nn.Module:
    """
    Load a pretrained model state from a specified path.

    Args:
        path (str): Path to the saved model state.
        model (torch.nn.Module): The model instance to which the state will be loaded.

    Returns:
        torch.nn.Module: The model with the loaded state.
    """
    model.load_state_dict(torch.load(path)["model_state_dict"])
    return model

def save_config(path: str, config: DictConfig) -> None:
    """
    Save a configuration object to a file.

    Args:
        path (str): Path where the configuration should be saved.
        config (DictConfig): The OmegaConf configuration object to save.
    """
    OmegaConf.save(config, path)
