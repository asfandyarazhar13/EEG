import os
from typing import Union, Dict
from omegaconf import OmegaConf, DictConfig, ListConfig

def parse_config(cfg: Union[str, Dict[str, Union[str, Dict]], DictConfig]) -> Union[DictConfig, ListConfig]:
    """
    Parses a configuration input into an OmegaConf configuration object. It can handle
    configurations provided as file paths (assuming TXT format), plain text strings (assuming JSON or key-value pairs),
    dictionaries, or directly as OmegaConf objects.

    Args:
        cfg (Union[str, Dict[str, Union[str, Dict]], DictConfig]): Configuration input, which can be
            a string representing a file path or data, a dictionary, or an OmegaConf DictConfig.

    Returns:
        Union[DictConfig, ListConfig]: OmegaConf configuration object based on the input.

    Raises:
        ValueError: If the input cfg is not one of the expected types or cannot be parsed.

    Examples:
        >>> parse_config('config.txt')  # From a file path
        >>> parse_config({'key': 'value'})  # From a dictionary
        >>> parse_config("key: value")  # From a plain text string
    """
    if isinstance(cfg, DictConfig):
        # If it's already a DictConfig, return it directly
        return cfg

    elif isinstance(cfg, dict):
        # If it's a dictionary, convert it to DictConfig using OmegaConf
        return OmegaConf.create(cfg)

    elif isinstance(cfg, str):
        # If it's a string, determine if it's a filename or a plain text string
        if os.path.isfile(cfg):
            # Load OmegaConf configuration from a TXT file
            with open(cfg, 'r') as file:
                content = file.read()
                try:
                    # Attempt to parse the file content as YAML (assuming simple YAML-like syntax in TXT)
                    return OmegaConf.create(content)
                except Exception as e:
                    raise ValueError(f"Failed to parse the content of {cfg}. Error: {str(e)}")
        else:
            # Create OmegaConf configuration from a plain text string
            try:
                return OmegaConf.create(cfg)
            except Exception as e:
                raise ValueError(f"Failed to parse the provided string as configuration. Error: {str(e)}")

    else:
        # Raise an error if none of the above types match
        raise ValueError(
            f"Unsupported configuration type '{type(cfg)}'. Expected file path, plain text string, dictionary, "
            "or DictConfig."
        )
