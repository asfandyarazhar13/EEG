import argparse
import dataclasses
import inspect
import os
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import ConfigDict
from pydantic.type_adapter import TypeAdapter

from config_mod import parse_config
import registry_mod

# Constants to indicate missing configuration parameters
missing = "lol?"

# Global Pydantic configuration allowing arbitrary types
pydantic_config_dict = ConfigDict(arbitrary_types_allowed=True)


def pre_parse_user_config(
    default_config: Optional[str] = None,
    *,
    aliases: Sequence[str] = ["-c", "--config"],
    argv: Optional[List[str]] = None,
    dest: str = "config"
) -> DictConfig:
    """
    Pre-parse user-provided configuration to allow dynamic modifications before
    full CLI argument parsing.

    Args:
        default_config (Optional[str]): Default configuration file path.
        aliases (Sequence[str]): Command-line flags to accept configuration file.
        argv (Optional[List[str]]): List of command-line arguments to parse.
        dest (str): The destination attribute name for parsed config.

    Returns:
        DictConfig: Parsed OmegaConf configuration object.

    Raises:
        ValueError: If provided config file path does not exist.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(*aliases, default=default_config, type=str, dest=dest)
    args, _ = parser.parse_known_args(argv)

    if args.config is not None and not os.path.exists(args.config):
        raise ValueError(f"The config path '{args.config}' does not exist!")

    config = parse_config(getattr(args, dest) or {})
    if isinstance(config, ListConfig):
        raise RuntimeError("The config must be a dictionary, not a list.")

    return config


def yaml_type_handler(f_type: object) -> Callable:
    """
    Return a callable that parses and validates a YAML string into a specific type.

    Args:
        f_type (object): The type to validate the YAML string against.

    Returns:
        Callable: A function that takes a string, parses it as YAML, and validates
        it against the provided type.
    """
    validator = TypeAdapter(f_type, config=pydantic_config_dict)

    def caster(f_inp: str) -> Any:
        f_inp = _static_handler(f_inp)
        value = yaml.load(f_inp, Loader=get_yaml_loader())
        value = validator.validate_python(value)
        return value

    return caster


def add_register_group_to_parser(
    parser: argparse.ArgumentParser, 
    name: str,
    *,
    default: Optional[Any] = None,
    exclude: List[str] = ["self"],
    **kwargs
) -> None:
    """
    Add a group of arguments to a parser based on a registry.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add arguments to.
        name (str): The name of the registry.
        default (Optional[Any]): Default value if not specified by user.
        exclude (List[str]): List of parameter names to exclude.
        kwargs: Additional keyword arguments for the argparse group.

    Description:
        Dynamically adds command-line arguments based on the contents of a specified
        registry, allowing the user to select configurations interactively.
    """
    register = registry_mod.get_registry(name)
    config = pre_parse_user_config()

    if name in config:
        default = config[name].get("_target_", default)
    dest = f"{name}._target_"

    group = parser.add_argument_group(name)
    group.add_argument("--" + name, default=default, type=str, dest=dest, **kwargs)

    registry_key = pre_parse_field(dest, default=default)
    if registry_key:
        add_class_to_parser(
            parser=group,
            cls=register.get(registry_key),
            prefix=name,
            exclude=exclude
        )


def add_class_to_parser(
    parser: argparse.ArgumentParser, 
    cls: object, 
    *, 
    prefix: Optional[str] = None, 
    exclude: List[str] = ["self"],
    **kwargs
) -> None:
    """
    Add arguments to a parser for all public attributes of a class that are not excluded.

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguments to.
        cls (object): The class from which to take the attributes.
        prefix (Optional[str]): A prefix to add before argument names.
        exclude (List[str]): Attributes to exclude from being added.
        kwargs: Additional keyword arguments for adding arguments.
    """
    params = _unfold_parameters_dict(_get_class_parameters(cls, exclude=exclude))
    _add_parameters_to_parser(parser, params, prefix=prefix, **kwargs)


def _get_class_parameters(cls: object, *, exclude: List[str] = ["self"]) -> Dict[str, Tuple[Type, Any]]:
    """
    Retrieve parameters for the class constructor.

    Args:
        cls (object): The class from which to retrieve constructor parameters.
        exclude (List[str]): A list of parameter names to exclude.

    Returns:
        Dict[str, Tuple[Type, Any]]: A dictionary mapping parameter names to their type and default value.
    """
    signature = inspect.signature(cls.__init__)
    return {
        name: (param.annotation if param.annotation != inspect.Parameter.empty else Any,
               param.default if param.default != inspect.Parameter.empty else missing)
        for name, param in signature.parameters.items() if name not in exclude
    }
