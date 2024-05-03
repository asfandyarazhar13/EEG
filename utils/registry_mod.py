import warnings
import inspect
import torch
from typing import Dict, Type, Any, List

# Global dictionary that holds all registries for different components
_REGISTRIES: Dict[str, Dict[str, Any]] = {
    "optimizer": {},
    "scheduler": {},
    "criterion": {}
}

def add_registry(name: str, registry: Dict[str, Any]) -> None:
    """
    Add or overwrite an existing registry.

    Args:
        name (str): The name of the registry.
        registry (Dict[str, Any]): A dictionary containing the registry items.

    Raises:
        Warning: If the registry name already exists and will be overwritten.
    """
    if name in _REGISTRIES:
        warnings.warn(
            f"The registry '{name}' already exists and will be overwritten. "
            f"Ensure that this is the expected behavior."
        )
    _REGISTRIES[name] = registry

def register(registry_name: str, key: str, value: Any) -> None:
    """
    Register a new key-value pair into a specified registry.

    Args:
        registry_name (str): The name of the registry.
        key (str): The key under which the item is to be stored.
        value (Any): The item to store in the registry.

    Raises:
        Warning: If the key already exists in the registry and will be overwritten.
    """
    registry = _REGISTRIES[registry_name]
    if key in registry:
        warnings.warn(
            f"The registry '{registry_name}' already contains the key '{key}', and it "
            f"will be overwritten. Ensure that this is the expected behavior."
        )
    registry[key] = value

def instantiate(registry_name: str, params: Dict[str, Any], **kwargs) -> Any:
    """
    Instantiate an object from a registry using parameters.

    Args:
        registry_name (str): The registry to use for instantiation.
        params (Dict[str, Any]): Parameters including the target class `_target_` and other initialization arguments.

    Returns:
        Any: An instance of the class specified by `_target_`.
    """
    target_class = params.pop("_target_")
    params.update(kwargs)
    return _REGISTRIES[registry_name][target_class](**params)

def get_registry(name: str) -> Dict[str, Any]:
    """
    Get the registry by name.

    Args:
        name (str): The name of the registry to retrieve.

    Returns:
        Dict[str, Any]: The registry dictionary.
    """
    return _REGISTRIES[name]

################################################################################
# Utility Functions
################################################################################

def _get_default_subclasses(search_module: object, object_types: List[Type]) -> Dict[str, Type]:
    """
    Find subclasses of given types within a module.

    Args:
        search_module (object): The module to search in.
        object_types (List[Type]): A list of types to find subclasses of.

    Returns:
        Dict[str, Type]: A dictionary of class names to class types.
    """
    subclasses = {}
    for name, obj in inspect.getmembers(search_module):
        if inspect.isclass(obj) and any(issubclass(obj, t) for t in object_types) and not any(obj is t for t in object_types):
            subclasses[name] = obj
    return subclasses

def _populate_torch_registries() -> None:
    """
    Automatically populate registries for torch optimizers, criterions, and schedulers.
    """
    optimizer_type = torch.optim.Optimizer
    criterion_type = torch.nn.modules.loss._Loss
    scheduler_types = [
        torch.optim.lr_scheduler.LRScheduler,
        torch.optim.lr_scheduler._LRScheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        torch.optim.lr_scheduler.CosineAnnealingLR
    ]

    for name, cls in _get_default_subclasses(torch.optim, [optimizer_type]).items():
        register("optimizer", name, cls)

    for name, cls in _get_default_subclasses(torch.nn, [criterion_type]).items():
        register("criterion", name, cls)

    for name, cls in _get_default_subclasses(torch.optim.lr_scheduler, scheduler_types).items():
        register("scheduler", name, cls)

# Populate torch registries upon module import
_populate_torch_registries()

if __name__ == "__main__":
    # Example usage: Print out the contents of the registries
    for category_name, category_registry in _REGISTRIES.items():
        print(f"\nRegistry: {category_name}")
        print("-" * 40)
        for key, value in category_registry.items():
            print(f"{key}: {value}")
