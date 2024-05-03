import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
from typing import Optional, Union, Dict, List, Any, Tuple
import logging
import os
import time
import csv
from collections import defaultdict
import re
import unicodedata
from tqdm.auto import tqdm
import gc

__version__ = "0.1.0"
logger = logging.getLogger("trainer")

class Callback:
    """ Base class for defining training callbacks. """
    def on_train_start(self, trainer: 'Trainer') -> None:
        pass

    def on_epoch_start(self, trainer: 'Trainer') -> None:
        pass

    def on_train_epoch_start(self, trainer: 'Trainer') -> None:
        pass

    def on_train_step_start(self, trainer: 'Trainer', batch: Any) -> None:
        pass

    def on_train_step_end(self, trainer: 'Trainer', output: Any) -> None:
        pass

    def on_train_epoch_end(self, trainer: 'Trainer') -> None:
        pass

    def on_val_epoch_start(self, trainer: 'Trainer') -> None:
        pass

    def on_val_step_start(self, trainer: 'Trainer', batch: Any) -> None:
        pass

    def on_val_step_end(self, trainer: 'Trainer', output: Any) -> None:
        pass

    def on_val_epoch_end(self, trainer: 'Trainer') -> None:
        pass

    def on_epoch_end(self, trainer: 'Trainer') -> None:
        pass

    def on_train_end(self, trainer: 'Trainer') -> None:
        pass

    def on_keyboard_interrupt(self, trainer: 'Trainer') -> None:
        pass

    def on_exception(self, trainer: 'Trainer', e: Exception) -> None:
        pass

    def log(self, trainer: 'Trainer', k: str, v: float, to_pbar: bool = True) -> None:
        pass

class TrainerCallbacks:
    """ Manages a collection of callbacks and delegates method calls to them. """
    def __init__(self) -> None:
        self.callbacks: List[Callback] = []

    def add_callback(self, callback: Callback) -> None:
        """ Add a callback to the list of callbacks. """
        self.callbacks.append(callback)

    def remove_callback(self, callback_type: type) -> None:
        """ Remove a callback from the list based on its type. """
        self.callbacks = [cb for cb in self.callbacks if not isinstance(cb, callback_type)]

    def trigger_event(self, event_name: str, *args, **kwargs) -> None:
        """ Trigger an event, calling the corresponding methods on all callbacks. """
        for callback in self.callbacks:
            method = getattr(callback, event_name)
            method(*args, **kwargs)

@dataclasses.dataclass
class TrainerArgs:
    device: Optional[str] = None
    min_epochs: int = 0
    max_epochs: int = 100
    grad_accum_steps: int = 1
    check_val_every_n_epochs: int = 1
    log_every_n_steps: int = 1
    resume_from_checkpoint: Optional[str] = None
    output_dir: str = "./content/"

@dataclasses.dataclass
class TrainerState:
    mode: Optional[str] = None
    max_train_steps: Optional[int] = None
    max_val_steps: Optional[int] = None
    curr_epoch: int = 0
    curr_step: int = 0
    is_training: bool = False
    should_stop: bool = False

class Trainer(TrainerCallbacks):
    """ Implements a lightweight training loop. """
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[lr_scheduler.LRScheduler],
        callbacks: Union[Callback, List[Callback]] = [],
        args: Optional[TrainerArgs] = None
    ) -> None:
        super().__init__()
        self.model = model.to(args.device if args and args.device else get_default_device())
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args or TrainerArgs()
        self.state = TrainerState()
        self.setup_callbacks(callbacks)

    def setup_callbacks(self, callbacks: Union[Callback, List[Callback]]) -> None:
        """ Initialize default and user-provided callbacks. """
        self.add_callback(TQDMLogger())
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        for cb in callbacks:
            self.add_callback(cb)

    def train(self) -> None:
        """ Start the training process. """
        self.trigger_event('on_train_start', self)
        for self.state.curr_epoch in range(self.args.min_epochs, self.args.max_epochs + 1):
            self.trigger_event('on_epoch_start', self)
            self.train_epoch()
            self.trigger_event('on_epoch_end', self)
            if self.state.should_stop:
                break
        self.trigger_event('on_train_end', self)

    def train_epoch(self) -> None:
        """ Run a single epoch of training and validation. """
        self.model.train()
        self.trigger_event('on_train_epoch_start', self)
        # Training loop - Implement training logic here
        self.trigger_event('on_train_epoch_end', self)
        self.model.eval()
        self.trigger_event('on_val_epoch_start', self)
        # Validation loop - Implement validation logic here
        self.trigger_event('on_val_epoch_end', self)

def get_default_device() -> str:
    """ Get the default device available for training. """
    return "cuda" if torch.cuda.is_available() else "cpu"

def regexer(value: str, allow_unicode: bool = False) -> str:
    """
    Convert strings to a URL by removing characters that aren't alphanumerics,
    underscores, or hyphens. Converts to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.

    Args:
        value (str): The string to apply regex on.
        allow_unicode (bool): Whether to allow unicode characters in the result.

    Returns:
        str: The final string.
    """
    value = unicodedata.normalize('NFKC', value) if allow_unicode else unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)

def write_to_csv(filename: str, rows: List[List[Any]], columns: Optional[List[str]] = None) -> None:
    """
    Write data to a CSV file.

    Args:
        filename (str): The path to the CSV file.
        rows (List[List[Any]]): The data rows to write.
        columns (Optional[List[str]]): The header columns for the CSV. If provided, write as the first row.
    """
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)
        if mode == 'w' and columns:
            writer.writerow(columns)
        writer.writerows(rows)
