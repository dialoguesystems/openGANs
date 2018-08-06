""" Main entry point of the odlg library """
from __future__ import division, print_function

import ogan.inputters
import ogan.modules
import ogan.utils
from ogan.trainer import Trainer
import sys
import ogan.utils.optimizers


# For Flake
__all__ = [ogan.inputters, ogan.modules, ogan.utils, "Trainer"]

__version__ = "0.0.1"