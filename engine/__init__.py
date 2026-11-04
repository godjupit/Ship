"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

# for register purpose
import os
if os.environ.get("INFERENCE_IMPORTS", "0") != "1":
    from . import optim
    from . import data
from . import rtv4

from .backbone import *

from .backbone import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)