import math
import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

from utils import BaseOutput

@dataclass
class DDIMSchedulerOutput(BaseOutput):
    """
    Output class of the step function
    Args:
        prev_sample: Previous timestep sample,, aka computed x_{t-1},this would be the input of next timestep
        prev_original_sample: Previous timestep original sample, aka the predicted denoised x_{0} based on the model output from the current timestep    
    """
    prev_sample: torch.FloatTensor
    prev_original_sample: Optional[torch.FloatTensor] = None


