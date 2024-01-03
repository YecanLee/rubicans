import transformers
import torch
import torch.nn as nn
import math
from dataclasses import dataclass
import numpy as np

from typing import List, Union

from typing import Optional

from utils import BaseOutput

# try to use the @measure_time decorator to measure the time of the function


# the dataclass decorator is used to create a class with a predefined structure, this means that we can define the output format of the scheduler
@dataclass
class DDPMSchedulerOutput(BaseOutput):
    """
    Output of the DDPMSchduler
    The prev_sample is the previous timestample x_{t-1} and this should also be the input of next step
    The prev_sample_output is the previous time step denoising output x_{0} , this can be used for checking and also for guidance
    """
    prev_sample = torch.FloatTensor
    prev_sample_output = Optional[torch.FloatTensor] = None

def rescale_alphas_terminal(betas: int) -> int:
    """
    This function is based on the paper "Common Diffusion Noise Schedules and Sample Steps are flawed,
    In that paper, the author argues that the diffusion process should be rescaled at the terminal step,
    the alpha_{T} should be rescaled to 0, and the beta_{T} should be rescaled to 1
    """
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_t_bar = alphas_bar.sqrt()

    # store the alphas_{0} and alphas_{T} for rescaling
    alphas_0_bar = alphas_t_bar[0].clone()
    alphas_T_bar = alphas_t_bar[-1].clone()

    # rescale the alphas_{T} to 0
    alphas_t_bar -= alphas_T_bar
    # Now the alphas_{T} is 0, we need to rescale other alphas_{t}
    alphas_t_bar *= alphas_0_bar/alphas_0_bar - alphas_T_bar
    
    # square the alphas_t_bar back to the alphas_t
    alphas_t = alphas_t_bar ** 2
    alphas_bar = alphas_t[1:] / alphas_t[:-1]
    alphas = torch.cat([alphas_0_bar.unsqueeze(0), alphas_bar])
    betas = 1 - alphas

def betas_for_alpha_bar(
    num_diffusions_steps:int,
    max_beta:0.999,
    alpha_transform_type:str = 'cosine',
):
    """
    alphas = 1 - betas
    x_{t} = math.sqrt((alpha)) * x_{t-1} + math.sqrt(1 - alpha) * eta_{t}, eta_{t} ~ N(0,1)
    eta: Gaussian noise
    x_t: output of the forward diffusion process, no training parameters 
    x_t-1: input of the forward diffusion process, no training parameters
    During every step of the diffusion process during range(1, T), we use a different value of alpha, and we use a different value of beta
    The beta would be used to calculate the noise level for the current step, it becomes larger and larger as the diffusion process goes on
    """
    if alpha_transform_type == 'cosine':
        return 

class DDPMSchduler(SchedulerMixin, ConfigMixin):
    """
    The SchedulerMixin and ConfigMixin are used to store the results of the scheduler
    """
    