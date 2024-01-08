import math
import torch
import torch.nn as nn

import math
import numpy as np

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


def betas_for_alpha_bar(
    num_diffusion_steps: Union[int, torch.IntTensor],
    max_beta: float = 0.999,
    alpha_transform_type: str = 'cosine') -> torch.tensor:
    """
    Args:
        num_diffusion_steps: Number of diffusion steps
        max_beta: Maximum value of bety, same with the DDPM paper since DDIM is not retrained 
        alpha_transform_type: Type of noise scheduler
        alpha ** 2 + beta ** 2 = 1
    This is the scheduler mentioned in the IDDPM paper
    This function defines the alpha, thus beta is also defined for every timestep
    This is the noise scheduler for our diffusion model
    """
    if alpha_transform_type == 'cosine':
        def alpha_bar_fn(t):
            return math.cos((t+0.008)/1.008*math.pi/2)**2
    elif alpha_transform_type == 'linear':
        def alpha_bar_fn(t):
            return math.exp(t*-12.0)
    else:
        raise ValueError(f'Unsupported Alpha Transform Type: {alpha_transform_type}')
    
    betas = []
    for i in range(num_diffusion_steps):
        t_1 = i / num_diffusion_steps
        t_2 = (i + 1) / num_diffusion_steps
        betas.append(max(1 - alpha_bar_fn(t_2)/alpha_bar_fn(t_1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

def rescale_zero_terminal_snr(beta: torch.FloatTensor) -> List[torch.FloatTensor]:
    """
    This function defined by the paper:
    https://arxiv.org/abs/2305.08891
    rescale the beta with zero terminal SNR
    SNR is the signal to noise ratio, in the diffusion model, during the diffusion process, the noise is added to the original image,
    the final goal of this forward process would be at the end of the process, we would have a pure noise function, with its distribution as a Gaussian distribution
    Thus the SNR would be zero at the end of the diffusion process
    The result of this function would be a beta list with the same length as the diffusion steps, with the last element as one (1 - alpha_T = 1 - 0 = 1)
    """
    alpha = 1 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    alpha_cumprod_sqrt = math.sqrt(alpha_cumprod)

    alpha_cumprod_sqrt_0 = alpha_cumprod_sqrt[0].clone()
    alpha_cumprod_sqrt_T = alpha_cumprod_sqrt[-1].clone()
    # rescale the last timestep alpha to zero
    alpha_cumprod_sqrt -= alpha_cumprod_sqrt_T
    # rescale the first timestep alpha back into the original value
    alpha_cumprod_sqrt *= alpha_cumprod_sqrt_0 / (alpha_cumprod_sqrt_0 - alpha_cumprod_sqrt_T)
    
    alpha_cumprod = alpha_cumprod_sqrt ** 2
    # calculate every single alpha at every time step
    alpha = alpha_cumprod[1:] / alpha_cumprod[:-1]
    alpha = torch.cat(alpha_cumprod[0:1], alpha)
    beta = 1 - alpha
    return beta

class DDIMScheduler():
    def __init__(self,
                 num_training_steps: int,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_scheduler: str = 'linear',
                 trained_beta: Optional[Union[np.ndarray, List[float]]] = None, # this line of code is super elegent
                 clip_sample: bool = True,
                 clip_sample_threshold: float = 1.0,
                 sample_max_value: float = 1.0,
                 set_alpha_to_one: bool = True,
                 thresholding: bool = False,
                 dynamic_thresholding_ratio: float = 0.995,
                 prediction_type: str = 'epsilon',
                 time_spacing: str = 'leading',
                 rescale_zero_terminal_snr: bool = True,
                 ):
        if trained_beta is not None: # transfer the trained beta into a tensor
            self.betas = torch.tensor(trained_beta, dtype=torch.float32)
        elif beta_scheduler == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_training_steps, dtype=torch.float32)
        elif beta_scheduler == 'scaled_linear':
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        elif beta_scheduler == 'squaredcos_cap_v2':
            # Glide cosine scheduler
            self.betas = betas_for_alpha_bar(num_training_steps)
        else:
            raise ValueError(f"{beta_scheduler} is not implemented for self.__class__") # ddpm = DDIMScheduler(), then ddpm.__class__ = DDIMScheduler
        
        # rescale the beta with zero terminal SNR   
        if rescale_zero_terminal_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alpha = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alpha_cumprod[0] # alpha should be a sequence in descending order

        self.init_noise_sigma = 1.0 #needs to be understood

        # set the inference 
        self.num_inference_step = None
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy().astype(np.float64))
    
    def scale_model_input(self, sample:torch.FloatTensor, timestep: Optional[int]=None) -> torch.FloatTensor:
        """
        Ensure smoothness between different diffuser schedulers
        """
        return sample
    
    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.FloatTensor:
        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        # check the DDIM paper for the following result
        variance = (beta_prod_t * beta_prod_t_prev) / (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
    
    def _thresholding_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        """
        Dynamic thresholding for the sample
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float() # convert to float32
        
        sample = sample.reshape(batch_size, channels*torch.prod(remaining_dims))

        abs_sample = torch.abs(sample)


        


    
