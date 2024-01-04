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
    variance: clip the variance of the noise to control the noise level
    clip_sample: clip the sample to control the noise level
    prediction_type: whether to predict the noise epsilon or to predict the x_0 directly
    steps_offset: the offset of the steps added to the inference step, 
    """
    def __init__(self,
                 num_training_steps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_scheduler: str = "linear",
                 trained_betas: Optional[Union[np.narray, List[float]]] = None,
                 variance: str = 'fixed_small',
                 clip_sample: bool = True,
                 clip_sample_threshold: float = 1.0,
                 prediction_type: str = 'epsilon',
                 thresholding: bool = False,
                 dynamic_thresholding_ratio: float = 0.995, 
                 sample_max_value: float = 1.0,
                 timestep_spacing: str = 'leading',
                 steps_offset: int = 0,
                 rescale_betas_zero_sbr: int = False
                 ):
        if trained_betas is not None:
            self.betas = torch.Tensor(trained_betas, dtype=torch.float32)
        elif beta_scheduler == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_training_steps, dtype = torch.float32)
        elif beta_scheduler == 'scaled_linear':
            # this scheduler only works latent diffusion models
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32)**2
        elif beta_scheduler == 'squaredcos_cap_v2':
            self.betas = betas_for_alpha_bar(num_training_steps)
        elif beta_scheduler == 'sigmoid':
            beta = torch.linspace(-6, 6, num_training_steps, dtype=torch.float32)
            self.betas = torch.sigmoid(beta)*(beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f'beta_scheduler {beta_scheduler} is not implemented for {self.__class__}')
        if rescale_alphas_terminal:
            self.betas = rescale_alphas_terminal(self.betas)
        
        self.variance = variance
        self.clip_sample = clip_sample

        self.alphas =  1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim = 0)
        self.one = torch.Tensor(1.0)

        self.init_noise_sigma = 1.0
        self.num_training_steps = num_training_steps    

        self.custom_timesteps = None
        self.num_inference_steps = False
        # Here is the basic timesteps scheduler without time spacing
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
        self.variance_type = variance

        self.steps_offset = steps_offset
        self.timestep_spacing = timestep_spacing
    
    def scale_model_input(self, x: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        This function is used in case the input of the denoising model needs to be scaled
        Args:
            x: the input of the denoising model
            timestep: the current timestep of the diffusion process
        Returns:
            torch.FloatTensor:
            The scaled input sample
        """
        return x
    
    def set_timesteps(self, 
                      num_inference_steps: Optional[int]=None,
                      device: Union[str, torch.device]=None,
                      timesteps: Optional[List[int]]=None
                      ):
        """
        Sets the timestep for the scheduler
        Args:
            num_inference_steps: the number of inference steps to generate an image based on the trained model
            device: the device to use
            timesteps: the custom timestep to use, this is the space between the timesteps
        """
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError('Only one kind of timestep can be set')
        if timesteps is not None:
            for i in range(1, len(timesteps)-1):
                if timesteps[i] >= timesteps[i-1]:
                    raise ValueError('The timesteps must be in descending order')
                if timesteps[i] >= self.num_training_steps:
                    raise ValueError(f'The timesteps must be smaller than the number of training steps {self.num_training_steps}')
                timesteps = np.array(timesteps)
                self.custom_timesteps = True
            else:
                if num_inference_steps > self.num_training_steps:
                    raise ValueError(f'The inference steps: {num_inference_steps} must be smaller than the training timesteps:
                                     {self.num_training_steps} as the UNet model with this specific schduler is only capaple of 
                                     dealing with this number of denoising timesteps: {self.num_training_steps}')

                self.num_inference_steps = num_inference_steps
                self.custom_timesteps = False

                # The following section was mentioned in the paper "Common Diffusion Noise Schedules and Sample Steps are flawed"
                # Please check the table 2, Discretization for more details    
                # We need a denoising scheduler, thus we need to add [::-1]
                # The comment under this table also mentioned that in practice the timestep range would be [0, 999] instead of [1, 1000]
            if self.timestep_spacing == 'linspace':
                timesteps = np.linspace(0, self.num_training_steps - 1, self.num_inference_steps).round()[::-1].copy().astype(np.int64)
            #elif self.timestep_spacing == 'leading':
                #timesteps = np.arange(1, self.num_training_steps, self.num_training_steps//self.num_inference_steps)[::-1].copy().astype(np.int64)
            elif self.timestep_spacing == 'leading':
                factor = self.num_training_steps//self.num_inference_steps
                timesteps = (np.arange(0, self.num_inference_steps) * factor).round()[::-1].copy().astype(np.int64)
                timesteps += self.steps_offset
            elif self.timestep_spacing == 'trailing':
                factor = self.num_training_steps/self.num_inference_steps
                timesteps = np.round(np.arange(self.num_training_steps, 0, -factor)).astype(np.int64)
            else:
                raise ValueError(f"The timestep spacing {self.timestep_spacing} is not supported yet,
                                  make sure to choose one of them from 'linspace', 'leading', 'trailing'.")
        
        # put the timesteps onto the Gpu
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def _get_thresholding(self, x: torch.FloatTensor) -> torch.FloatTensor:
        pass

    def _get_variance(self, t: int, prediction_type:Optional[str]=None, variance_type:Optional[str]=None):
        """
        This function is based on the DDPM paper function 6 and function 7, please check the paper for the beta_bar_{t} in the function 7
        """
        

