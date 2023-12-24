""" PyTorch ViT model."""

import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# from models.modeling_utils import PreTrainedModel

from einops import rearrange

class Config:
    image_size = [224, 224]
    patch_size = 16
    num_channels = 1
    num_patches = (image_size[0] * image_size[1] // patch_size * patch_size) ** 2
    hidden_size = 768
    intermediate_size = 3072    
    num_heads = 12
    qkv_bias = True
    layer_norm_eps = 1e-12
    num_hidden_layers = 12  

class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self):
        super().__init__()
        image_size, patch_size = Config.image_size, Config.patch_size
        num_channels, hidden_size = Config.num_channels, Config.hidden_size

        # image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        """
        Pixel value is the image itself
        """
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        # the shape would be [batch_size, hidden_size, height//patch_size, width//patch_size] before the flatten operation, after flatten we have [batch_size, hidden_size, num_patches]
        # then after the transpose operation, we have [batch_size, num_patches, hidden_size]
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self) -> None:
        super().__init__()
        self.patch_embeddings = ViTPatchEmbeddings()
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, Config.hidden_size))
        self.dropout = nn.Dropout(0., inplace=False)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.
        In our case the shape of the image would be (80, 15), resized from (1200, 240)
  
        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        
        # remove the cls token, since we are only using the ViT Encoder to encode the image
        # If no interpolation is needed, return the position embeddings as they are
        num_patches = embeddings.shape[1]
        num_positions = self.position_embeddings.shape[1]
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        
        # Dim is our ViT model's model dimension
        dim = embeddings.shape[-1]
        h0 = height // Config.patch_size
        w0 = width // Config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        # the int function in the following function is to make sure h and w are integers, so the weird number problem after we add some special tokens will be solved
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        # We first reshape the embedding dimensions from (batch_size, )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        _, _, height, width = pixel_values.shape

        # the parameters should be kept since they are calling the forward function of the ViTPatchEmbeddings class
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        
        # comment out the CLS token since we are not using it
        # add the [CLS] token to the embedded patch tokens
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # change the embedding length if it is longer than the default length, use the interpolation function defined earlier
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings
    

class DWConv(nn.Module):
    def __init__(self, dim = Config.hidden_size):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H = Config.image_size[0], W = Config.image_size[1]):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class FocusedLinearAttention(nn.Module):
    def __init__(self, dim=Config.hidden_size, num_patches=Config.patch_size, num_heads=Config.num_heads, qkv_bias=Config.qkv_bias, qk_scale=None, attn_drop=0., proj_drop=0.,
                 linear=False, focusing_factor=3, kernel_size=5, train_positional_encoding=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)
        self.dropout = nn.Dropout(attn_drop, inplace=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.train_positional_encoding = train_positional_encoding

        self.linear = linear

        self.pool = nn.AdaptiveAvgPool2d(7)
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches, dim)))
        self.vit_embedding = ViTEmbeddings()    
        # print('Linear Attention f{} kernel{}'.format(focusing_factor, kernel_size))

    def forward(self, x):
        b, n, c = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        h = int(math.sqrt(n))
        w = int(math.sqrt(n))

        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        
        # Whether to train the positional encoding or not
        if self.train_positional_encoding:
            k = k + self.positional_encoding
        else:
            k = k + self.vit_embedding.position_embeddings
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block. This is just the output of the block.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(Config.hidden_size, Config.hidden_size)
        self.dropout = nn.Dropout(0.1, inplace=False)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = FocusedLinearAttention()
        self.output = ViTSelfOutput()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self_outputs = self.attention(hidden_states)
        attention_output = self.output(self_outputs, hidden_states)
        
        return attention_output


# The following block will upsample the model dimensions from 768 to 3072
class ViTIntermediate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(Config.hidden_size, Config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

# The following block will downsample the model dimensions from 3072 to 768
class ViTOutput(nn.Module):
    def __init__(self) -> None:
        """
        This is actually the resnet block 
        """
        super().__init__()
        self.dense = nn.Linear(Config.intermediate_size, Config.hidden_size)
        self.dropout = nn.Dropout(0., inplace=True)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # residual connection
        hidden_states = hidden_states + input_tensor

        return hidden_states


class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self) -> None:
        super().__init__()
        self.attention = FocusedLinearAttention()
        self.intermediate = ViTIntermediate()
        self.output = ViTOutput()
        self.layernorm_before = nn.LayerNorm(Config.hidden_size, eps=Config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(Config.hidden_size, eps=Config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states)  # in ViT, layernorm is applied before self-attention
        )
        attention_output = self_attention_outputs
        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = layer_output

        return outputs


class ViTEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([ViTLayer() for _ in range(Config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self,hidden_states: torch.Tensor) -> torch.Tensor:
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs

            return hidden_states 


class ViTCustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embeddings = ViTEmbeddings()
        self.encoder = ViTEncoder()

        self.layernorm = nn.LayerNorm(Config.hidden_size, eps=Config.layer_norm_eps)

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        A technique to prune heads with less informations than others in the self-attention modules
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel, this probably would not be used in our case
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, pixel_values: Optional[torch.Tensor] = None, interpolate_pos_encoding: Optional[bool] = None) -> torch.Tensor:
        if pixel_values is None:
            raise ValueError("You need to input an image!")
        # change the dtype of the pixel_values to the same dtype as the weights of the embedding layer,
        # in case the dtype of the pixel_values is different from the dtype of the weights
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output
        )

        sequence_output = encoder_outputs
        sequence_output = self.layernorm(sequence_output)

        return sequence_output  

from transformers import ViTModel
pretrained_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTCustomModel()

# Prepare to load weights
pretrained_dict = pretrained_model.state_dict()
custom_dict = model.state_dict()

# Filter out unnecessary keys and update custom model dict
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in custom_dict and 'attention' in k}
custom_dict.update(pretrained_dict)

# Load the updated state dict into custom model
model.load_state_dict(custom_dict, strict=False)

# Save the model
# model.save_pretrained("model_weights/my-vit-bert")

print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
model.to(device)

# A simple debuggin test
test_image = torch.randn(32, 1, 224, 224)
test_image = test_image.to(device)
with torch.no_grad():
    model_output = model(test_image)
print(model_output.shape)