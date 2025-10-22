from dataclasses import dataclass, field
import torch
import torch.nn as nn

from config import *

# `TimeEmbeddingBlock`
# Takes a 1D tensor of timesteps of size batch_size (B,)
# and gives us a time embedding of (B * t_emb_dim)

def get_time_embedding(time_steps, t_emb_dim):

    # so basically, $\sin(pos / 10000^{2i/d_{\text{model}}})$ AND $\cos(pos / 10000^{2i/d_{\text{model}}})$
    
    factor = 10000 ** ((torch.arange(
        start=0, end=t_emb_dim//2, device=time_steps.device) / (t_emb_dim // 2)
    ))
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    t_emb = torch.cat(tensors=[torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample
        
        # `nn.Sequential(a, b c)` is in place of `x = c(b(a(x)))`
        self.resnet_conv_first = nn.Sequential(
            # Normalization that splits channels into groups and normalizes within each group.
            nn.GroupNorm(num_groups=8, num_channels=in_channels),

            # [how i feel not having to implement that swiglu bullshit](https://tenor.com/view/ishowspeed-try-not-to-laugh-gif-7682731162751353849)
            nn.SiLU(),
            
            # applies a learnable filter that slides across an image to extract features
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )

        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # residual ensures input of entire resnet block can be added to output of last conv layer
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4,
                                         stride=2, padding=1) if self.down_sample else nn.Identity() # $I$ -> no-op.
    
    def forward(self, x, t_emb):
        
        out = x
        
        # Resnet block
        resnet_input = out
        out = self.resnet_conv_first(out)
        
        # t_emb_layers(t_emb) outputs a tensor with shape (batch_size, channels)
        # We need to add it to out, which has shape (batch_size, channels, height, width)
        # The slicing operation adds two new dimensions at the end,
        # transforming the shape from (batch_size, channels) to (batch_size, channels, 1, 1)
        # None creates new axis with size 1
        out = out + self.t_emb_layers(t_emb)[:, :, None, None]

        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)
        
        # Attention block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h*w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn
        
        out = self.down_sample_conv(out)
        return out

# Will have same kid of layers as `DownBlock`
# but we need 2 kinds of instances of layers that belong
# to the resnet block
class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads):
        super().__init__()
        
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])
        
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
        ])
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])

        self.attention_norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        ])

    # the forward method will have 1 difference, that is
    # that is we call the first resnet block
    # and then self-attn and second resnet

    # btw self-attention is when a sequence attends to itself. 
    # we lowk callin self.attention(in_attn, in_attn, in_attn)
    # where kq&v are all in_attn
    # btw if we impl.'ed multiple layers the self attention and following resnet block would
    # have a loop!
    def forward(self, x, t_emb):
        out = x
        
        # first resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        # attention block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h*w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn

        # second resnet block
        resnet_input = out
        out = self.resnet_conv_first[1](out)
        out = out + self.t_emb_layers[1](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[1](out)
        out = out + self.residual_input_conv[1](resnet_input)
        
        return out

# upblock is exact same as downlbock but instead of downsampling we upsample
class UpBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads):
        super().__init__()
        
        self.up_sample = up_sample
        
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self.attention_norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # We'll use conv transpose to do the up sampling for us
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4,
                                                        stride=2, padding=1) if self.up_sample else nn.Identity()

    def forward(self, x, out_down, t_emb):
        # this is new!!!
        # we could also concat -> resnet + self attn -> then upsample
        # but we dont lol
        x = self.up_sample_conv(x)

        # If x has shape        (batch, channels_x, h, w) and 
        # out_down has shape    (batch, channels_down, h, w), 
        # the result has shape  (batch, channels_x + channels_down, h, w).
        x = torch.cat([x, out_down], dim=1)

        # Resnet block
        out = x
        resnet_input = out
        out = self.resnet_conv_first(out)
        out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)
        
        # Attention Block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn
        
        return out

class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """
    def __init__(self, config: UnetConfig = UnetConfig()):
        super().__init__()
        im_channels = config.im_channels
        self.down_channels = config.down_channels
        self.mid_channels = config.mid_channels
        self.t_emb_dim = config.time_emb_dim
        self.down_sample = config.down_sample
        #self.num_down_layers = config.num_down_layers
        #self.num_mid_layers = config.num_mid_layers
        #self.num_up_layers = config.num_up_layers
        self.num_heads = config.num_heads
 
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim,
                                        down_sample=self.down_sample[i], num_heads=self.num_heads))
        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim,
                                      num_heads=self.num_heads))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpBlock(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16,
                                    self.t_emb_dim, up_sample=self.down_sample[i], num_heads=self.num_heads))
        
        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
            # Shapes assuming downblocks are [C1, C2, C3, C4]
            
            # Shapes assuming midblocks are [C4, C4, C3]
            
            # Shapes assuming downsamples are [True, True, False]

            # $B \times C \times H \times W$
            out = self.conv_in(x)
            # $B \times C_1 \times H \times W$
            
            # $t_{emb} \rightarrow B \times t_{emb\_dim}$
            t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
            t_emb = self.t_proj(t_emb)
            
            down_outs = []
            
            for _idx, down in enumerate(self.downs):
                down_outs.append(out)
                out = down(out, t_emb)
        
            # down_outs: 

            # $$[B \times C_1 \times H \times W, \quad B \times C_2 \times \dfrac{H}{2} \times \dfrac{W}{2}, \quad B \times C_3 \times \dfrac{H}{4} \times \dfrac{W}{4}]$$
            
            # out: $B \times C_4 \times \dfrac{H}{4} \times \dfrac{W}{4}$
                
            for mid in self.mids:
                out = mid(out, t_emb)
        
            # out: $B \times C_3 \times \dfrac{H}{4} \times \dfrac{W}{4}$
            
            for up in self.ups:
                down_out = down_outs.pop()
                out = up(out, down_out, t_emb)
        
            # out: $$[B \times C_2 \times \dfrac{H}{4} \times \dfrac{W}{4}, \quad B \times C_1 \times \dfrac{H}{2} \times \dfrac{W}{2}, \quad B \times 16 \times H \times W]$$

            out = self.norm_out(out)
            out = nn.SiLU()(out)
            out = self.conv_out(out)
            
            # out: $B \times C \times H \times W$
            return out