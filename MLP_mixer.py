# original code by Rishikesh https://github.com/rishikksh20
# code review comments by younghan, deokjoong

import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    """
    MLP structure
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        """
        :param dim:        linear input dimension
        :param hidden_dim: linear hidden dimension
        :param dropout:    ratio of dropout
        """
        # token-mixer (dim=196, hidden_dim=256)
        # channel-mixer (dim=512, hidden_dim=2048)
        super().__init__()
        # input (B, ch/patch, dim)
        self.net = nn.Sequential(
            # (B, ch/patch, dim) -> (B, ch/patch, hidden_dim)
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # (B, ch/patch, hidden_dim) -> (B, ch/patch, dim)
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
            # output (B, ch/patch, dim)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    """
    Mixer layer : token-mixer MLP, channel-mixer MLP
    """
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        """
        :param dim:         patch embedding dimension
        :param num_patch:   # of patches
        :param token_dim:   token-mixing MLP hidden dimension
        :param channel_dim: channel-mixing MLP hidden dimension
        :param dropout:     ratio of dropout in Feedforward newtork
        """
        # (dim=512, num_patch=196, token_dim=256, channel_dim=2048)
        super().__init__()

        # Token-mixing MLP : "mixing" spatial information
        # input (B, 196, 512) -> output (B, 196, 512)
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            # (B, 196, 512) -> (B, 512, 196)
            Rearrange('b n d -> b d n'),
            # num_patch=196, token_dim=256
            # (B, 512, 196) -> (B, 512, 256) -> (B, 512, 196)
            FeedForward(num_patch, token_dim, dropout),
            # (B, 512, 196) -> (B, 196, 512)
            Rearrange('b d n -> b n d')
        )

        # Channel-mixing MLP : "mixing" the per-location features
        # input (B, 196, 512) -> output (B, 196, 512)
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            # (B, 196, 512) -> (B, 196, 2048) -> (B, 196, 512)
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        # skip connection with token mixing
        x = x + self.token_mix(x)
        # skip connection with channel mixing
        x = x + self.channel_mix(x)

        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        """
        :param in_channels: RGB channel
        :param dim:         patch embedding dimension
        :param num_classes: # of classes
        :param patch_size:  patch size in image
        :param image_size:  input image size
        :param depth:       # of MLP Mixer blocks
        :param token_dim:   token-mixing MLP hidden dimension
        :param channel_dim: channel-mixing MLP hidden dimension
        """
        # in_channels=3, dim=512, num_classses=1000, patch_size=16,
        # image_size=224, depth=8, token_dim=256, channel_dim=2048
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size//patch_size) ** 2  # 14*14=196 patch

        # image to patch embedding (B, 3, 224, 224) -> (B, 512, 14, 14) -> (B, 196, 512)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),  # filter size=stride : non overlapping
            # (B, 3, 224, 224) -> (B, 512, 14, 14)
            Rearrange('b c h w -> b (h w) c'),  # Rearrange 설명
            # (B, 512, 14, 14) -> (B, 196, 512)

        )
        # Holds Mixer modules in a list.
        self.mixer_blocks = nn.ModuleList([])

        # 8 depth : channel, token mixer block
        for _ in range(depth):  # (dim=512, num_patch=196, token_dim=256, ch_dim=2048)
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        # classifier
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # conv (B, 3, 224, 224) -> (B, 196, 512)
        x = self.to_patch_embedding(x)

        # (B, 196, 512) -> (B, 196, 512)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        # GAP, make 196 patch to one : (B, 196, 512) -> (B, 512)
        x = x.mean(dim=1)

        return self.mlp_head(x)




if __name__ == "__main__":
    img = torch.ones([1, 3, 224, 224])

    model = MLPMixer(in_channels=3, image_size=224, patch_size=16, num_classes=1000,
                     dim=512, depth=8, token_dim=256, channel_dim=2048)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]

