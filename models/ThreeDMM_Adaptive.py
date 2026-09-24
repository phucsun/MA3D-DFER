import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreeDMMEncoder(nn.Module):

    def __init__(self, embed_dim=256, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(334, 512),
            nn.ReLU(True),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(dropout),

            nn.Linear(512, embed_dim)
        )

    def forward(self, x_3d):
        return self.net(x_3d)

class ThreeDMMConditionGenerator(nn.Module):

    def __init__(self, embed_dim, num_blocks, channels):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_blocks * channels * 2)
        )

    def forward(self, embed):
        B = embed.shape[0]

        gamma_beta = self.net(embed)
        gamma_beta = gamma_beta.view(
            B, self.num_blocks, 2, self.channels
        )

        gamma = gamma_beta[:, :, 0]
        beta = gamma_beta[:, :, 1]

        return gamma, beta

class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.net(x)
        return x * scale

class ThreeDMMAdaptiveBlock(nn.Module):

    def __init__(self, channels, bottleneck_ratio=4, use_se=False, dropout=0.1):
        super().__init__()
        self.use_se = use_se
        hidden = channels // bottleneck_ratio

        # Bottleneck convs
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, channels, 1)

        self.norm1 = nn.InstanceNorm2d(channels, affine=False)
        self.norm2 = nn.InstanceNorm2d(hidden, affine=False)

        self.dropout = nn.Dropout2d(dropout)

        if use_se:
            self.se = SEBlock(channels)

    def forward(self, x, gamma, beta):

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # FiLM modulation
        out = self.norm1(x)
        out = out * (1 + gamma) + beta
        out = F.leaky_relu(out, 0.2)

        # Bottleneck
        out = self.conv1(out)

        out = self.norm2(out)
        out = F.leaky_relu(out, 0.2)

        out = self.conv2(out)
        out = self.dropout(out)

        out = F.leaky_relu(out, 0.2)
        out = self.conv3(out)

        if self.use_se:
            out = self.se(out)

        return x + out



class ThreeDMMFusion(nn.Module):

    def __init__(self, feat_nc=512, embed_dim=256, num_blocks=5):

        super().__init__()

        self.num_blocks = num_blocks

        self.encoder_3dmm = ThreeDMMEncoder(embed_dim)

        self.film_generator = ThreeDMMConditionGenerator(
            embed_dim,
            num_blocks,
            feat_nc
        )

        self.blocks = nn.ModuleList([
            ThreeDMMAdaptiveBlock(feat_nc, use_se=True),
            ThreeDMMAdaptiveBlock(feat_nc, use_se=True),
            ThreeDMMAdaptiveBlock(feat_nc, use_se=True),
            ThreeDMMAdaptiveBlock(feat_nc, use_se=True),
            ThreeDMMAdaptiveBlock(feat_nc, use_se=True)
        ])

    def forward(self, x, x_3d):
        mm_embed = self.encoder_3dmm(x_3d)
        gamma, beta = self.film_generator(mm_embed)

        for i, block in enumerate(self.blocks):
            x = block(x, gamma[:, i], beta[:, i])

        return x