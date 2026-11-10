"""Action Chunking with Transformers (ACT), Zhao et al. 2023, re-implemented compactly.

Inputs: 3 RGB cameras (84x84) + 50-dim proprioception. Output: a chunk of K
future raw actions (normalised by dataset statistics). Training uses a CVAE
over the action chunk; inference uses the prior mean (z = 0).
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ACTConfig:
    state_dim: int = 50
    action_dim: int = 20
    num_cameras: int = 3
    chunk: int = 50
    d_model: int = 256
    nheads: int = 8
    ff_dim: int = 2048
    enc_layers: int = 4
    dec_layers: int = 4
    vae_layers: int = 4
    latent_dim: int = 32
    dropout: float = 0.1
    pretrained_backbone: bool = True

    def to_dict(self):
        return asdict(self)


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm with fixed statistics (as in DETR/ACT), safe for small batches."""

    def __init__(self, n):
        super().__init__()
        for name, init in (("weight", 1.0), ("bias", 0.0), ("running_mean", 0.0)):
            self.register_buffer(name, torch.full((n,), init))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        scale = self.weight * (self.running_var + 1e-5).rsqrt()
        bias = self.bias - self.running_mean * scale
        return x * scale[None, :, None, None] + bias[None, :, None, None]


def _resnet18(pretrained: bool) -> nn.Module:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
    return nn.Sequential(*list(net.children())[:-2])  # (B, 512, h, w)


def _transformer_encoder(cfg, layers):
    layer = nn.TransformerEncoderLayer(
        cfg.d_model, cfg.nheads, cfg.ff_dim, cfg.dropout, batch_first=True, norm_first=True
    )
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class ACT(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.backbone = _resnet18(cfg.pretrained_backbone)
        self.img_proj = nn.Conv2d(512, d, 1)
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        # 84x84 input -> 3x3 feature map per camera.
        self.img_tokens = 9
        self.img_pos = nn.Parameter(torch.randn(1, cfg.num_cameras * self.img_tokens, d) * 0.02)
        self.cam_embed = nn.Parameter(torch.randn(1, cfg.num_cameras, 1, d) * 0.02)

        # CVAE encoder: [cls, state, a_1..a_K] -> (mu, logvar)
        self.vae_cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.vae_state = nn.Linear(cfg.state_dim, d)
        self.vae_action = nn.Linear(cfg.action_dim, d)
        self.vae_pos = nn.Parameter(torch.randn(1, cfg.chunk + 2, d) * 0.02)
        self.vae_encoder = _transformer_encoder(cfg, cfg.vae_layers)
        self.vae_out = nn.Linear(d, 2 * cfg.latent_dim)

        # Main encoder over [latent, state, image tokens]; decoder with K queries.
        self.latent_proj = nn.Linear(cfg.latent_dim, d)
        self.state_proj = nn.Linear(cfg.state_dim, d)
        self.extra_pos = nn.Parameter(torch.randn(1, 2, d) * 0.02)
        self.encoder = _transformer_encoder(cfg, cfg.enc_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d, cfg.nheads, cfg.ff_dim, cfg.dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.dec_layers)
        self.queries = nn.Parameter(torch.randn(1, cfg.chunk, d) * 0.02)
        self.head = nn.Linear(d, cfg.action_dim)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def encode_images(self, rgb):
        """rgb: (B, C, 3, H, W) float in [0, 1] -> (B, C*9, d)."""
        b, c = rgb.shape[:2]
        x = rgb.flatten(0, 1)
        x = (x - self.img_mean) / self.img_std
        feat = self.img_proj(self.backbone(x))  # (B*C, d, 3, 3)
        if feat.shape[-2:] != (3, 3):
            feat = F.adaptive_avg_pool2d(feat, 3)
        feat = feat.flatten(2).transpose(1, 2).reshape(b, c, self.img_tokens, -1)
        feat = feat + self.cam_embed
        return feat.flatten(1, 2) + self.img_pos

    def forward(self, rgb, state, actions=None):
        """Returns (pred_actions (B,K,A), mu, logvar). actions given => training."""
        b = state.shape[0]
        if actions is not None:
            tokens = torch.cat(
                [
                    self.vae_cls.expand(b, -1, -1),
                    self.vae_state(state)[:, None],
                    self.vae_action(actions),
                ],
                1,
            )
            h = self.vae_encoder(tokens + self.vae_pos)[:, 0]
            mu, logvar = self.vae_out(h).chunk(2, -1)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            mu = logvar = None
            z = torch.zeros(b, self.cfg.latent_dim, device=state.device, dtype=state.dtype)

        extra = torch.stack([self.latent_proj(z), self.state_proj(state)], 1) + self.extra_pos
        memory = self.encoder(torch.cat([extra, self.encode_images(rgb)], 1))
        out = self.decoder(self.queries.expand(b, -1, -1), memory)
        return self.head(out), mu, logvar
