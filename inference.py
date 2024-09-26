from dataclasses import dataclass
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from skimage.io import imsave
from torch import Tensor, nn
from transformers import ViTConfig, ViTModel
from transformers.utils import ModelOutput


def min_max_standardize(im, min_perc=0, max_perc=100):
    min_val = np.percentile(im, min_perc, axis=(1, 2, 3), keepdims=True)
    max_val = np.percentile(im, max_perc, axis=(1, 2, 3), keepdims=True)

    im = (im - min_val) / (max_val - min_val + 1e-6)
    return im.astype(np.float32) 


class GatedAttentionPooler(nn.Module):
    def __init__(
        self, dim: int, int_dim: int = 512, num_heads: int = 1, out_dim: int = None
    ):
        super().__init__()

        self.num_heads = num_heads

        self.attention_v = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(dim, int_dim), nn.Tanh()
        )
        self.attention_u = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(dim, int_dim), nn.GELU()
        )
        self.attention = nn.Linear(int_dim, num_heads)

        self.softmax = nn.Softmax(dim=-1)

        if out_dim is None:
            self.out_dim = dim * num_heads
            self.out_proj = nn.Identity()
        else:
            self.out_dim = out_dim
            self.out_proj = nn.Linear(dim * num_heads, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        v = self.attention_v(x)
        u = self.attention_u(x)

        attn = self.attention(v * u).permute(0, 2, 1)
        attn = self.softmax(attn)

        x = torch.bmm(attn, x)
        x = x.view(x.shape[0], -1)

        x = self.out_proj(x)
        return x, attn


@dataclass
class ViTPoolModelOutput(ModelOutput):
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Tuple[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None

    pool_op: torch.FloatTensor = None
    pool_attn: torch.FloatTensor = None

    probabilties: torch.FloatTensor = None


class ViTPoolClassifier(nn.Module):
    def __init__(self, config: Dict):
        super(ViTPoolClassifier, self).__init__()

        self.vit_config = ViTConfig(**config["vit_model"])

        self.encoder = ViTModel(self.vit_config, add_pooling_layer=False)

        pool_config = config.get("pool_model")
        self.pool_model = GatedAttentionPooler(**pool_config) if pool_config else None

        self.classifier = nn.Sequential(
            nn.Linear(self.pool_model.out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, config["num_classes"]),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        outputs = self.encoder(x, output_attentions=True, interpolate_pos_encoding=True)

        if self.pool_model:
            pool_op, pool_attn = self.pool_model(outputs.last_hidden_state)
        else:
            pool_op = torch.mean(outputs.last_hidden_state, dim=1)
            pool_attn = None

        probs = self.sigmoid(self.classifier(pool_op))

        h_feat = h // self.vit_config.patch_size
        w_feat = w // self.vit_config.patch_size

        attentions = outputs.attentions[-1][:, :, 0, 1:].reshape(
            b, self.vit_config.num_attention_heads, h_feat, w_feat
        )

        pool_attn = pool_attn[:, :, 1:].reshape(
            b, self.pool_model.num_heads, h_feat, w_feat
        )

        return ViTPoolModelOutput(
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=attentions,
            pool_op=pool_op,
            pool_attn=pool_attn,
            probabilties=probs,
        )

    def load_model_dict(
        self,
        encoder_path: str,
        classifier_path: str,
        device="cpu",
    ):
        checkpoint = torch.load(encoder_path, map_location=device)
        encoder_ckpt = {
            k[len("encoder.") :]: v for k, v in checkpoint.items() if "encoder." in k
        }

        status = self.encoder.load_state_dict(encoder_ckpt)
        print(f"Encoder status: {status}")

        pool_ckpt = {
            k.replace("pool_model.", ""): v
            for k, v in checkpoint.items()
            if "pool_model." in k
        }
        if pool_ckpt:
            status = self.pool_model.load_state_dict(pool_ckpt)
            print(f"Pool model status: {status}")

        classifier_ckpt = torch.load(classifier_path, map_location=device)
        status = self.classifier.load_state_dict(classifier_ckpt)
        print(f"Classifier status: {status}")


def run_model(model, cell_crop, output_path):
    cell_crop = np.stack(cell_crop, axis=1)
    cell_crop = min_max_standardize(cell_crop)
    cell_crop = torch.tensor(cell_crop).float()
    output = model(cell_crop)
    probabilities = output.probabilties.detach()[0]
    np.save(output_path + "_embedding.npy", output.pool_op.detach()[0])
    np.save(output_path + "_probabilities.npy", probabilities)
    imsave(output_path + "_attention_map.png", np.int8(output.pool_attn.detach()[0] * 255))
