from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from skimage.io import imsave
from torch import Tensor, nn
from transformers import ViTConfig, ViTModel
from transformers.utils import ModelOutput


CLASS2NAME = {
    0: "Actin filaments",
    1: "Aggresome",
    2: "Cell Junctions",
    3: "Centriolar satellite",
    4: "Centrosome",
    5: "Cytokinetic bridge",
    6: "Cytoplasmic bodies",
    7: "Cytosol",
    8: "Endoplasmic reticulum",
    9: "Endosomes",
    10: "Focal adhesion sites",
    11: "Golgi apparatus",
    12: "Intermediate filaments",
    13: "Lipid droplets",
    14: "Lysosomes",
    15: "Microtubules",
    16: "Midbody",
    17: "Mitochondria",
    18: "Mitotic chromosome",
    19: "Mitotic spindle",
    20: "Nuclear bodies",
    21: "Nuclear membrane",
    22: "Nuclear speckles",
    23: "Nucleoli",
    24: "Nucleoli fibrillar center",
    25: "Nucleoli rim",
    26: "Nucleoplasm",
    27: "Peroxisomes",
    28: "Plasma membrane",
    29: "Vesicles",
    30: "Negative",
}

CLASS2COLOR = {
    0: '#ffeb3b',
    1: '#76ff03',
    2: '#ff6d00',
    3: '#eb30c1',
    4: '#faadd4',
    5: '#795548',
    6: '#64ffda',
    7: '#00e676',
    8: '#03a9f4',
    9: '#4caf50',
    10: '#ffc107',
    11: '#00bcd4',
    12: '#cddc39',
    13: '#212121',
    14: '#8bc34a',
    15: '#ff9800',
    16: '#ae8c08',
    17: '#ffff00',
    18: '#31b61f',
    19: '#9e9e9e',
    20: '#2196f3',
    21: '#e91e63',
    22: '#3f51b5',
    23: '#9c27b0',
    24: '#673ab7',
    25: '#d3a50b',
    26: "#f44336",
    27: "#009688",
    28: "#ff9e80",
    29: "#242e4b",
    30: "#000000",
}


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
    embedding = output.pool_op.detach()[0]
    np.save(output_path + "_embedding.npy", embedding)
    np.save(output_path + "_probabilities.npy", probabilities)
    imsave(output_path + "_attention_map.png", np.int8(output.pool_attn.detach()[0] * 255))

    return np.array(embedding), np.array(probabilities)
