from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from ...utils import BaseModule
from .vit import ViTModel, dino_vit_b16


class DINOSingleImageTokenizer(BaseModule):
    """TripoSR DINO ViT-B/16 image tokenizer using a local ViT implementation.

    The official TripoSR reference downloads the DINO ViT config from the
    Hugging Face Hub, but the tokenizer weights are stored in the TripoSR
    checkpoint (model.ckpt).  This implementation uses a local ViT-B/16 with
    the original parameter names so the checkpoint loads correctly without
    any Hub downloads.
    """

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "facebook/dino-vitb16"
        enable_gradient_checkpointing: bool = False

    cfg: Config

    def configure(self) -> None:
        # Use the local ViT-B/16 implementation with the original parameter
        # names.  The weights are loaded from the TripoSR checkpoint via
        # system.py, so no download from the Hub is needed.
        self.model: ViTModel = dino_vit_b16()

        if self.cfg.enable_gradient_checkpointing:
            for layer in self.model.encoder.layer:
                layer.requires_grad_(True)

        self.register_buffer(
            "image_mean",
            torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        packed = False
        if images.ndim == 4:
            packed = True
            images = images.unsqueeze(1)

        batch_size, n_input_views = images.shape[:2]
        images = (images - self.image_mean) / self.image_std
        out = self.model(
            rearrange(images, "B N C H W -> (B N) C H W"),
            interpolate_pos_encoding=True,
        )
        local_features, global_features = out.last_hidden_state, out.pooler_output
        local_features = local_features.permute(0, 2, 1)
        local_features = rearrange(
            local_features, "(B N) Ct Nt -> B N Ct Nt", B=batch_size
        )
        if packed:
            local_features = local_features.squeeze(1)

        return local_features

    def detokenize(self, *args, **kwargs):
        raise NotImplementedError