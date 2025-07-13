# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from fairseq2.config_registry import ConfigRegistry
from fairseq2.logging import get_log_writer
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.typing import DataType, Device

from lcm.datasets.batch import EmbeddingsBatch
from lcm.models.abstract_lcm import (
    AbstractLCModel,
    AbstractLCModelBuilder,
    AbstractLCModelConfig,
)
from lcm.models.base_lcm.frontend import LCMFrontend, LCMFrontendConfig
from lcm.nn.projection import Projection, ProjectionConfig
from lcm.nn.transformer import (
    LCMTransformerDecoder,
    TransformerConfig,
    TransformerFactory,
)

logger = get_log_writer(__name__)

THEMATIC_LCM_MODEL_TYPE = "thematic_lcm"


@dataclass
class ThematicLCModelConfig(AbstractLCModelConfig):
    model_type: str = THEMATIC_LCM_MODEL_TYPE

    max_seq_len: int = 2048
    """Maximum sequence length for paragraph processing."""

    model_dim: int = 1024
    """Internal model dimension."""

    num_themes: Optional[int] = None
    """Number of themes for classification. If None, outputs continuous theme embeddings."""

    theme_embed_dim: int = 512
    """Dimension of theme embeddings."""

    frontend: LCMFrontendConfig = field(default_factory=lambda: LCMFrontendConfig())
    """Frontend config mapping from sentence embeddings to model_dim."""

    encoder: TransformerConfig = field(default_factory=lambda: TransformerConfig())
    """Transformer encoder for paragraph-level processing."""

    theme_projection: ProjectionConfig = field(
        default_factory=lambda: ProjectionConfig()
    )
    """Theme projection config mapping to theme space."""


thematic_lcm_archs = ConfigRegistry[ThematicLCModelConfig]()
thematic_lcm_arch = thematic_lcm_archs.decorator


class ThematicLCModel(AbstractLCModel):
    """Thematic LCM model for paragraph-level theme prediction"""

    config: ThematicLCModelConfig

    def __init__(
        self,
        config: ThematicLCModelConfig,
        frontend: LCMFrontend,
        encoder: LCMTransformerDecoder,
        theme_projection: Projection,
        theme_classifier: Optional[nn.Linear] = None,
    ) -> None:
        """
        Thematic LCM model with:
            - frontend: maps sentence embeddings to model_dim
            - encoder: transformer encoder for paragraph processing
            - theme_projection: maps to theme embedding space
            - theme_classifier: optional classification head
        """
        super().__init__(config)

        self.frontend = frontend
        self.encoder = encoder
        self.theme_projection = theme_projection
        self.theme_classifier = theme_classifier

        self.model_dim = encoder.model_dim
        self.theme_embed_dim = config.theme_embed_dim
        self.num_themes = config.num_themes

    def forward(
        self,
        batch: EmbeddingsBatch,
        state_bag: Optional[IncrementalStateBag] = None,
        **kwargs,
    ) -> EmbeddingsBatch:
        """
        Forward pass for thematic modeling.

        Args:
            batch: Input batch containing sentence embeddings for paragraphs

        Returns:
            EmbeddingsBatch containing theme representations
        """
        # Frontend processing: sentence embeddings -> model_dim
        seqs, padding_mask = self.frontend(
            batch.seqs,
            batch.padding_mask,
            state_bag=state_bag,
            **kwargs,
        )

        # Encoder: paragraph-level processing
        encoded_seqs, encoded_padding_mask = self.encoder(
            seqs,
            padding_mask,
            state_bag=state_bag,
            **kwargs,
        )

        # Aggregate paragraph representation (mean pooling over non-padded positions)
        if encoded_padding_mask is not None:
            # Create mask for valid positions
            mask = ~encoded_padding_mask
            # Mean pooling with masking
            paragraph_repr = (encoded_seqs * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
                dim=1, keepdim=True
            )
        else:
            # Simple mean pooling
            paragraph_repr = encoded_seqs.mean(dim=1)

        # Project to theme space
        theme_embeddings = self.theme_projection(paragraph_repr)

        # Return theme embeddings in batch format
        return EmbeddingsBatch(
            seqs=theme_embeddings.unsqueeze(1),  # Add sequence dimension
            padding_mask=None,
        )

    def predict_themes(
        self,
        batch: EmbeddingsBatch,
        return_probabilities: bool = False,
    ) -> torch.Tensor:
        """
        Predict themes for paragraphs.

        Args:
            batch: Input batch containing sentence embeddings
            return_probabilities: If True and classifier exists, return class probabilities

        Returns:
            Theme predictions (embeddings or class indices/probabilities)
        """
        with torch.no_grad():
            theme_batch = self.forward(batch)
            theme_embeddings = theme_batch.seqs.squeeze(1)  # Remove sequence dimension

            if self.theme_classifier is not None:
                # Classification mode
                logits = self.theme_classifier(theme_embeddings)
                if return_probabilities:
                    return torch.softmax(logits, dim=-1)
                else:
                    return torch.argmax(logits, dim=-1)
            else:
                # Embedding mode
                return theme_embeddings


class ThematicLCModelBuilder(AbstractLCModelBuilder):
    """Builds modules of a Thematic LCM"""

    config: ThematicLCModelConfig

    def __init__(
        self,
        config: ThematicLCModelConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(config, device=device, dtype=dtype)

    def build_frontend(self) -> LCMFrontend:
        """Build the frontend."""
        return LCMFrontend(
            self.config.frontend,
            model_dim=self.config.model_dim,
            sonar_embed_dim=self.config.sonar_embed_dim,
            max_seq_len=self.config.max_seq_len,
            sonar_normalizer=self.build_sonar_normalizer(),
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder(self) -> LCMTransformerDecoder:
        """Build the transformer encoder."""
        transformer_factory = TransformerFactory(
            self.config.encoder,
            device=self.device,
            dtype=self.dtype,
        )
        return transformer_factory.build_lcm_decoder()

    def build_theme_projection(self) -> Projection:
        """Build the theme projection layer."""
        return Projection(
            self.config.theme_projection,
            input_dim=self.config.model_dim,
            output_dim=self.config.theme_embed_dim,
            device=self.device,
            dtype=self.dtype,
        )

    def build_theme_classifier(self) -> Optional[nn.Linear]:
        """Build optional theme classifier."""
        if self.config.num_themes is not None:
            classifier = nn.Linear(
                self.config.theme_embed_dim,
                self.config.num_themes,
            )
            if self.device is not None:
                classifier = classifier.to(device=self.device)
            if self.dtype is not None:
                classifier = classifier.to(dtype=self.dtype)
            return classifier
        return None

    def build_model(self) -> ThematicLCModel:
        """Build a Thematic LCM model."""
        frontend = self.build_frontend()
        encoder = self.build_encoder()
        theme_projection = self.build_theme_projection()
        theme_classifier = self.build_theme_classifier()

        return ThematicLCModel(
            self.config,
            frontend=frontend,
            encoder=encoder,
            theme_projection=theme_projection,
            theme_classifier=theme_classifier,
        )


def create_thematic_lcm_model(
    config: ThematicLCModelConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> ThematicLCModel:
    """Create a Thematic LCM model."""
    builder = ThematicLCModelBuilder(config, device=device, dtype=dtype)
    return builder.build_model()
