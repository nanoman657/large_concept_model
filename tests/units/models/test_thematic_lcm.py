# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

"""
Testing thematic LCM model functionality
"""

import pytest
import torch

from lcm.datasets.batch import EmbeddingsBatch
from lcm.models.thematic_lcm import (
    ThematicLCModel,
    ThematicLCModelConfig,
    create_thematic_lcm_model,
)
from lcm.models.thematic_lcm.builder import THEMATIC_LCM_MODEL_TYPE
from lcm.nn.transformer import TransformerConfig
from lcm.utils.model_type_registry import lcm_model_type_registry


def test_thematic_lcm_creation():
    """Test that we can create a thematic LCM model"""
    config = ThematicLCModelConfig(
        encoder=TransformerConfig(num_layers=2),
        theme_embed_dim=256,
    )
    model = create_thematic_lcm_model(config)
    assert isinstance(model, ThematicLCModel)
    assert model.theme_embed_dim == 256


@pytest.mark.parametrize(
    "model_cfg_kwargs",
    [
        {},  # default
        {"theme_embed_dim": 512},
        {"num_themes": 10},  # with classifier
        {"model_dim": 512, "theme_embed_dim": 256},
    ],
)
def test_thematic_lcm_forward(model_cfg_kwargs):
    """Testing that we can create a thematic LCM model and do a forward step"""
    # Creating a toy model
    model_cfg = ThematicLCModelConfig(
        encoder=TransformerConfig(num_layers=2), **model_cfg_kwargs
    )
    model = create_thematic_lcm_model(model_cfg)

    # Creating a toy batch representing paragraphs
    batch_size, paragraph_len = 3, 5  # 3 paragraphs, each with 5 sentences
    sonar_dim, sonar_std = 1024, 0.006
    x = torch.randn(size=[batch_size, paragraph_len, sonar_dim]) * sonar_std

    batch = EmbeddingsBatch(x, padding_mask=None)

    # Doing the forward step
    output = model(batch)

    # Testing that the output is adequate
    assert output.seqs.shape[0] == batch_size  # Same batch size
    assert output.seqs.shape[1] == 1  # Single theme per paragraph
    assert (
        output.seqs.shape[2] == model_cfg.theme_embed_dim
    )  # Theme embedding dimension
    assert torch.isfinite(output.seqs).all()


def test_thematic_lcm_theme_prediction():
    """Test theme prediction functionality"""
    # Create model with classifier
    config = ThematicLCModelConfig(
        encoder=TransformerConfig(num_layers=2),
        theme_embed_dim=256,
        num_themes=5,
    )
    model = create_thematic_lcm_model(config)
    model.eval()

    # Create test batch
    batch_size, paragraph_len = 2, 4
    sonar_dim = 1024
    x = torch.randn(size=[batch_size, paragraph_len, sonar_dim]) * 0.006
    batch = EmbeddingsBatch(x, padding_mask=None)

    # Test embedding prediction
    theme_embeddings = model.predict_themes(batch, return_probabilities=False)
    assert theme_embeddings.shape == (batch_size,)  # Class indices
    assert theme_embeddings.dtype == torch.long

    # Test probability prediction
    theme_probs = model.predict_themes(batch, return_probabilities=True)
    assert theme_probs.shape == (batch_size, 5)  # Probabilities for 5 classes
    assert torch.allclose(theme_probs.sum(dim=-1), torch.ones(batch_size))  # Sum to 1


def test_thematic_lcm_without_classifier():
    """Test theme prediction without classifier (embedding mode)"""
    config = ThematicLCModelConfig(
        encoder=TransformerConfig(num_layers=2),
        theme_embed_dim=256,
        num_themes=None,  # No classifier
    )
    model = create_thematic_lcm_model(config)
    model.eval()

    # Create test batch
    batch_size, paragraph_len = 2, 4
    sonar_dim = 1024
    x = torch.randn(size=[batch_size, paragraph_len, sonar_dim]) * 0.006
    batch = EmbeddingsBatch(x, padding_mask=None)

    # Test embedding prediction
    theme_embeddings = model.predict_themes(batch)
    assert theme_embeddings.shape == (batch_size, 256)  # Theme embeddings
    assert torch.isfinite(theme_embeddings).all()


def test_thematic_lcm_with_padding():
    """Test handling of padded sequences"""
    config = ThematicLCModelConfig(
        encoder=TransformerConfig(num_layers=2),
        theme_embed_dim=256,
    )
    model = create_thematic_lcm_model(config)

    # Create batch with padding
    batch_size, max_len = 2, 6
    sonar_dim = 1024

    # Create sequences of different lengths
    x = torch.randn(size=[batch_size, max_len, sonar_dim]) * 0.006
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    padding_mask[0, 4:] = True  # First sequence has length 4
    padding_mask[1, 3:] = True  # Second sequence has length 3

    batch = EmbeddingsBatch(x, padding_mask=padding_mask)

    # Forward pass should handle padding correctly
    output = model(batch)
    assert output.seqs.shape == (batch_size, 1, 256)
    assert torch.isfinite(output.seqs).all()


def test_thematic_lcm_model_registry():
    """Test that the thematic model is properly registered"""
    assert THEMATIC_LCM_MODEL_TYPE in [
        config.model_type for config in lcm_model_type_registry._registry.values()
    ]

    # Test loading toy architecture
    model_type_cfg = lcm_model_type_registry.get_config(THEMATIC_LCM_MODEL_TYPE)
    arch_registry = model_type_cfg.config_loader._arch_configs
    toy_config = arch_registry.get(f"toy_{THEMATIC_LCM_MODEL_TYPE}")
    assert toy_config is not None

    # Test creating model from registry
    toy_model: ThematicLCModel = model_type_cfg.model_factory(toy_config)
    assert isinstance(toy_model, ThematicLCModel)


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (1, 1),  # Single paragraph, single sentence
        (1, 10),  # Single paragraph, multiple sentences
        (4, 5),  # Multiple paragraphs
        (2, 1),  # Multiple paragraphs, single sentence each
    ],
)
def test_thematic_lcm_different_sizes(batch_size, seq_len):
    """Test thematic LCM with different input sizes"""
    config = ThematicLCModelConfig(
        encoder=TransformerConfig(num_layers=2),
        theme_embed_dim=128,
    )
    model = create_thematic_lcm_model(config)
    model.eval()

    sonar_dim = 1024
    x = torch.randn(size=[batch_size, seq_len, sonar_dim]) * 0.006
    batch = EmbeddingsBatch(x, padding_mask=None)

    with torch.no_grad():
        output = model(batch)
        assert output.seqs.shape == (batch_size, 1, 128)
        assert torch.isfinite(output.seqs).all()

        theme_embeddings = model.predict_themes(batch)
        assert theme_embeddings.shape == (batch_size, 128)
        assert torch.isfinite(theme_embeddings).all()
