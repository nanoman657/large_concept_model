# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from lcm.models.thematic_lcm.builder import (
    LCMFrontendConfig,
    ProjectionConfig,
    ThematicLCModelConfig,
    TransformerConfig,
    thematic_lcm_arch,
)


# Every model must register a toy_{model_family}
@thematic_lcm_arch("toy_thematic_lcm")
def toy_thematic_lcm() -> ThematicLCModelConfig:
    return ThematicLCModelConfig(
        encoder=TransformerConfig(num_layers=2),
        theme_embed_dim=256,
    )


@thematic_lcm_arch("thematic_lcm_small")
def thematic_lcm_small() -> ThematicLCModelConfig:
    """Small thematic LCM model for theme prediction
    Suitable for initial experimentation and testing.
    """
    model_dim: int = 512
    num_attn_heads: int = 8
    return ThematicLCModelConfig(
        max_seq_len=1024,
        model_dim=model_dim,
        theme_embed_dim=256,
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        encoder=TransformerConfig(
            final_dropout_p=0.1,
            attention_dropout_p=0.1,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            ffn_inner_dim=model_dim * 4,
            num_attn_heads=num_attn_heads,
            num_layers=6,
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style="rms",
        ),
        theme_projection=ProjectionConfig(),
    )


@thematic_lcm_arch("thematic_lcm_base")
def thematic_lcm_base() -> ThematicLCModelConfig:
    """Base thematic LCM model for theme prediction
    Balanced model for paragraph-level theme understanding.
    """
    model_dim: int = 1024
    num_attn_heads: int = 16
    return ThematicLCModelConfig(
        max_seq_len=2048,
        model_dim=model_dim,
        theme_embed_dim=512,
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        encoder=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            ffn_inner_dim=model_dim * 4,
            num_attn_heads=num_attn_heads,
            num_layers=12,
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style="rms",
        ),
        theme_projection=ProjectionConfig(),
    )


@thematic_lcm_arch("thematic_lcm_large")
def thematic_lcm_large() -> ThematicLCModelConfig:
    """Large thematic LCM model for theme prediction
    High-capacity model for complex thematic understanding.
    """
    model_dim: int = 1536
    num_attn_heads: int = 24
    return ThematicLCModelConfig(
        max_seq_len=4096,
        model_dim=model_dim,
        theme_embed_dim=768,
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        encoder=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            ffn_inner_dim=model_dim * 4,
            num_attn_heads=num_attn_heads,
            num_layers=24,
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style="rms",
        ),
        theme_projection=ProjectionConfig(),
    )


@thematic_lcm_arch("thematic_lcm_classifier")
def thematic_lcm_classifier() -> ThematicLCModelConfig:
    """Thematic LCM model with classification head
    For discrete theme classification tasks.
    """
    model_dim: int = 1024
    num_attn_heads: int = 16
    return ThematicLCModelConfig(
        max_seq_len=2048,
        model_dim=model_dim,
        theme_embed_dim=512,
        num_themes=50,  # 50 theme categories
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        encoder=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            ffn_inner_dim=model_dim * 4,
            num_attn_heads=num_attn_heads,
            num_layers=12,
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style="rms",
        ),
        theme_projection=ProjectionConfig(),
    )


@thematic_lcm_arch("thematic_lcm_mdd_classifier")
def thematic_lcm_mdd_classifier() -> ThematicLCModelConfig:
    """Thematic LCM model for DSM-5 Major Depressive Disorder classification

    Classifies text based on the 9 DSM-5 MDD diagnostic criteria:
    1. Depressed mood most of the day, nearly every day
    2. Markedly diminished interest or pleasure in activities (anhedonia)
    3. Significant weight loss/gain or decrease/increase in appetite
    4. Insomnia or hypersomnia nearly every day
    5. Psychomotor agitation or retardation
    6. Fatigue or loss of energy nearly every day
    7. Feelings of worthlessness or excessive/inappropriate guilt
    8. Diminished ability to think or concentrate, or indecisiveness
    9. Recurrent thoughts of death, suicidal ideation, or suicide attempt
    """
    model_dim: int = 1024
    num_attn_heads: int = 16
    return ThematicLCModelConfig(
        max_seq_len=2048,
        model_dim=model_dim,
        theme_embed_dim=512,
        num_themes=9,  # 9 DSM-5 MDD diagnostic criteria
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        encoder=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            ffn_inner_dim=model_dim * 4,
            num_attn_heads=num_attn_heads,
            num_layers=12,
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style="rms",
        ),
        theme_projection=ProjectionConfig(),
    )
