# Thematic Large Concept Model (Thematic LCM)

The Thematic LCM is a specialized variant of the Large Concept Model designed to understand and predict themes at the paragraph level. While standard LCMs operate on sentence-level representations, the Thematic LCM processes sequences of sentences (paragraphs) to extract higher-level thematic information.

## Overview

The Thematic LCM follows the same architectural principles as other LCM models but is specifically designed for theme prediction:

- **Input**: Paragraphs represented as sequences of SONAR sentence embeddings
- **Processing**: Transformer encoder processes sentence sequences to understand paragraph-level semantics
- **Output**: Theme embeddings or discrete theme classifications

## Architecture

The model consists of three main components:

1. **Frontend**: Maps SONAR sentence embeddings (1024-dim) to the model's internal dimension
2. **Encoder**: Transformer encoder that processes paragraph-level information across sentences
3. **Theme Projection**: Maps the aggregated paragraph representation to thematic space

### Key Features

- **Paragraph-level Processing**: Unlike sentence-level LCMs, processes entire paragraphs
- **Theme Embeddings**: Outputs continuous theme representations
- **Optional Classification**: Can include discrete theme classification head
- **Flexible Input**: Handles variable-length paragraphs with padding support

## Model Variants

### Available Architectures

- `toy_thematic_lcm`: Minimal model for testing (2 layers, 256-dim themes)
- `thematic_lcm_small`: Small model for experimentation (6 layers, 512-dim model, 256-dim themes)
- `thematic_lcm_base`: Standard model (12 layers, 1024-dim model, 512-dim themes)
- `thematic_lcm_large`: Large model (24 layers, 1536-dim model, 768-dim themes)
- `thematic_lcm_classifier`: Base model with 50-class theme classifier

## Usage

### Basic Model Creation

```python
from lcm.models.thematic_lcm import ThematicLCModelConfig, create_thematic_lcm_model
from lcm.nn.transformer import TransformerConfig

# Create a basic thematic model
config = ThematicLCModelConfig(
    model_arch="thematic_lcm_base",
    theme_embed_dim=512,
)
model = create_thematic_lcm_model(config)
```

### Theme Prediction

```python
import torch
from lcm.datasets.batch import EmbeddingsBatch

# Prepare paragraph data (batch_size, num_sentences, sonar_dim)
paragraphs = torch.randn(2, 5, 1024) * 0.006  # 2 paragraphs, 5 sentences each
batch = EmbeddingsBatch(paragraphs, padding_mask=None)

# Get theme embeddings
model.eval()
with torch.no_grad():
    theme_embeddings = model.predict_themes(batch)
    print(f"Theme embeddings shape: {theme_embeddings.shape}")  # [2, 512]
```

### With Classification Head

```python
# Model with discrete theme classification
config = ThematicLCModelConfig(
    model_arch="thematic_lcm_classifier",
    num_themes=50,  # 50 theme categories
    theme_embed_dim=512,
)
classifier_model = create_thematic_lcm_model(config)

# Predict theme classes
theme_classes = classifier_model.predict_themes(batch, return_probabilities=False)
print(f"Predicted theme classes: {theme_classes}")  # [2] - class indices

# Get class probabilities
theme_probs = classifier_model.predict_themes(batch, return_probabilities=True)
print(f"Theme probabilities shape: {theme_probs.shape}")  # [2, 50]
```

## Training

### Pre-training

To train a thematic LCM model, use the provided training recipe:

```bash
python -m lcm.train \
    +pretrain=thematic \
    ++trainer.output_dir="checkpoints/thematic_lcm" \
    ++trainer.experiment_name=training_thematic_lcm
```

### Data Requirements

The thematic model expects paragraph-level data where each paragraph is represented as:
- A sequence of sentence embeddings (SONAR 1024-dimensional)
- Variable length sequences with padding support
- Minimum 3 sentences per paragraph recommended

### Custom Training

For custom theme prediction tasks:

1. Prepare your paragraph data in the expected format
2. Create custom datacards pointing to your data
3. Modify the training recipe for your specific requirements
4. Optionally add theme classification heads for discrete theme prediction

## Evaluation

### Using the Thematic Predictor

```python
from lcm.evaluation.predictors.thematic_lcm import ThematicLCMPredictor

# Load trained model
predictor = ThematicLCMPredictor("path/to/model_card.yaml")

# Predict themes for paragraphs
paragraphs = [
    [sentence_emb1, sentence_emb2, ...],  # Paragraph 1
    [sentence_emb1, sentence_emb2, ...],  # Paragraph 2
]

theme_predictions = predictor.predict_themes(paragraphs)
```

### Text-based Prediction

```python
# Predict themes from raw text (requires sentence encoder)
from sonar_space import SonarEncoder  # Example

encoder = SonarEncoder()
texts = [
    "This is the first paragraph. It contains multiple sentences. Each sentence contributes to the overall theme.",
    "This is another paragraph. It has a different thematic focus. The model should distinguish between themes."
]

theme_predictions = predictor.predict_paragraph_themes(
    texts, 
    sentence_encoder=encoder.encode
)
```

## Applications

The Thematic LCM is designed for:

- **Document Categorization**: Classify documents by thematic content
- **Content Analysis**: Understand thematic patterns in text collections
- **Semantic Search**: Find documents with similar thematic content
- **Content Generation**: Generate text with specific thematic properties
- **Topic Modeling**: Discover and track themes across document collections

## Comparison with Other LCM Models

| Feature | Base LCM | Two-Tower Diffusion LCM | Thematic LCM |
|---------|----------|-------------------------|--------------|
| Input Level | Sentence | Sentence | Paragraph |
| Output | Next sentence prediction | Diffusion generation | Theme prediction |
| Aggregation | N/A | N/A | Mean pooling |
| Use Case | Language modeling | Creative generation | Theme understanding |

## Technical Details

### Model Parameters

- **Thematic LCM Base**: ~35M parameters (smaller than base LCM due to paragraph-level processing)
- **Theme Embedding Dimension**: Configurable (256, 512, 768)
- **Max Sequence Length**: 2048 sentences per paragraph (configurable)
- **SONAR Compatibility**: Full compatibility with SONAR sentence embeddings

### Performance Considerations

- **Memory**: Lower memory usage per token due to paragraph-level processing
- **Speed**: Faster inference for theme prediction tasks
- **Accuracy**: Optimized for thematic understanding rather than token-level generation

## Examples

See `test_thematic_concept.py` for a complete working example demonstrating:
- Model creation and configuration
- Forward pass with padding
- Theme similarity computation
- Different paragraph types

The example shows a simplified version achieving the core thematic modeling concept with ~7M parameters.