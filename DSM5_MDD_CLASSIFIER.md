# DSM-5 Major Depressive Disorder (MDD) Thematic LCM Classifier

## Overview

The **DSM-5 MDD Thematic LCM Classifier** is a specialized variant of the Thematic Large Concept Model designed specifically for psychiatric assessment and research. This model classifies text content according to the 9 diagnostic criteria for Major Depressive Disorder as defined in the DSM-5 (Diagnostic and Statistical Manual of Mental Disorders, 5th Edition).

## Clinical Background

Major Depressive Disorder is characterized by specific symptoms that must be present for diagnosis. The DSM-5 specifies 9 core criteria:

### DSM-5 MDD Diagnostic Criteria (9 Classification Targets)

1. **Depressed Mood** - Depressed mood most of the day, nearly every day
2. **Anhedonia** - Markedly diminished interest or pleasure in activities  
3. **Appetite/Weight Changes** - Significant weight loss/gain or appetite changes
4. **Sleep Disturbances** - Insomnia or hypersomnia nearly every day
5. **Psychomotor Changes** - Psychomotor agitation or retardation
6. **Fatigue** - Fatigue or loss of energy nearly every day
7. **Worthlessness/Guilt** - Feelings of worthlessness or excessive guilt
8. **Concentration Issues** - Diminished ability to think, concentrate, or decide
9. **Death/Suicidal Thoughts** - Recurrent thoughts of death or suicidal ideation

## Model Architecture

The MDD classifier extends the base Thematic LCM architecture with the following specifications:

```python
@thematic_lcm_arch("thematic_lcm_mdd_classifier")
def thematic_lcm_mdd_classifier() -> ThematicLCModelConfig:
    return ThematicLCModelConfig(
        max_seq_len=2048,
        model_dim=1024,           # 1024-dimensional internal representation
        theme_embed_dim=512,      # 512-dimensional theme embeddings
        num_themes=9,             # 9 DSM-5 MDD diagnostic criteria
        num_layers=12,            # 12 transformer layers
        num_attn_heads=16,        # 16 attention heads
        # ... additional configuration
    )
```

**Key Features:**
- **Specialized Classification**: 9-class output corresponding to MDD criteria
- **Clinical Focus**: Optimized for psychiatric text analysis
- **Robust Architecture**: 13.3M parameters for reliable classification
- **Variable Input**: Handles paragraphs of varying lengths (3-15+ sentences)

## Usage

### Basic Usage

```python
from lcm.models.thematic_lcm import ThematicLCModelConfig, create_thematic_lcm_model
from lcm.datasets.batch import EmbeddingsBatch

# Create MDD classifier
config = ThematicLCModelConfig(model_arch="thematic_lcm_mdd_classifier")
model = create_thematic_lcm_model(config)

# Process clinical text (batch_size, num_sentences, sonar_dim)
clinical_text_embeddings = torch.randn(1, 8, 1024) * 0.006  
batch = EmbeddingsBatch(clinical_text_embeddings, padding_mask=None)

# Classify according to MDD criteria
theme_embeddings = model.predict_themes(batch)  # [1, 512]

# Get classification probabilities (if classifier head is available)
if hasattr(model, 'theme_classifier') and model.theme_classifier is not None:
    logits = model.theme_classifier(theme_embeddings)
    probabilities = torch.softmax(logits, dim=-1)  # [1, 9]
    predicted_criterion = torch.argmax(probabilities, dim=-1)
```

### Example Use Cases

#### 1. Clinical Assessment Support
```python
# Analyze patient journal entries for MDD symptoms
patient_text = "I wake up feeling empty and sad every morning..."
# Process with SONAR encoder -> MDD classifier
# Output: Probability distribution over 9 MDD criteria
```

#### 2. Research Applications
```python
# Analyze large corpora of mental health texts
research_corpus = [text1, text2, text3, ...]
# Batch process -> Statistical analysis of MDD symptom prevalence
```

#### 3. Screening Tool Development
```python
# Develop automated screening questionnaires
screening_responses = ["Question 1 response", "Question 2 response", ...]
# Classify responses -> Risk assessment scores
```

## Model Variants

| Variant | Parameters | Use Case | Performance |
|---------|------------|----------|-------------|
| `thematic_lcm_mdd_classifier` | 13.3M | Clinical assessment | High accuracy |
| *Future: `thematic_lcm_mdd_small`* | 1.9M | Mobile/edge deployment | Fast inference |
| *Future: `thematic_lcm_mdd_large`* | 43.6M | Research applications | Maximum accuracy |

## Clinical Applications

### Primary Applications
- **Symptom Detection**: Identify presence of specific MDD criteria in text
- **Severity Assessment**: Quantify symptom intensity through embedding analysis
- **Longitudinal Tracking**: Monitor symptom changes over time
- **Research Support**: Large-scale analysis of mental health literature

### Validation Requirements
⚠️ **Important Clinical Disclaimer**: This model is intended for research and supportive applications only. It should NOT be used as a standalone diagnostic tool. Clinical decisions should always involve qualified mental health professionals.

### Ethical Considerations
- **Privacy**: Ensure secure handling of sensitive mental health data
- **Bias**: Regular evaluation for demographic and cultural biases
- **Transparency**: Clear communication about model limitations
- **Professional Oversight**: Integration with clinical workflows and supervision

## Training and Evaluation

### Training Data Requirements
- **Annotated Clinical Text**: Text samples labeled with MDD criteria
- **Diverse Demographics**: Representative patient populations
- **Professional Validation**: Clinician-verified annotations
- **Balanced Classes**: Adequate representation of all 9 criteria

### Evaluation Metrics
- **Criterion-Specific Accuracy**: Performance per MDD criterion
- **Sensitivity/Specificity**: Clinical detection rates
- **Inter-Rater Reliability**: Agreement with clinical assessments
- **Fairness Metrics**: Performance across demographic groups

### Example Training Configuration
```yaml
# recipes/train/pretrain/mdd_classification.yaml
model_arch: thematic_lcm_mdd_classifier
dataset: clinical_mdd_corpus
batch_size: 32
learning_rate: 1e-4
num_epochs: 50
evaluation_metrics:
  - criterion_accuracy
  - sensitivity_specificity
  - clinical_agreement
```

## Testing and Validation

### Test Scripts
1. **`test_mdd_classifier.py`** - Comprehensive functionality test
2. **`validate_mdd_classifier.py`** - Configuration validation
3. **Clinical validation scripts** (to be developed)

### Running Tests
```bash
# Basic functionality test
python test_mdd_classifier.py

# Configuration validation
python validate_mdd_classifier.py

# Unit tests (requires dependencies)
python -m pytest tests/units/models/test_thematic_lcm.py::test_mdd_classifier
```

## Integration with Existing Framework

The MDD classifier integrates seamlessly with the existing Thematic LCM infrastructure:

- **Model Registry**: Automatically registered as `thematic_lcm_mdd_classifier`
- **Training Pipeline**: Compatible with existing training recipes
- **Evaluation Framework**: Uses standard thematic LCM evaluation tools
- **Data Processing**: Standard SONAR sentence embedding pipeline

## Future Development

### Planned Enhancements
1. **Multi-Label Classification**: Simultaneous detection of multiple criteria
2. **Severity Scoring**: Quantitative severity assessment for each criterion
3. **Temporal Modeling**: Tracking symptom changes over time
4. **Multimodal Integration**: Integration with speech and behavioral data
5. **Clinical Validation Studies**: Extensive validation with clinical populations

### Research Directions
- **Personalized Models**: Patient-specific adaptation
- **Cross-Cultural Validation**: Performance across different populations
- **Intervention Monitoring**: Treatment response tracking
- **Early Detection**: Prodromal symptom identification

## References

1. American Psychiatric Association. (2013). Diagnostic and statistical manual of mental disorders (5th ed.)
2. [Thematic LCM Documentation](THEMATIC_LCM.md)
3. Clinical validation studies (in progress)

## Support and Contribution

For questions about the MDD classifier or to contribute clinical validation data:
- Review existing issues and documentation
- Ensure compliance with clinical data privacy requirements
- Follow ethical guidelines for mental health research

---

**Disclaimer**: This tool is for research purposes only and should not replace professional clinical assessment. Always consult qualified mental health professionals for clinical decisions.