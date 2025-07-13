# GitHub Actions Thematic Model Testing Workflow

## Overview

This workflow provides an easy way to test the Thematic LCM model directly from the GitHub repository using GitHub Actions. You can analyze any paragraph and see its predicted themes without needing to set up a local development environment.

## How to Use

1. **Navigate to Actions Tab**: Go to the repository's Actions tab in GitHub
2. **Select Workflow**: Click on "Test Thematic Model" in the left sidebar
3. **Run Workflow**: Click "Run workflow" button
4. **Enter Input**: Fill in the required inputs:
   - **Paragraph text**: The text you want to analyze for themes
   - **Model size** (optional): Choose from small, medium, or large (default: medium)
5. **Run**: Click "Run workflow" to start the analysis

## Workflow Features

### Manual Trigger
- **workflow_dispatch**: Can be triggered manually from the GitHub Actions interface
- **Text Input**: Accepts any paragraph text for theme analysis
- **Model Selection**: Choose between different model sizes for comparison

### Theme Prediction Capabilities
- **Four Theme Categories**:
  - Scientific/Research
  - News/Politics  
  - Sports/Entertainment
  - Personal/Narrative
- **Multiple Model Sizes**: Test with small (1.9M), medium (13.3M), and large (43.6M) parameter models
- **Detailed Analysis**: View theme scores, confidence metrics, and embedding statistics

### Output Information
- **Primary Analysis**: Human-readable theme prediction with confidence scores
- **Comparison View**: Side-by-side results across all model sizes
- **JSON Details**: Complete analysis data in structured format
- **Example Tests**: Automatic testing with predefined theme examples

## Example Inputs

### Scientific Research
```
The recent study published in Nature demonstrates a significant breakthrough in quantum computing. Researchers have successfully implemented a new error correction algorithm that reduces quantum decoherence by 95%. This advancement brings us closer to practical quantum computers that could revolutionize cryptography and drug discovery.
```

### Sports/Entertainment  
```
The championship game last night was absolutely thrilling. Our team scored three goals in the final quarter, with the winning goal coming in the last two minutes. The crowd went wild as the players celebrated their hard-fought victory. This season has been their best performance in over a decade.
```

### Personal Narrative
```
I woke up this morning feeling incredibly grateful for my family. Yesterday was my daughter's birthday, and seeing her face light up when she opened her presents reminded me of what truly matters in life. These moments of pure joy and connection are what make all the challenges worthwhile.
```

### News/Politics
```
The new policy proposal has sparked intense debate in Congress this week. Representatives from both parties are scheduled to meet tomorrow to discuss potential amendments to the healthcare reform bill. Public opinion polls show mixed reactions from constituents across different districts.
```

## Output Format

The workflow provides multiple views of the analysis:

1. **Summary View**: Quick theme prediction with model info
2. **Detailed Scores**: Bar chart visualization of theme scores  
3. **Statistical Analysis**: Theme embedding statistics
4. **Comparison Table**: Results across different model sizes
5. **JSON Export**: Complete structured data for programmatic use

## Technical Details

- **Model Architecture**: Simplified Thematic LCM implementation
- **Input Processing**: Simulated SONAR sentence embeddings (1024D)
- **Theme Embedding**: 128D-384D depending on model size
- **Processing**: Transformer encoder with mean pooling aggregation

## Local Testing

You can also run the theme analysis locally:

```bash
# Basic usage
python test_paragraph_theme.py "Your paragraph text here" --model-size medium

# JSON output
python test_paragraph_theme.py "Your text" --model-size large --json

# Help
python test_paragraph_theme.py --help
```

This workflow demonstrates the core capabilities of the Thematic LCM model and provides an accessible way to experiment with paragraph-level theme prediction.