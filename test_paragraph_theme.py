#!/usr/bin/env python3
"""
Script to test a single paragraph with the Thematic LCM model.
This script is designed to work with GitHub Actions for manual testing.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import argparse
import json


class ThematicTestModel(nn.Module):
    """
    Simplified Thematic LCM model for testing purposes.
    This works without full dependencies and demonstrates theme prediction.
    """
    
    def __init__(self, input_dim=1024, model_dim=512, theme_dim=256, num_layers=4):
        super().__init__()
        
        # Frontend: sentence embeddings -> model dimension
        self.frontend = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.Dropout(0.1),
        )
        
        # Encoder: transformer layers for paragraph processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=8,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Theme projection
        self.theme_projection = nn.Sequential(
            nn.Linear(model_dim, theme_dim),
            nn.Tanh(),  # Normalize theme embeddings
        )
        
        self.theme_dim = theme_dim
        
    def forward(self, paragraph_embeddings: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for thematic modeling.
        
        Args:
            paragraph_embeddings: [batch_size, seq_len, input_dim] sentence embeddings
            padding_mask: [batch_size, seq_len] True for padded positions
            
        Returns:
            theme_embeddings: [batch_size, theme_dim] paragraph theme representations
        """
        batch_size, seq_len, _ = paragraph_embeddings.shape
        
        # Frontend processing
        x = self.frontend(paragraph_embeddings)  # [B, S, model_dim]
        
        # Encoder processing
        if padding_mask is not None:
            attn_mask = padding_mask
        else:
            attn_mask = None
            
        encoded = self.encoder(x, src_key_padding_mask=attn_mask)  # [B, S, model_dim]
        
        # Aggregate paragraph representation (mean pooling over non-padded positions)
        if padding_mask is not None:
            valid_mask = ~padding_mask
            paragraph_repr = (encoded * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True)
        else:
            paragraph_repr = encoded.mean(dim=1)  # [B, model_dim]
        
        # Project to theme space
        theme_embeddings = self.theme_projection(paragraph_repr)  # [B, theme_dim]
        
        return theme_embeddings


def simulate_sentence_embeddings(text: str, target_sentences: int = None) -> torch.Tensor:
    """
    Simulate SONAR sentence embeddings for the input text.
    In practice, this would use real SONAR embeddings.
    """
    # Split text into sentences (simple approximation)
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    
    if target_sentences is None:
        num_sentences = max(len(sentences), 3)  # At least 3 sentences
    else:
        num_sentences = target_sentences
    
    # Create embeddings based on text characteristics
    torch.manual_seed(hash(text.lower()) % 2**31)  # Deterministic but text-dependent
    embeddings = torch.randn(1, num_sentences, 1024) * 0.006
    
    # Add some structure based on text content
    if any(word in text.lower() for word in ['research', 'study', 'analysis', 'experiment', 'scientific']):
        embeddings[0, :, :256] += 0.1  # Scientific theme signal
    elif any(word in text.lower() for word in ['news', 'government', 'election', 'policy', 'politics']):
        embeddings[0, :, 256:512] += 0.1  # News/Politics theme signal
    elif any(word in text.lower() for word in ['game', 'sport', 'team', 'player', 'match']):
        embeddings[0, :, 512:768] += 0.1  # Sports theme signal
    elif any(word in text.lower() for word in ['feel', 'emotion', 'personal', 'my', 'i was']):
        embeddings[0, :, 768:1024] += 0.1  # Personal theme signal
    
    return embeddings


def predict_theme_category(theme_embedding: torch.Tensor) -> str:
    """
    Predict theme category based on theme embedding characteristics.
    This is a simplified classification for demonstration.
    """
    # Analyze different regions of the theme embedding
    embedding = theme_embedding.squeeze().detach()
    theme_dim = embedding.shape[0]
    
    # Dynamically determine the regions based on theme dimension
    region_size = max(1, theme_dim // 4)
    
    theme_scores = {}
    theme_names = ['Scientific/Research', 'News/Politics', 'Sports/Entertainment', 'Personal/Narrative']
    
    for i, theme_name in enumerate(theme_names):
        start_idx = i * region_size
        end_idx = min((i + 1) * region_size, theme_dim)
        if start_idx < theme_dim:
            score = embedding[start_idx:end_idx].mean().item()
            # Handle NaN values
            if torch.isnan(torch.tensor(score)):
                score = 0.0
            theme_scores[theme_name] = score
        else:
            theme_scores[theme_name] = 0.0
    
    # Find the dominant theme
    predicted_theme = max(theme_scores.items(), key=lambda x: x[1])
    
    return predicted_theme[0], theme_scores


def analyze_paragraph_theme(paragraph_text: str, model_size: str = "small") -> dict:
    """
    Analyze the theme of a paragraph using the Thematic LCM model.
    
    Args:
        paragraph_text: Input paragraph text
        model_size: Model size (small, medium, large)
    
    Returns:
        Dictionary with theme analysis results
    """
    # Model configurations
    configs = {
        "small": {"model_dim": 256, "theme_dim": 128, "num_layers": 2},
        "medium": {"model_dim": 512, "theme_dim": 256, "num_layers": 4},
        "large": {"model_dim": 768, "theme_dim": 384, "num_layers": 6},
    }
    
    config = configs.get(model_size, configs["medium"])
    
    # Create model
    model = ThematicTestModel(
        input_dim=1024,
        model_dim=config["model_dim"],
        theme_dim=config["theme_dim"],
        num_layers=config["num_layers"]
    )
    
    # Set to evaluation mode
    model.eval()
    
    # Simulate sentence embeddings for the paragraph
    paragraph_embeddings = simulate_sentence_embeddings(paragraph_text)
    
    # Predict theme
    with torch.no_grad():
        theme_embedding = model(paragraph_embeddings)
    
    # Analyze the theme
    predicted_theme, theme_scores = predict_theme_category(theme_embedding)
    
    # Calculate theme strength (how confident the prediction is)
    theme_values = list(theme_scores.values())
    theme_strength = max(theme_values) - min(theme_values)
    
    return {
        "paragraph": paragraph_text,
        "model_size": model_size,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "predicted_theme": predicted_theme,
        "theme_scores": theme_scores,
        "theme_strength": round(theme_strength, 4),
        "theme_embedding_stats": {
            "mean": round(theme_embedding.mean().item(), 4),
            "std": round(theme_embedding.std().item(), 4),
            "min": round(theme_embedding.min().item(), 4),
            "max": round(theme_embedding.max().item(), 4),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Test paragraph theme prediction with Thematic LCM")
    parser.add_argument("paragraph", help="Input paragraph text to analyze")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="medium",
                        help="Model size to use (default: medium)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    args = parser.parse_args()
    
    try:
        # Analyze the paragraph
        results = analyze_paragraph_theme(args.paragraph, args.model_size)
        
        if args.json:
            # Output JSON for programmatic use
            print(json.dumps(results, indent=2))
        else:
            # Human-readable output
            print("🔍 Thematic LCM Analysis Results")
            print("=" * 50)
            print(f"📝 Input Paragraph: {results['paragraph'][:100]}{'...' if len(results['paragraph']) > 100 else ''}")
            print(f"🤖 Model: {results['model_size']} ({results['model_parameters']:,} parameters)")
            print(f"🎯 Predicted Theme: {results['predicted_theme']}")
            print(f"💪 Theme Strength: {results['theme_strength']}")
            print()
            print("📊 Theme Scores:")
            for theme, score in results['theme_scores'].items():
                bar_length = int(score * 20) if score > 0 else 0
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {theme:<20} {score:>6.3f} {bar}")
            print()
            print("📈 Theme Embedding Statistics:")
            stats = results['theme_embedding_stats']
            print(f"  Mean: {stats['mean']:>8.4f}")
            print(f"  Std:  {stats['std']:>8.4f}")
            print(f"  Range: [{stats['min']:>6.4f}, {stats['max']:>6.4f}]")
    
    except Exception as e:
        print(f"❌ Error analyzing paragraph: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()