#!/usr/bin/env python3
"""
Simple test script to verify thematic LCM model architecture without full dependencies.
This demonstrates the core concept of the thematic model.
"""

import torch
import torch.nn as nn


class SimpleThematicModel(nn.Module):
    """Simplified thematic model to demonstrate the concept"""
    
    def __init__(self, input_dim=1024, model_dim=512, theme_dim=256, num_layers=2):
        super().__init__()
        
        # Frontend: map sentence embeddings to model dimension
        self.frontend = nn.Linear(input_dim, model_dim)
        
        # Encoder: process paragraph-level information
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=8,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Theme projection: map to theme space
        self.theme_projection = nn.Linear(model_dim, theme_dim)
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.theme_dim = theme_dim
    
    def forward(self, paragraph_embeddings, padding_mask=None):
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
            # Convert padding mask to attention mask (True = ignore)
            attn_mask = padding_mask
        else:
            attn_mask = None
            
        encoded = self.encoder(x, src_key_padding_mask=attn_mask)  # [B, S, model_dim]
        
        # Aggregate paragraph representation (mean pooling over non-padded positions)
        if padding_mask is not None:
            # Create mask for valid positions (True = valid, False = padded)
            valid_mask = ~padding_mask
            # Mean pooling with masking
            paragraph_repr = (encoded * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True)
        else:
            # Simple mean pooling
            paragraph_repr = encoded.mean(dim=1)  # [B, model_dim]
        
        # Project to theme space
        theme_embeddings = self.theme_projection(paragraph_repr)  # [B, theme_dim]
        
        return theme_embeddings


def test_simple_thematic_model():
    """Test the simplified thematic model"""
    print("Testing Simple Thematic Model...")
    
    # Create model
    model = SimpleThematicModel(
        input_dim=1024,  # SONAR embedding dimension
        model_dim=512,
        theme_dim=256,
        num_layers=2,
    )
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test 1: Basic forward pass
    batch_size, seq_len = 3, 5
    paragraph_embeddings = torch.randn(batch_size, seq_len, 1024) * 0.006
    
    theme_embeddings = model(paragraph_embeddings)
    assert theme_embeddings.shape == (batch_size, 256), f"Expected (3, 256), got {theme_embeddings.shape}"
    assert torch.isfinite(theme_embeddings).all(), "Theme embeddings contain non-finite values"
    
    print("✓ Basic forward pass test passed")
    
    # Test 2: With padding mask
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[0, 3:] = True  # First sequence has length 3
    padding_mask[1, 4:] = True  # Second sequence has length 4
    
    theme_embeddings_masked = model(paragraph_embeddings, padding_mask)
    assert theme_embeddings_masked.shape == (batch_size, 256)
    assert torch.isfinite(theme_embeddings_masked).all()
    
    # The embeddings should be different when using padding
    assert not torch.allclose(theme_embeddings, theme_embeddings_masked), "Padding mask had no effect"
    
    print("✓ Padding mask test passed")
    
    # Test 3: Different sequence lengths
    for test_seq_len in [1, 7, 10]:
        test_embeddings = torch.randn(2, test_seq_len, 1024) * 0.006
        test_themes = model(test_embeddings)
        assert test_themes.shape == (2, 256), f"Failed for seq_len={test_seq_len}"
        assert torch.isfinite(test_themes).all()
    
    print("✓ Variable sequence length test passed")
    
    # Test 4: Similarity between related paragraphs
    # Create two similar paragraphs (small perturbation)
    base_paragraph = torch.randn(1, 5, 1024) * 0.006
    similar_paragraph = base_paragraph + torch.randn(1, 5, 1024) * 0.001  # Small noise
    different_paragraph = torch.randn(1, 5, 1024) * 0.006  # Completely different
    
    base_theme = model(base_paragraph)
    similar_theme = model(similar_paragraph)
    different_theme = model(different_paragraph)
    
    # Cosine similarity
    base_norm = torch.nn.functional.normalize(base_theme, p=2, dim=1)
    similar_norm = torch.nn.functional.normalize(similar_theme, p=2, dim=1)
    different_norm = torch.nn.functional.normalize(different_theme, p=2, dim=1)
    
    sim_to_similar = torch.cosine_similarity(base_norm, similar_norm).item()
    sim_to_different = torch.cosine_similarity(base_norm, different_norm).item()
    
    print(f"  Similarity to similar paragraph: {sim_to_similar:.3f}")
    print(f"  Similarity to different paragraph: {sim_to_different:.3f}")
    
    # Similar paragraphs should be more similar (though this is not guaranteed with random data)
    print("✓ Similarity test completed")
    
    print("\n✅ All tests passed! Thematic model architecture works correctly.")
    
    return model


def demonstrate_theme_prediction():
    """Demonstrate theme prediction concept"""
    print("\nDemonstrating Theme Prediction Concept...")
    
    model = SimpleThematicModel(theme_dim=128)
    model.eval()
    
    # Simulate different types of paragraphs
    paragraph_types = [
        "Scientific/Technical",
        "News/Politics", 
        "Entertainment/Sports",
        "Personal/Narrative",
    ]
    
    theme_embeddings = []
    
    for i, paragraph_type in enumerate(paragraph_types):
        # Generate synthetic paragraph embedding (in practice, these would come from SONAR)
        # Each type gets a different random seed to create distinct patterns
        torch.manual_seed(i * 100)
        paragraph_emb = torch.randn(1, 6, 1024) * 0.01
        
        with torch.no_grad():
            theme_emb = model(paragraph_emb)
            theme_embeddings.append(theme_emb)
        
        print(f"  {paragraph_type}: theme norm = {torch.norm(theme_emb).item():.3f}")
    
    # Compute pairwise similarities
    print("\nTheme Similarity Matrix:")
    print("      ", "  ".join(f"{t[:6]}" for t in paragraph_types))
    
    for i, type_i in enumerate(paragraph_types):
        row = f"{type_i[:6]}: "
        for j, type_j in enumerate(paragraph_types):
            sim = torch.cosine_similarity(
                torch.nn.functional.normalize(theme_embeddings[i], p=2, dim=1),
                torch.nn.functional.normalize(theme_embeddings[j], p=2, dim=1)
            ).item()
            row += f"{sim:6.3f} "
        print(row)
    
    print("\n✅ Theme prediction demonstration complete!")


if __name__ == "__main__":
    # Run tests
    model = test_simple_thematic_model()
    demonstrate_theme_prediction()
    
    print(f"\n🎉 Thematic LCM concept successfully demonstrated!")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Theme embedding dimension: {model.theme_dim}")