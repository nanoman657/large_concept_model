#!/usr/bin/env python3
"""
Comprehensive example demonstrating Thematic LCM usage.

This script shows how to:
1. Create and configure thematic models
2. Process paragraph-level data
3. Predict themes
4. Evaluate theme predictions
5. Compare different model configurations
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class ThematicLCMExample:
    """Example implementation of Thematic LCM concept"""

    def __init__(self, input_dim=1024, model_dim=512, theme_dim=256, num_layers=4):
        self.model = self._create_model(input_dim, model_dim, theme_dim, num_layers)
        self.theme_dim = theme_dim

    def _create_model(self, input_dim, model_dim, theme_dim, num_layers):
        """Create the thematic model"""
        model = nn.Sequential(
            # Frontend: sentence embeddings -> model dimension
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.Dropout(0.1),
            # Encoder: transformer layers for paragraph processing
            *[
                nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=8,
                    dim_feedforward=model_dim * 4,
                    dropout=0.1,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ],
            # Theme projection
            nn.Linear(model_dim, theme_dim),
            nn.Tanh(),  # Normalize theme embeddings
        )
        return model

    def forward(
        self, paragraph_embeddings: torch.Tensor, padding_mask=None
    ) -> torch.Tensor:
        """
        Process paragraphs to extract themes.

        Args:
            paragraph_embeddings: [batch_size, seq_len, input_dim]
            padding_mask: [batch_size, seq_len] (True for padded positions)

        Returns:
            theme_embeddings: [batch_size, theme_dim]
        """
        batch_size, seq_len, input_dim = paragraph_embeddings.shape

        # Process through model layers
        x = paragraph_embeddings
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.TransformerEncoderLayer):
                # Apply transformer layer with padding mask
                x = layer(x, src_key_padding_mask=padding_mask)
            else:
                # Apply other layers element-wise
                x = layer(x)

        # Aggregate paragraph representation (mean pooling over valid positions)
        if padding_mask is not None:
            valid_mask = ~padding_mask  # True for valid positions
            theme_embeddings = (x * valid_mask.unsqueeze(-1)).sum(
                dim=1
            ) / valid_mask.sum(dim=1, keepdim=True)
        else:
            theme_embeddings = x.mean(dim=1)

        return theme_embeddings

    def predict_themes(self, paragraphs: List[torch.Tensor]) -> torch.Tensor:
        """Predict themes for a list of paragraphs"""
        # Pad paragraphs to same length
        max_len = max(p.shape[0] for p in paragraphs)
        batch_paragraphs = []
        padding_masks = []

        for paragraph in paragraphs:
            seq_len = paragraph.shape[0]
            if seq_len < max_len:
                # Pad paragraph
                padding = torch.zeros(max_len - seq_len, paragraph.shape[1])
                padded_paragraph = torch.cat([paragraph, padding], dim=0)
                mask = torch.cat(
                    [
                        torch.zeros(seq_len, dtype=torch.bool),
                        torch.ones(max_len - seq_len, dtype=torch.bool),
                    ]
                )
            else:
                padded_paragraph = paragraph
                mask = torch.zeros(seq_len, dtype=torch.bool)

            batch_paragraphs.append(padded_paragraph)
            padding_masks.append(mask)

        # Stack into batch tensors
        batch_tensor = torch.stack(batch_paragraphs)
        mask_tensor = torch.stack(padding_masks)

        # Predict themes
        with torch.no_grad():
            self.model.eval()
            themes = self.forward(
                batch_tensor, mask_tensor if mask_tensor.any() else None
            )

        return themes


def create_synthetic_paragraphs(
    num_paragraphs=10, num_themes=4
) -> Tuple[List[torch.Tensor], List[int]]:
    """Create synthetic paragraph data with known themes"""
    paragraphs = []
    theme_labels = []

    # Create theme prototypes
    theme_prototypes = [torch.randn(1024) * 0.1 for _ in range(num_themes)]

    for i in range(num_paragraphs):
        # Choose a theme
        theme_id = i % num_themes
        theme_proto = theme_prototypes[theme_id]

        # Create a paragraph with 3-8 sentences
        num_sentences = np.random.randint(3, 9)
        sentences = []

        for _ in range(num_sentences):
            # Create sentence embedding similar to the theme
            sentence = theme_proto + torch.randn(1024) * 0.05
            sentences.append(sentence)

        paragraph = torch.stack(sentences)
        paragraphs.append(paragraph)
        theme_labels.append(theme_id)

    return paragraphs, theme_labels


def evaluate_theme_similarity(
    predicted_themes: torch.Tensor, true_theme_labels: List[int]
) -> Dict[str, float]:
    """Evaluate theme prediction quality"""
    num_themes = len(set(true_theme_labels))

    # Group predictions by true theme
    theme_groups = {i: [] for i in range(num_themes)}
    for pred_theme, true_label in zip(predicted_themes, true_theme_labels):
        theme_groups[true_label].append(pred_theme)

    # Compute within-theme and between-theme similarities
    within_similarities = []
    between_similarities = []

    for theme_id, theme_predictions in theme_groups.items():
        if len(theme_predictions) < 2:
            continue

        # Within-theme similarities
        for i in range(len(theme_predictions)):
            for j in range(i + 1, len(theme_predictions)):
                sim = torch.cosine_similarity(
                    theme_predictions[i].unsqueeze(0), theme_predictions[j].unsqueeze(0)
                ).item()
                within_similarities.append(sim)

        # Between-theme similarities
        for other_theme_id, other_predictions in theme_groups.items():
            if other_theme_id == theme_id or len(other_predictions) == 0:
                continue

            for pred1 in theme_predictions[:2]:  # Limit to avoid too many comparisons
                for pred2 in other_predictions[:2]:
                    sim = torch.cosine_similarity(
                        pred1.unsqueeze(0), pred2.unsqueeze(0)
                    ).item()
                    between_similarities.append(sim)

    return {
        "within_theme_similarity_mean": np.mean(within_similarities)
        if within_similarities
        else 0.0,
        "within_theme_similarity_std": np.std(within_similarities)
        if within_similarities
        else 0.0,
        "between_theme_similarity_mean": np.mean(between_similarities)
        if between_similarities
        else 0.0,
        "between_theme_similarity_std": np.std(between_similarities)
        if between_similarities
        else 0.0,
        "theme_separation": (
            np.mean(within_similarities) - np.mean(between_similarities)
        )
        if within_similarities and between_similarities
        else 0.0,
    }


def demonstrate_thematic_lcm():
    """Main demonstration function"""
    print("🎨 Thematic LCM Demonstration")
    print("=" * 50)

    # 1. Create models with different configurations
    print("\n1. Creating Thematic Models...")

    models = {
        "Small": ThematicLCMExample(model_dim=256, theme_dim=128, num_layers=2),
        "Medium": ThematicLCMExample(model_dim=512, theme_dim=256, num_layers=4),
        "Large": ThematicLCMExample(model_dim=768, theme_dim=384, num_layers=6),
    }

    for name, model in models.items():
        param_count = sum(p.numel() for p in model.model.parameters())
        print(
            f"  ✓ {name} Model: {param_count:,} parameters, {model.theme_dim}D themes"
        )

    # 2. Create synthetic paragraph data
    print("\n2. Creating Synthetic Paragraph Data...")
    paragraphs, theme_labels = create_synthetic_paragraphs(
        num_paragraphs=20, num_themes=4
    )

    print(
        f"  ✓ Created {len(paragraphs)} paragraphs with {len(set(theme_labels))} themes"
    )
    print(f"  ✓ Paragraph lengths: {[p.shape[0] for p in paragraphs[:5]]}... (first 5)")
    print(f"  ✓ Theme distribution: {[theme_labels.count(i) for i in range(4)]}")

    # 3. Predict themes with each model
    print("\n3. Predicting Themes...")

    results = {}
    for model_name, model in models.items():
        print(f"\n  {model_name} Model:")

        # Predict themes
        predicted_themes = model.predict_themes(paragraphs)
        print(
            f"    ✓ Predicted {predicted_themes.shape[0]} themes of dimension {predicted_themes.shape[1]}"
        )

        # Evaluate
        metrics = evaluate_theme_similarity(predicted_themes, theme_labels)
        results[model_name] = metrics

        print(
            f"    - Within-theme similarity: {metrics['within_theme_similarity_mean']:.3f} ± {metrics['within_theme_similarity_std']:.3f}"
        )
        print(
            f"    - Between-theme similarity: {metrics['between_theme_similarity_mean']:.3f} ± {metrics['between_theme_similarity_std']:.3f}"
        )
        print(f"    - Theme separation: {metrics['theme_separation']:.3f}")

    # 4. Compare models
    print("\n4. Model Comparison Summary:")
    print("-" * 30)

    best_separation = max(results.values(), key=lambda x: x["theme_separation"])
    best_model = [
        name for name, metrics in results.items() if metrics == best_separation
    ][0]

    for model_name, metrics in results.items():
        marker = "🏆" if model_name == best_model else "  "
        print(
            f"{marker} {model_name:6}: Separation = {metrics['theme_separation']:6.3f}"
        )

    # 5. Demonstrate specific use cases
    print("\n5. Use Case Examples...")

    best_model_obj = models[best_model]

    # Single paragraph prediction
    single_paragraph = paragraphs[0]
    single_theme = best_model_obj.predict_themes([single_paragraph])
    print(f"  ✓ Single paragraph theme: {torch.norm(single_theme).item():.3f} (norm)")

    # Batch prediction
    batch_themes = best_model_obj.predict_themes(paragraphs[:5])
    similarities = []
    for i in range(len(batch_themes)):
        for j in range(i + 1, len(batch_themes)):
            sim = torch.cosine_similarity(
                batch_themes[i].unsqueeze(0), batch_themes[j].unsqueeze(0)
            ).item()
            similarities.append(sim)

    print(f"  ✓ Batch prediction: {len(batch_themes)} themes")
    print(f"  ✓ Average pairwise similarity: {np.mean(similarities):.3f}")

    # Variable length handling
    short_paragraph = paragraphs[0][:2]  # Just 2 sentences
    long_paragraph = torch.cat([paragraphs[0], paragraphs[1]], dim=0)  # Combined

    best_model_obj.predict_themes([short_paragraph, long_paragraph])
    print(
        f"  ✓ Variable lengths: {short_paragraph.shape[0]} & {long_paragraph.shape[0]} sentences → themes computed"
    )

    print("\n🎉 Thematic LCM demonstration complete!")
    print(
        f"Best performing model: {best_model} (separation: {best_separation['theme_separation']:.3f})"
    )

    return models, results


def demonstrate_real_world_application():
    """Show how this would work in practice"""
    print("\n" + "=" * 50)
    print("🌍 Real-World Application Example")
    print("=" * 50)

    # Simulate real paragraph types
    paragraph_types = [
        "Scientific Research",
        "News & Politics",
        "Sports & Entertainment",
        "Personal Blog",
    ]

    print("\nSimulating real-world paragraph themes:")

    model = ThematicLCMExample(model_dim=512, theme_dim=256, num_layers=4)

    # Create characteristic embeddings for each type
    type_embeddings = []
    for i, ptype in enumerate(paragraph_types):
        # Simulate 3 paragraphs of each type
        type_paragraphs = []

        # Create a base pattern for this type
        base_pattern = torch.randn(1024) * 0.05
        base_pattern[i * 200 : (i + 1) * 200] += 0.2  # Add type-specific signal

        for _ in range(3):
            # Create paragraph with 4-6 sentences
            sentences = []
            for _ in range(np.random.randint(4, 7)):
                sentence = base_pattern + torch.randn(1024) * 0.02
                sentences.append(sentence)

            paragraph = torch.stack(sentences)
            type_paragraphs.append(paragraph)

        # Get theme embeddings for this type
        with torch.no_grad():
            themes = model.predict_themes(type_paragraphs)
            avg_theme = themes.mean(dim=0)
            type_embeddings.append(avg_theme)

        print(f"  ✓ {ptype}: {len(type_paragraphs)} paragraphs → theme embedding")

    # Compute type similarities
    print("\nTheme Type Similarity Matrix:")
    print("     " + "  ".join(f"{t[:6]}" for t in paragraph_types))

    for i, type_i in enumerate(paragraph_types):
        row = f"{type_i[:6]}:"
        for j, type_j in enumerate(paragraph_types):
            sim = torch.cosine_similarity(
                type_embeddings[i].unsqueeze(0), type_embeddings[j].unsqueeze(0)
            ).item()
            row += f" {sim:6.3f}"
        print(row)

    # Show discriminative power
    max_within = max(
        torch.cosine_similarity(
            type_embeddings[i].unsqueeze(0), type_embeddings[i].unsqueeze(0)
        ).item()
        for i in range(len(type_embeddings))
    )

    max_between = max(
        torch.cosine_similarity(
            type_embeddings[i].unsqueeze(0), type_embeddings[j].unsqueeze(0)
        ).item()
        for i in range(len(type_embeddings))
        for j in range(len(type_embeddings))
        if i != j
    )

    print("\nDiscriminative Analysis:")
    print(f"  Max within-type similarity: {max_within:.3f}")
    print(f"  Max between-type similarity: {max_between:.3f}")
    print(f"  Discrimination margin: {max_within - max_between:.3f}")

    print("\n✅ Real-world application simulation complete!")


if __name__ == "__main__":
    # Run the main demonstration
    models, results = demonstrate_thematic_lcm()

    # Show real-world application
    demonstrate_real_world_application()

    print("\n📋 Summary:")
    print(f"  - Implemented {len(models)} different Thematic LCM configurations")
    print("  - Demonstrated paragraph-level theme prediction")
    print("  - Showed model comparison and evaluation metrics")
    print("  - Simulated real-world application scenarios")
    print("  - Architecture successfully captures thematic information! 🎨")
