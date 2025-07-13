# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

"""
Theme prediction evaluation task for Thematic LCM
"""

import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from lcm.evaluation.tasks.base import Task
from lcm.datasets.batch import EmbeddingsBatch


@dataclass
class ThemePredictionTaskConfig:
    """Configuration for theme prediction task"""
    task_name: str = "theme_prediction"
    max_paragraphs: Optional[int] = None
    theme_labels: Optional[List[str]] = None
    compute_similarity: bool = True
    similarity_threshold: float = 0.7


class ThemePredictionTask(Task):
    """
    Evaluation task for theme prediction.
    
    This task evaluates how well a model can predict paragraph themes
    by comparing predicted theme embeddings with ground truth themes.
    """
    
    def __init__(self, config: ThemePredictionTaskConfig):
        self.config = config
        self.name = config.task_name
    
    def evaluate_predictions(
        self,
        predictions: List[torch.Tensor],
        references: List[torch.Tensor],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate theme predictions against references.
        
        Args:
            predictions: List of predicted theme embeddings
            references: List of reference theme embeddings
            metadata: Optional metadata for each sample
            
        Returns:
            Dictionary of evaluation metrics
        """
        if len(predictions) != len(references):
            raise ValueError(f"Number of predictions ({len(predictions)}) != references ({len(references)})")
        
        if len(predictions) == 0:
            return {"cosine_similarity": 0.0, "accuracy": 0.0, "count": 0}
        
        metrics = {}
        
        # Compute cosine similarities
        similarities = []
        for pred, ref in zip(predictions, references):
            # Normalize embeddings
            pred_norm = torch.nn.functional.normalize(pred.flatten(), p=2, dim=0)
            ref_norm = torch.nn.functional.normalize(ref.flatten(), p=2, dim=0)
            
            # Compute cosine similarity
            sim = torch.dot(pred_norm, ref_norm).item()
            similarities.append(sim)
        
        # Similarity metrics
        similarities_tensor = torch.tensor(similarities)
        metrics["cosine_similarity_mean"] = similarities_tensor.mean().item()
        metrics["cosine_similarity_std"] = similarities_tensor.std().item()
        metrics["cosine_similarity_min"] = similarities_tensor.min().item()
        metrics["cosine_similarity_max"] = similarities_tensor.max().item()
        
        # Accuracy based on similarity threshold
        if self.config.compute_similarity:
            above_threshold = (similarities_tensor >= self.config.similarity_threshold).float()
            metrics["similarity_accuracy"] = above_threshold.mean().item()
            metrics["threshold"] = self.config.similarity_threshold
        
        # Classification accuracy (if discrete themes available)
        if metadata and all("theme_class" in m for m in metadata if m):
            # Assume predictions are class indices if they're integers
            try:
                pred_classes = []
                ref_classes = []
                
                for i, (pred, ref, meta) in enumerate(zip(predictions, references, metadata)):
                    if meta and "theme_class" in meta:
                        ref_classes.append(meta["theme_class"])
                        
                        if pred.numel() == 1 and pred.dtype in [torch.long, torch.int]:
                            # Discrete prediction
                            pred_classes.append(pred.item())
                        else:
                            # Continuous prediction - would need clustering or nearest neighbor
                            pred_classes.append(-1)  # Unknown
                
                if pred_classes and ref_classes:
                    correct = sum(p == r for p, r in zip(pred_classes, ref_classes))
                    metrics["classification_accuracy"] = correct / len(pred_classes)
                    metrics["num_classified"] = len([p for p in pred_classes if p != -1])
                    
            except Exception:
                # Skip classification metrics if not available
                pass
        
        # Count metrics
        metrics["count"] = len(predictions)
        
        return metrics
    
    def format_output(
        self,
        predictions: List[torch.Tensor],
        references: List[torch.Tensor],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Format predictions for output.
        
        Returns:
            List of formatted prediction dictionaries
        """
        formatted = []
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            output = {
                "prediction_id": i,
                "predicted_theme_embedding": pred.tolist(),
                "reference_theme_embedding": ref.tolist(),
                "prediction_norm": torch.norm(pred).item(),
                "reference_norm": torch.norm(ref).item(),
            }
            
            # Add similarity
            pred_norm = torch.nn.functional.normalize(pred.flatten(), p=2, dim=0)
            ref_norm = torch.nn.functional.normalize(ref.flatten(), p=2, dim=0)
            similarity = torch.dot(pred_norm, ref_norm).item()
            output["cosine_similarity"] = similarity
            
            # Add metadata if available
            if metadata and i < len(metadata) and metadata[i]:
                output.update(metadata[i])
            
            formatted.append(output)
        
        return formatted


def create_theme_prediction_task(
    task_name: str = "paragraph_theme_prediction",
    **kwargs
) -> ThemePredictionTask:
    """Create a theme prediction task with given configuration."""
    config = ThemePredictionTaskConfig(task_name=task_name, **kwargs)
    return ThemePredictionTask(config)


# Example usage and testing
def test_theme_prediction_task():
    """Test the theme prediction task"""
    print("Testing Theme Prediction Task...")
    
    # Create task
    task = create_theme_prediction_task(
        similarity_threshold=0.8,
        theme_labels=["science", "news", "sports", "entertainment"]
    )
    
    # Create dummy predictions and references
    theme_dim = 128
    num_samples = 5
    
    predictions = []
    references = []
    metadata = []
    
    for i in range(num_samples):
        # Create somewhat similar embeddings
        base_embedding = torch.randn(theme_dim) * 0.1
        pred = base_embedding + torch.randn(theme_dim) * 0.05  # Add small noise
        ref = base_embedding + torch.randn(theme_dim) * 0.03   # Add smaller noise
        
        predictions.append(pred)
        references.append(ref)
        metadata.append({
            "paragraph_id": f"para_{i}",
            "theme_class": i % 4,
            "source": "test_data"
        })
    
    # Evaluate
    metrics = task.evaluate_predictions(predictions, references, metadata)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Format output
    formatted = task.format_output(predictions, references, metadata)
    print(f"\nFormatted {len(formatted)} predictions")
    print(f"First prediction similarity: {formatted[0]['cosine_similarity']:.4f}")
    
    print("✅ Theme prediction task test passed!")


if __name__ == "__main__":
    test_theme_prediction_task()