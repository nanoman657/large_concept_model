# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import torch
from typing import Dict, List, Optional, Union

from lcm.datasets.batch import EmbeddingsBatch
from lcm.evaluation.predictors.lcm import LCMPredictor
from lcm.models.thematic_lcm import ThematicLCModel


class ThematicLCMPredictor(LCMPredictor):
    """Predictor for Thematic LCM models that predict paragraph themes."""

    model: ThematicLCModel

    def __init__(
        self,
        model_card: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(model_card, device, dtype)

    def predict_themes(
        self,
        paragraphs: List[List[torch.Tensor]],
        return_probabilities: bool = False,
        batch_size: int = 1,
    ) -> List[Union[torch.Tensor, int]]:
        """
        Predict themes for a list of paragraphs.
        
        Args:
            paragraphs: List of paragraphs, where each paragraph is a list of sentence embeddings
            return_probabilities: If True and model has classifier, return probabilities
            batch_size: Batch size for processing
            
        Returns:
            List of theme predictions (embeddings, class indices, or probabilities)
        """
        predictions = []
        
        for i in range(0, len(paragraphs), batch_size):
            batch_paragraphs = paragraphs[i:i + batch_size]
            
            # Convert to batch format
            batch_seqs = []
            for paragraph in batch_paragraphs:
                # Stack sentence embeddings for this paragraph
                if len(paragraph) > 0:
                    para_tensor = torch.stack(paragraph, dim=0)
                else:
                    # Empty paragraph - create dummy embedding
                    para_tensor = torch.zeros(1, self.model.config.sonar_embed_dim)
                batch_seqs.append(para_tensor)
            
            # Pad sequences to same length
            max_len = max(seq.size(0) for seq in batch_seqs)
            padded_seqs = []
            padding_masks = []
            
            for seq in batch_seqs:
                seq_len = seq.size(0)
                if seq_len < max_len:
                    # Pad sequence
                    pad_size = max_len - seq_len
                    padded_seq = torch.cat([
                        seq,
                        torch.zeros(pad_size, seq.size(1), dtype=seq.dtype, device=seq.device)
                    ], dim=0)
                    # Create padding mask (True for padded positions)
                    mask = torch.cat([
                        torch.zeros(seq_len, dtype=torch.bool),
                        torch.ones(pad_size, dtype=torch.bool)
                    ])
                else:
                    padded_seq = seq
                    mask = torch.zeros(seq_len, dtype=torch.bool)
                
                padded_seqs.append(padded_seq)
                padding_masks.append(mask)
            
            # Stack into batch
            batch_tensor = torch.stack(padded_seqs, dim=0)
            batch_mask = torch.stack(padding_masks, dim=0)
            
            # Move to device
            if self.device is not None:
                batch_tensor = batch_tensor.to(self.device)
                batch_mask = batch_mask.to(self.device)
            
            # Create batch object
            batch = EmbeddingsBatch(
                seqs=batch_tensor,
                padding_mask=batch_mask if batch_mask.any() else None,
            )
            
            # Get predictions
            with torch.no_grad():
                batch_predictions = self.model.predict_themes(
                    batch, return_probabilities=return_probabilities
                )
                
                # Convert to list
                if isinstance(batch_predictions, torch.Tensor):
                    if batch_predictions.dim() == 1:
                        # Class indices
                        predictions.extend(batch_predictions.cpu().tolist())
                    else:
                        # Embeddings or probabilities
                        predictions.extend([pred.cpu() for pred in batch_predictions])
                else:
                    predictions.extend(batch_predictions)
        
        return predictions

    def predict_paragraph_themes(
        self,
        texts: List[str],
        sentence_encoder: Optional[callable] = None,
        **kwargs
    ) -> List[Union[torch.Tensor, int]]:
        """
        Predict themes for text paragraphs.
        
        Args:
            texts: List of paragraph texts
            sentence_encoder: Function to encode sentences to embeddings
            **kwargs: Additional arguments for theme prediction
            
        Returns:
            List of theme predictions
        """
        if sentence_encoder is None:
            raise ValueError("sentence_encoder must be provided to encode text to embeddings")
        
        # Convert texts to sentence embeddings
        paragraphs = []
        for text in texts:
            # Split into sentences (simple approach)
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                sentences = [text]  # Fallback to full text
            
            # Encode sentences
            sentence_embeddings = []
            for sentence in sentences:
                embedding = sentence_encoder(sentence)
                if isinstance(embedding, torch.Tensor):
                    sentence_embeddings.append(embedding)
                else:
                    # Convert to tensor if needed
                    sentence_embeddings.append(torch.tensor(embedding))
            
            paragraphs.append(sentence_embeddings)
        
        return self.predict_themes(paragraphs, **kwargs)

    def generate_theme_descriptions(
        self,
        theme_embeddings: List[torch.Tensor],
        theme_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Union[str, torch.Tensor]]]:
        """
        Generate theme descriptions from embeddings.
        
        Args:
            theme_embeddings: List of theme embedding tensors
            theme_labels: Optional list of theme labels
            
        Returns:
            List of theme descriptions with embeddings and optional labels
        """
        descriptions = []
        for i, embedding in enumerate(theme_embeddings):
            desc = {
                "theme_id": i,
                "embedding": embedding,
                "embedding_norm": torch.norm(embedding).item(),
            }
            if theme_labels and i < len(theme_labels):
                desc["label"] = theme_labels[i]
            descriptions.append(desc)
        
        return descriptions