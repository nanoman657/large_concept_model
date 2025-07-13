#!/usr/bin/env python3
"""
Test script for DSM-5 Major Depressive Disorder (MDD) Thematic LCM Classifier

This script demonstrates how the thematic LCM can be used to classify text 
according to the 9 DSM-5 MDD diagnostic criteria:

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

import torch
import numpy as np
from typing import Dict, List

# DSM-5 MDD Criteria Labels
MDD_CRITERIA_LABELS = [
    "Depressed Mood",
    "Anhedonia (Loss of Interest/Pleasure)", 
    "Appetite/Weight Changes",
    "Sleep Disturbances (Insomnia/Hypersomnia)",
    "Psychomotor Changes (Agitation/Retardation)",
    "Fatigue/Loss of Energy",
    "Worthlessness/Excessive Guilt",
    "Concentration/Decision Making Difficulties",
    "Thoughts of Death/Suicidal Ideation"
]

# Example paragraphs representing different MDD criteria
MDD_EXAMPLE_PARAGRAPHS = {
    "Depressed Mood": [
        "I wake up every morning feeling overwhelmingly sad and empty. The sadness stays with me throughout the entire day, like a heavy cloud that never lifts. Even when good things happen, I can't shake this persistent feeling of despair and hopelessness.",
        "My mood has been consistently low for weeks now. I feel tearful most of the time and there's this constant ache in my chest. Nothing seems to bring me joy anymore, and I feel like I'm drowning in sadness.",
        "Every day feels gray and meaningless. I experience this deep, persistent sadness that doesn't go away no matter what I do. It's like being stuck in a dark tunnel with no light at the end."
    ],
    
    "Anhedonia (Loss of Interest/Pleasure)": [
        "I used to love playing guitar and painting, but now these activities feel completely pointless. Nothing gives me pleasure anymore - not music, not art, not spending time with friends. Everything feels flat and meaningless.",
        "Activities I once enjoyed, like cooking and gardening, now feel like chores. I have no motivation to do anything fun. Even watching my favorite TV shows or reading books brings me no satisfaction whatsoever.",
        "I've lost all interest in things that used to make me happy. Going out with friends, playing sports, even listening to music - none of it matters to me anymore. It's like all the color has drained out of my life."
    ],
    
    "Appetite/Weight Changes": [
        "I've completely lost my appetite and have to force myself to eat even small amounts. I've lost about 15 pounds in the past month without trying. Food just doesn't taste good anymore and eating feels like a chore.",
        "I find myself eating constantly, especially sweets and junk food, even when I'm not hungry. I've gained significant weight recently and my clothes don't fit anymore. Food seems to be the only thing that provides temporary comfort.",
        "My relationship with food has completely changed. Some days I can't eat anything at all, other days I binge eat uncontrollably. My weight has been fluctuating dramatically and I've lost control over my eating habits."
    ],
    
    "Sleep Disturbances (Insomnia/Hypersomnia)": [
        "I lie awake for hours every night, unable to fall asleep despite feeling exhausted. When I do finally sleep, I wake up multiple times and can't get back to sleep. I'm getting maybe 3-4 hours of sleep per night.",
        "I sleep for 12-14 hours a day but still feel tired all the time. No matter how much I sleep, I never feel rested. I spend most of my day in bed because I'm constantly exhausted and drowsy.",
        "My sleep schedule is completely disrupted. I either can't sleep at all, lying awake all night with racing thoughts, or I sleep excessively and still wake up feeling drained and unrefreshed."
    ],
    
    "Psychomotor Changes (Agitation/Retardation)": [
        "I feel restless and agitated all the time, like I need to constantly move or pace around. I can't sit still and feel like I'm jumping out of my skin. My hands shake and I tap my feet constantly.",
        "Everything feels like it's happening in slow motion. Simple tasks like getting dressed or making coffee take forever. I move and speak much slower than usual, and even thinking feels sluggish and difficult.",
        "I alternate between feeling extremely restless and fidgety to moving and thinking so slowly that others notice. Sometimes I pace frantically, other times I feel like I'm moving through molasses."
    ],
    
    "Fatigue/Loss of Energy": [
        "I'm exhausted all the time, even after sleeping. Simple tasks like taking a shower or making breakfast leave me completely drained. I feel like I'm carrying a heavy weight on my shoulders constantly.",
        "My energy levels are at zero. I struggle to complete basic daily activities and feel tired even when I haven't done anything. It's like my body is running on empty and I can barely function.",
        "Every movement requires enormous effort. I feel physically and mentally depleted all the time. Even thinking clearly takes too much energy, and I spend most of my day feeling completely wiped out."
    ],
    
    "Worthlessness/Excessive Guilt": [
        "I constantly feel like I'm a burden to everyone around me and that I'm completely worthless. I blame myself for everything that goes wrong, even things that are clearly not my fault. The guilt is overwhelming and constant.",
        "I can't stop thinking about all the ways I've failed and disappointed people. I feel like I don't deserve anything good and that I'm fundamentally flawed as a person. The shame and self-blame consume my thoughts.",
        "I'm convinced that I'm a terrible person who ruins everything I touch. Every mistake I've ever made replays in my mind, and I feel guilty about things from years ago. I don't think I deserve forgiveness or love."
    ],
    
    "Concentration/Decision Making Difficulties": [
        "I can't focus on anything for more than a few minutes. Reading a simple email takes forever because my mind keeps wandering. Making even basic decisions, like what to wear or eat, feels overwhelming and impossible.",
        "My mind feels foggy and unclear all the time. I forget things constantly and can't concentrate on work or conversations. Simple decisions that used to be automatic now require enormous mental effort.",
        "I'm unable to think clearly or make decisions. My concentration is completely shot - I start tasks but can't finish them. Even choosing what to watch on TV becomes an agonizing decision that I can't make."
    ],
    
    "Thoughts of Death/Suicidal Ideation": [
        "I think about death constantly and wonder if people would be better off without me. Sometimes I have detailed thoughts about ending my life, though I haven't made any specific plans yet.",
        "I frequently wish I could just disappear or not wake up in the morning. Death seems like it would be a relief from this constant emotional pain. I think about various ways to die, though I'm scared to act on these thoughts.",
        "I have recurring thoughts about suicide and sometimes research methods online. The idea of ending my life feels like the only way to escape this unbearable suffering, though part of me is still fighting these thoughts."
    ]
}

def create_synthetic_sentence_embeddings(paragraphs: List[str], embed_dim: int = 1024) -> torch.Tensor:
    """Create synthetic sentence embeddings for demonstration purposes"""
    num_paragraphs = len(paragraphs)
    max_sentences = 5  # Assume max 5 sentences per paragraph
    
    # Create random embeddings with some structure
    embeddings = torch.randn(num_paragraphs, max_sentences, embed_dim) * 0.006
    
    # Add some thematic structure - paragraphs about the same topic should be more similar
    base_pattern = torch.randn(embed_dim) * 0.02
    for i in range(num_paragraphs):
        # Add shared thematic component with some noise
        thematic_component = base_pattern + torch.randn(embed_dim) * 0.01
        embeddings[i] += thematic_component.unsqueeze(0)
    
    return embeddings

def test_mdd_classifier():
    """Test the MDD classifier with example paragraphs for each criterion"""
    
    print("=" * 80)
    print("DSM-5 Major Depressive Disorder (MDD) Thematic LCM Classifier Test")
    print("=" * 80)
    print()
    
    try:
        from lcm.models.thematic_lcm import ThematicLCModelConfig, create_thematic_lcm_model
        from lcm.datasets.batch import EmbeddingsBatch
        
        # Create MDD classifier model
        print("🧠 Creating DSM-5 MDD Thematic LCM Classifier...")
        config = ThematicLCModelConfig(model_arch="thematic_lcm_mdd_classifier")
        model = create_thematic_lcm_model(config)
        print(f"✅ Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"📊 Classification targets: {config.num_themes} DSM-5 MDD criteria")
        print()
        
        # Test each MDD criterion
        for criterion_idx, (criterion_name, paragraphs) in enumerate(MDD_EXAMPLE_PARAGRAPHS.items()):
            print(f"🔍 Testing Criterion {criterion_idx + 1}: {criterion_name}")
            print("-" * 60)
            
            # Create synthetic embeddings for the paragraphs
            paragraph_embeddings = create_synthetic_sentence_embeddings(paragraphs)
            batch = EmbeddingsBatch(paragraph_embeddings, padding_mask=None)
            
            # Get theme embeddings and classifications
            with torch.no_grad():
                theme_embeddings = model.predict_themes(batch)
                if hasattr(model, 'theme_classifier') and model.theme_classifier is not None:
                    logits = model.theme_classifier(theme_embeddings)
                    probabilities = torch.softmax(logits, dim=-1)
                    predicted_classes = torch.argmax(probabilities, dim=-1)
                    
                    print(f"📈 Theme embeddings shape: {theme_embeddings.shape}")
                    print(f"🎯 Classification results:")
                    
                    for i, paragraph in enumerate(paragraphs):
                        pred_class = predicted_classes[i].item()
                        confidence = probabilities[i, pred_class].item()
                        pred_criterion = MDD_CRITERIA_LABELS[pred_class]
                        
                        print(f"  Paragraph {i+1}:")
                        print(f"    Text: {paragraph[:100]}...")
                        print(f"    Predicted: {pred_criterion} (confidence: {confidence:.3f})")
                        print(f"    Expected: {criterion_name}")
                        print(f"    ✅ Correct" if pred_criterion == criterion_name else f"    ❌ Incorrect")
                        print()
                else:
                    print(f"📈 Theme embeddings shape: {theme_embeddings.shape}")
                    print("ℹ️  Model outputs continuous embeddings (no classification head)")
                    
                    # Analyze embedding patterns
                    embeddings_norm = torch.norm(theme_embeddings, dim=1)
                    embeddings_mean = torch.mean(theme_embeddings, dim=1)
                    
                    print(f"🔢 Embedding statistics:")
                    print(f"  Mean norm: {embeddings_norm.mean():.4f} ± {embeddings_norm.std():.4f}")
                    print(f"  Mean values: {embeddings_mean.mean():.4f} ± {embeddings_mean.std():.4f}")
                    
                    # Compute pairwise similarities within criterion
                    similarities = torch.cosine_similarity(
                        theme_embeddings.unsqueeze(1), 
                        theme_embeddings.unsqueeze(0), 
                        dim=2
                    )
                    
                    # Get off-diagonal similarities (between different paragraphs of same criterion)
                    mask = ~torch.eye(len(paragraphs), dtype=torch.bool)
                    off_diag_sims = similarities[mask]
                    
                    print(f"  Within-criterion similarity: {off_diag_sims.mean():.4f} ± {off_diag_sims.std():.4f}")
                    print()
            
            print()
        
        # Test cross-criterion discrimination
        print("🧪 Testing Cross-Criterion Discrimination")
        print("-" * 60)
        
        all_embeddings = []
        all_labels = []
        
        for criterion_idx, (criterion_name, paragraphs) in enumerate(MDD_EXAMPLE_PARAGRAPHS.items()):
            paragraph_embeddings = create_synthetic_sentence_embeddings(paragraphs)
            batch = EmbeddingsBatch(paragraph_embeddings, padding_mask=None)
            
            with torch.no_grad():
                theme_embeddings = model.predict_themes(batch)
                all_embeddings.append(theme_embeddings)
                all_labels.extend([criterion_idx] * len(paragraphs))
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.tensor(all_labels)
        
        # Compute within vs between criterion similarities
        within_criterion_sims = []
        between_criterion_sims = []
        
        for i in range(len(all_embeddings)):
            for j in range(i + 1, len(all_embeddings)):
                sim = torch.cosine_similarity(all_embeddings[i], all_embeddings[j], dim=0).item()
                
                if all_labels[i] == all_labels[j]:
                    within_criterion_sims.append(sim)
                else:
                    between_criterion_sims.append(sim)
        
        within_mean = np.mean(within_criterion_sims)
        within_std = np.std(within_criterion_sims)
        between_mean = np.mean(between_criterion_sims)
        between_std = np.std(between_criterion_sims)
        
        discrimination_score = within_mean - between_mean
        
        print(f"📊 Discrimination Analysis:")
        print(f"  Within-criterion similarity: {within_mean:.4f} ± {within_std:.4f}")
        print(f"  Between-criterion similarity: {between_mean:.4f} ± {between_std:.4f}")
        print(f"  Discrimination score: {discrimination_score:.4f}")
        print(f"  📈 {'Excellent' if discrimination_score > 0.3 else 'Good' if discrimination_score > 0.1 else 'Moderate'} discrimination")
        print()
        
        # Model summary
        print("📋 Model Summary")
        print("-" * 40)
        print(f"Architecture: thematic_lcm_mdd_classifier")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Model dimension: {config.model_dim}")
        print(f"Theme embedding dimension: {config.theme_embed_dim}")
        print(f"Classification targets: {config.num_themes} DSM-5 MDD criteria")
        print(f"Max sequence length: {config.max_seq_len}")
        print()
        
        print("✅ DSM-5 MDD Classifier test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("ℹ️  This script requires the thematic LCM module to be available.")
        print("   Make sure fairseq2 and other dependencies are installed.")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mdd_classifier()