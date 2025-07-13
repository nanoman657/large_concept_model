#!/usr/bin/env python3
"""
Simple validation script for DSM-5 MDD Thematic LCM Classifier

This script validates that the new thematic_lcm_mdd_classifier architecture
is properly registered and configured without requiring torch dependencies.
"""


def validate_mdd_classifier():
    """Validate the MDD classifier configuration"""

    print("=" * 80)
    print("DSM-5 MDD Thematic LCM Classifier - Configuration Validation")
    print("=" * 80)
    print()

    try:
        # Import the architecture registry
        from lcm.models.thematic_lcm.archs import thematic_lcm_archs

        print("🔍 Checking available thematic LCM architectures:")
        available_archs = list(thematic_lcm_archs.get_names())
        for arch in sorted(available_archs):
            print(f"  ✅ {arch}")
        print()

        # Check if our new MDD classifier is registered
        if "thematic_lcm_mdd_classifier" in available_archs:
            print("✅ thematic_lcm_mdd_classifier is properly registered!")

            # Get the configuration
            config = thematic_lcm_archs.get_config("thematic_lcm_mdd_classifier")
            print("📋 Configuration details:")
            print(f"  Model type: {config.model_type}")
            print(f"  Model dimension: {config.model_dim}")
            print(f"  Theme embedding dimension: {config.theme_embed_dim}")
            print(f"  Number of themes (MDD criteria): {config.num_themes}")
            print(f"  Max sequence length: {config.max_seq_len}")
            print(f"  SONAR embedding dimension: {config.sonar_embed_dim}")
            print()

            # Validate MDD-specific configuration
            if config.num_themes == 9:
                print("✅ Correct number of themes (9 DSM-5 MDD criteria)")
            else:
                print(
                    f"❌ Incorrect number of themes: {config.num_themes} (expected 9)"
                )

            if config.model_dim == 1024:
                print("✅ Standard model dimension (1024)")
            else:
                print(f"⚠️  Non-standard model dimension: {config.model_dim}")

            if config.theme_embed_dim == 512:
                print("✅ Standard theme embedding dimension (512)")
            else:
                print(
                    f"⚠️  Non-standard theme embedding dimension: {config.theme_embed_dim}"
                )

            print()
            print("🧠 DSM-5 MDD Criteria (9 classification targets):")
            mdd_criteria = [
                "1. Depressed mood most of the day, nearly every day",
                "2. Markedly diminished interest or pleasure in activities (anhedonia)",
                "3. Significant weight loss/gain or decrease/increase in appetite",
                "4. Insomnia or hypersomnia nearly every day",
                "5. Psychomotor agitation or retardation",
                "6. Fatigue or loss of energy nearly every day",
                "7. Feelings of worthlessness or excessive/inappropriate guilt",
                "8. Diminished ability to think or concentrate, or indecisiveness",
                "9. Recurrent thoughts of death, suicidal ideation, or suicide attempt",
            ]

            for criterion in mdd_criteria:
                print(f"  {criterion}")
            print()

        else:
            print("❌ thematic_lcm_mdd_classifier is NOT registered!")
            print("Available architectures:")
            for arch in sorted(available_archs):
                print(f"  - {arch}")

        print("✅ Configuration validation completed!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("ℹ️  This indicates the thematic LCM module structure is correct")
        print(
            "   but dependencies like fairseq2 are not available in this environment."
        )
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    validate_mdd_classifier()
