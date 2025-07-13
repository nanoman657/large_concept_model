#  Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

"""
LCM: Modular and Extensible Reasoning in an Embedding Space
Code base for training different LCM models.
"""

try:
    from fairseq2 import setup_extensions
    from fairseq2.assets import default_asset_store

    def setup_fairseq2() -> None:
        default_asset_store.add_package_metadata_provider("lcm.cards")

    # This call activates setup_fairseq2 and potentially other extensions,
    setup_extensions()
except ImportError:
    # Fallback for newer fairseq2 versions that don't have setup_extensions
    try:
        from fairseq2.assets import default_asset_store

        def setup_fairseq2() -> None:
            default_asset_store.add_package_metadata_provider("lcm.cards")

        setup_fairseq2()
    except ImportError:
        # Final fallback - define dummy function
        def setup_fairseq2() -> None:
            pass


__version__ = "0.1.0.dev0"
