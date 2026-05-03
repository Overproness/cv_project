"""flame_pytorch — a lightweight PyTorch FLAME 3D face model.

Exports ``FLAME`` and ``get_config`` so that the rest of Aura3D can do:

    from flame_pytorch import FLAME, get_config

The package ships no model weights.  You must download the FLAME 2020
assets (requires free registration at https://flame.is.tue.mpg.de/) and
point ``config.flame_model_path`` at ``generic_model.pkl``.
"""

from .flame import FLAME, get_config  # noqa: F401

__all__ = ["FLAME", "get_config"]
