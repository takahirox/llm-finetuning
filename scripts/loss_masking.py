"""Project-owned loss hook for the inspected MLX-LM 0.32.0 trainer.

No installed files are modified. The only objective difference between variants
is the strict end bound; target positions and prompt offsets are unchanged.
"""

import importlib.metadata
from pathlib import Path

VARIANTS = ('corrected', 'legacy')
UPSTREAM_HASHES = {
    'tuner/trainer.py': '0b6027fdb11803e951ceee52c83a81f82daa922ea8c6b7ca6888953461064e8e',
    'tuner/datasets.py': 'dcd675376d68d1813261c8389236b586b16b9a58a5ea80be8eb535bb9d0e9689',
    'lora.py': 'e366eec6ae2f8a72446f1af65faf189a92a01ca509fb0cafb3fa54b89b4291f7',
}


def validate_variant(variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown loss-mask variant: {variant}')
    return variant


def target_mask(lengths, width, variant, xp):
    """Position p labels batch[:, p]; length is an exclusive token-array bound."""
    validate_variant(variant)
    positions = xp.arange(1, width + 1)
    end = positions < lengths[:, 1:] if variant == 'corrected' else positions <= lengths[:, 1:]
    return xp.logical_and(positions >= lengths[:, 0:1], end)


def make_loss(variant):
    validate_variant(variant)
    import mlx.core as mx
    import mlx.nn as nn

    def loss(model, batch, lengths):
        targets = batch[:, 1:]
        mask = target_mask(lengths, targets.shape[1], variant, mx)
        ce = nn.losses.cross_entropy(model(batch[:, :-1]), targets) * mask
        ntoks = mask.sum()
        return ce.astype(mx.float32).sum() / ntoks, ntoks

    return loss


def integration_info(variant):
    from scripts.experiment import file_hash
    validate_variant(variant)
    if importlib.metadata.version('mlx-lm') != '0.32.0':
        raise ValueError('Loss hook requires pinned mlx-lm==0.32.0; run uv sync --locked.')
    import mlx_lm
    root = Path(mlx_lm.__file__).parent
    actual = {name: file_hash(root / name) for name in UPSTREAM_HASHES}
    if actual != UPSTREAM_HASHES:
        raise ValueError('Installed MLX-LM source differs from the inspected pin; refusing unsafe loss integration.')
    return {'variant': variant, 'implementation': 'exclusive-target-bound-v1',
            'upstream_files': actual, 'hook_sha256': file_hash(Path(__file__)),
            'runner_sha256': file_hash(Path(__file__).with_name('masked_lora.py'))}
