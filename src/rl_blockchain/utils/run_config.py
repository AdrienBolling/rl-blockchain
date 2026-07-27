"""Persist a run's arguments next to its checkpoints.

A checkpoint stores weights, not the architecture that produced them. Restoring
into the wrong ``--gat-arch`` / ``--n-nodes`` fails, but only later and with an
opaque flax shape error. Writing the arguments beside the checkpoint lets ``eval``
rebuild the exact model without the user having to remember the training flags.
"""

from __future__ import annotations

import json
import logging
import pathlib
from argparse import Namespace
from enum import Enum
from typing import Any, Union

from rl_blockchain.scripts.parser import UpdateDistStrat, UpdateValStrat

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "run_config.json"

#: Arguments that determine the model/env structure, i.e. the ones a checkpoint
#: cannot be restored without. Everything else in the run config is informational.
MODEL_CONFIG_KEYS = (
    "env", "n_nodes", "gat_arch", "voting_nodes", "reward_weights",
    "update_params", "next_edge_type", "ref_map_file", "horizon",
    "gini_reward_mode", "gamma",
)

_ENUM_KEYS = {"update_params": UpdateValStrat, "next_edge_type": UpdateDistStrat}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def save_run_config(chkpt_dir: Union[str, pathlib.Path], args: Namespace) -> pathlib.Path:
    """Write ``args`` as JSON inside ``chkpt_dir``. Returns the file path."""
    path = pathlib.Path(chkpt_dir) / CONFIG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: _jsonable(v) for k, v in vars(args).items()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def load_run_config(chkpt_dir: Union[str, pathlib.Path]) -> dict:
    """Return the raw saved run config dict, or ``{}`` if the run predates this file.

    Unlike :func:`apply_model_config` (which only restores the model/env subset onto
    ``args``), this exposes *every* saved training flag -- useful for labelling eval
    outputs with things like ``gini_lambda`` that don't rebuild the model.
    """
    path = pathlib.Path(chkpt_dir) / CONFIG_FILENAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def apply_model_config(args: Namespace, chkpt_dir: Union[str, pathlib.Path]) -> bool:
    """Overwrite the model/env args on ``args`` with the ones the run was trained with.

    Returns False (leaving ``args`` untouched) when the run predates this file, so
    older checkpoints keep working with explicitly-passed flags.
    """
    path = pathlib.Path(chkpt_dir) / CONFIG_FILENAME
    if not path.is_file():
        logger.warning(
            f"No {CONFIG_FILENAME} in {chkpt_dir}; using the model flags given on the "
            f"command line. A mismatch will fail when the checkpoint is restored.")
        return False

    saved = json.loads(path.read_text())
    applied = {}
    for key in MODEL_CONFIG_KEYS:
        if key not in saved:
            continue
        value = saved[key]
        if key in _ENUM_KEYS and value is not None:
            value = _ENUM_KEYS[key].parse(str(value))
        elif key == "ref_map_file" and value is not None:
            value = pathlib.Path(value)
        setattr(args, key, value)
        applied[key] = value
    logger.info(f"Model config restored from {path}: {applied}")
    return True
