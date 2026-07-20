"""Allocate run names from a counter on disk rather than from the wandb API.

wandb runs offline here, so counting runs over the network either fails or
returns 0 -- and every offline run then reused the name ``<prefix>_0``.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from argparse import Namespace

logger = logging.getLogger(__name__)

COUNTER_FILENAME = "run_counter.json"


def project_dir(args: Namespace) -> pathlib.Path:
    """The directory holding every run of one entity/project pair."""
    return pathlib.Path(args.checkpoint_dir) / f"{args.wandb_entity}_{args.wandb_project}"


def _seed_from_disk(directory: pathlib.Path) -> int:
    """Continue after the highest run present, so a pre-counter project keeps counting up."""
    highest = -1
    for child in directory.glob("*"):
        if not child.is_dir():
            continue
        suffix = child.name.rsplit("_", 1)[-1]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


def _write_number(path: pathlib.Path, number: int) -> None:
    # Temp file + rename: a run killed mid-write must not truncate the counter.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"next_run_number": number}, indent=2))
    os.replace(tmp, path)


def allocate_run_name(args: Namespace) -> str:
    """Reserve and return the next run name, e.g. ``train_206``."""
    directory = project_dir(args)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / COUNTER_FILENAME

    try:
        number = json.loads(path.read_text())["next_run_number"]
    except FileNotFoundError:
        number = _seed_from_disk(directory)
        logger.info(f"No {COUNTER_FILENAME} yet; seeding run numbering at {number}.")
    except (json.JSONDecodeError, KeyError, TypeError):
        number = _seed_from_disk(directory)
        logger.warning(f"Unreadable {path}; reseeding run numbering at {number}.")

    # An existing dir means the counter fell behind (restored backup, older run).
    while (directory / f"{args.logging_prefix}_{number}").exists():
        number += 1

    _write_number(path, number + 1)
    return f"{args.logging_prefix}_{number}"
