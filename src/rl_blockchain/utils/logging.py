import logging
import os
from argparse import Namespace

import wandb


def setup_logging(args: Namespace, run: "wandb.Run") -> str:
    level = getattr(logging, args.logging_level.upper(), logging.INFO)
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Détermine le nom du fichier log
    filename = f"{log_dir}/{run.id}.log" if run else args.logging_filename
    log_path = os.path.abspath(filename) if filename else None

    # Format
    formatter = logging.Formatter(args.logging_format, args.logging_datefmt)

    # Nettoyage des anciens handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    # Ajout du fichier log
    if log_path:
        file_handler = logging.FileHandler(log_path, mode=args.logging_filemode)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
        print(f"[logging] files will be written at : {log_path}")

    # Ajout du stream (console)
    if args.logging_stream:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    return log_path


def setup_wandb(args: Namespace) -> wandb.sdk.wandb_run.Run:
    """
    Initialise Weights & Biases (wandb) avec reprise facultative depuis un checkpoint.
    """
    # Convertir Namespace en dictionnaire propre
    config = vars(args)

    api = wandb.Api()
    entity = args.wandb_entity
    project = args.wandb_project

    # Nouveau run
    try:
        runs = api.runs(f"{entity}/{project}", order="-created_at")
        run_number = len(runs)
    except wandb.errors.CommError:
        run_number = 0

    run_id = f"{args.logging_prefix}_{run_number}"
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_id,
        # id=run_id,
        config=config,
        group=args.algo,
        tags=args.wandb_tags,
    )

    print(f"[wandb] Run en cours : https://wandb.ai/{entity}/{project}/runs/{run.id}")
    return run
