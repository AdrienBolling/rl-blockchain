import logging
from argparse import Namespace

import wandb

def setup_logging(args:Namespace, run : "wandb.Run") -> None:
    level = args.logging_level
    format = args.logging_format
    datefmt = args.logging_datefmt
    log_dir = args.log_dir
    filename = f"{log_dir}/run_{run.id}.log" if run else None
    filemode = args.logging_filemode
    stream = args.logging_stream

    if filename:
        logging.basicConfig(
            level=level,
            format=format,
            datefmt=datefmt,
            filename=filename,
            filemode=filemode,
        )
    if stream:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(format, datefmt)
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(level)

def setup_wandb(ARGS: Namespace)-> wandb.run:
    """
    Set up Weights & Biases (wandb) logging.

    Args:
        ARGS (argparse.Namespace): Command line arguments containing wandb settings.

    Returns:
        None
    """

    if ARGS.checkpoint is not None:
        # If we need to resume a training, get the name of the checkpoint
        chkpt_name = ARGS.checkpoint
        # If the checkpoint is 'latest', get the latest run id
        if chkpt_name == "latest":
            api = wandb.Api()
            try:
                runs = api.runs(
                    f"{ARGS.wandb_entity}/{ARGS.wandb_project}",
                    order="created_at",
                )
                chkpt_name = runs[-1].id
            except ValueError:
                # When the project does not exist yet, assume no runs
                chkpt_name = "run_0"

        # Resume the run
        run = wandb.init(
            project=ARGS.wandb_project,
            entity=ARGS.wandb_entity,
            id=chkpt_name,
            resume="must",
            config=ARGS,
            # job_type=ARGS.mode,
            group=ARGS.algo,
            tags=ARGS.wandb_tags,
        )

    else:
        api = wandb.Api()
        try:
            runs = api.runs(
                f"{ARGS.wandb_entity}/{ARGS.wandb_project}",
                order="created_at",
            )
            new_run_id = f"run_{len(runs)}"
        except ValueError:
            # When the project does not exist yet, assume no runs
            new_run_id = "run_0"
        run = wandb.init(
            project=ARGS.wandb_project,
            entity=ARGS.wandb_entity,
            name=f"{new_run_id}",
            id=new_run_id,
            config=ARGS,
            # job_type=ARGS.mode,
            group=ARGS.algo,
            tags=ARGS.wandb_tags,
        )
    return run