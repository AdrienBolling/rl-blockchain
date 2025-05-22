import logging
import wandb

def setup_logging(args
) -> None:
    
    """
    Set up logging configuration.

    Args:
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        format (str): Format of the log messages.
        datefmt (str): Date format for the log messages.
        filename (str): Name of the file to log to. If None, logs to console.
        filemode (str): File mode for logging ('a' for append, 'w' for overwrite).
        stream (bool): If True, logs to console as well.

    Returns:
        None
    """
    
    level = args.logging_level
    format = args.logging_format
    datefmt = args.logging_datefmt
    filename = args.logging_filename
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
    
def setup_wandb(ARGS):
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
        wandb.init(
            project=ARGS.wandb_project,
            entity=ARGS.wandb_entity,
            id=chkpt_name,
            resume="must",
            config=ARGS,
            job_type=ARGS.mode,
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
        wandb.init(
            project=ARGS.wandb_project,
            entity=ARGS.wandb_entity,
            name=f"run_{new_run_id}",
            id=new_run_id,
            config=ARGS,
            job_type=ARGS.mode,
            group=ARGS.algo,
            tags=ARGS.wandb_tags,
        )