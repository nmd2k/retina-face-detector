import os
import wandb
from model.config import *


def log_data_wandb(run, data_name=DATASET, data_type=None, root_path=DATA_PATH):
    """
    Logging new dataset as artifact into wandb
    Args:
        run (wandb.Run): wandb run
        data_name (string): dataset name (default: warm-up-8k)
        data_type (string): dataset type (default: RAW DATASET)
        root_path (string): path to parent dir (default: DATA_PATH)
    """
    if data_type == None:
        data_type = 'DATASET'

    artifact = wandb.Artifact(data_name, data_type)

    if os.path.isdir(root_path):
        artifact.add_dir(root_path)
    elif os.path.isfile(root_path):
        artifact.add_file(root_path)

    else:
        print("Can not log data dir/file into wandb, please double check root_path")

    run.log_artifact(artifact)

def use_data_wandb(run, data_name=DATASET, data_ver=DVERSION, data_type=None, root_path=DATA_PATH, download=True):
    """
    Use and download dataset from wandb database
    Args:
        run (wandb.Run): wandb run
        data_name (string): dataset name (default: warm-up-8k)
        data_ver (string): dataset version (default: latest)
        data_type (string): dataset type (default: RAW DATASET)
        root_path (string): path to parent dir (default: DATA_PATH)
        download (bool): trigger for download artifact (default: True)
    """
    if data_type == None:
        artifact = run.use_artifact(data_name+':'+data_ver)
    else:
        artifact = run.use_artifact(data_name+':'+data_ver, data_type)

    if download:
        artifact.download(root_path)

    return artifact