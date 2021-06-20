import os
import torch
import wandb
from model.config import BATCH_SIZE, RANDOM_SEED, VALID_RATIO, NUM_WORKERS
from torch.utils.data import Dataset
from torchvision import transforms

class WiderFaceDataset(Dataset):
    """Wider Face custom dataset."""

    def __init__(self, root_path):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

def log_dataset(use_artifact, 
        artifact_name, 
        artifact_path, dataset_name, 
        job_type='preprocess dataset', 
        project_name='Content-based RS'):

    run = wandb.init(project=project_name, job_type=job_type)
    run.use_artifact(use_artifact)
    artifact = wandb.Artifact(artifact_name, dataset_name)

    if os.path.isdir(artifact_path):
        artifact.add_dir(artifact_path)
    else:
        artifact.add_file(artifact_path)
    run.log_artifact(artifact)

def get_dataloader(dataset, 
        batch_size=BATCH_SIZE, random_seed=RANDOM_SEED, 
        valid_ratio=VALID_RATIO, shuffle=True, num_workers=NUM_WORKERS):

    pass