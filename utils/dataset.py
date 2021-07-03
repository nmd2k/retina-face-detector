import os
import torch
import wandb
import numpy as np
from model.config import BATCH_SIZE, RANDOM_SEED, TRAIN_PATH, VALID_PATH, VALID_RATIO, NUM_WORKERS
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class WiderFaceDataset(Dataset):
    """
    Wider Face custom dataset.
    Args:
        root_path (string): Path to dataset directory
        is_train (bool): Train dataset or test dataset
        transform (function): whether to apply the data augmentation scheme
                mentioned in the paper. Only applied on the train split.
    """

    def __init__(self, root_path, is_train=True, transform=None):
        self.ids       = []
        if transform != None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

        if is_train: 
            self.path = os.path.join(root_path, TRAIN_PATH)
        else: 
            self.path = os.path.join(root_path, VALID_PATH)
        
        for dirname in os.listdir(self.path):
            for subdir in os.listdir(os.path.join(self.path, dirname)):
                for file in os.listdir(os.path.join(self.path, dirname, subdir)):
                    self.ids.append(os.path.join(subdir, file)[:-4])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path, 'images', self.ids[index]+'.jpg'))

        f = open(os.path.join(self.path, 'labels', self.ids[index]+'.txt'), 'r')
        lines = f.readlines()
        annotations = np.zeros((len(lines), 15))

        if len(lines) == 0:
            return annotations
        
        for idx, line in enumerate(lines):
            line = line.strip().split()
            line = [float(x) for x in line]

            # bbox
            annotations[idx, 0] = line[0]               # x1
            annotations[idx, 1] = line[1]               # y1
            annotations[idx, 2] = line[0] + line[2]     # x2
            annotations[idx, 3] = line[1] + line[3]     # y2

            # landmarks
            annotations[idx, 4] = line[4]               # l0_x
            annotations[idx, 5] = line[5]               # l0_y
            annotations[idx, 6] = line[7]               # l1_x
            annotations[idx, 7] = line[8]               # l1_y
            annotations[idx, 8] = line[10]              # l2_x
            annotations[idx, 9] = line[11]              # l2_y
            annotations[idx, 10] = line[13]             # l3_x
            annotations[idx, 11] = line[14]             # l3_y
            annotations[idx, 12] = line[16]             # l4_x
            annotations[idx, 13] = line[17]             # l4_y

            if (annotations[idx, 4]<0):
                annotations[idx, 14] = -1
            else:
                annotations[idx, 14] = 1

        img = self.transform(img)
        annotations = self.transform(annotations)

        return img, annotations

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