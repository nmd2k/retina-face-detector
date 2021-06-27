import argparse
from model.multibox_loss import MultiBoxLoss

from model.config import *

import os 
import wandb
import time
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, dataloader
from model.model import RetinaFace
from utils.dataset import WiderFaceDataset


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train image segmentation')
    parser.add_argument('--run', type=str, default='demo', help="run name")
    parser.add_argument('--epoch', type=int, default=EPOCHS, help="number of epoch")
    parser.add_argument('--weight_decay', type=int, default=WEIGHT_DECAY, help="weight decay of optimizer")
    parser.add_argument('--momentum', type=int, default=MOMENTUM, help="momemtum of optimizer")
    parser.add_argument('--startfm', type=int, default=START_FRAME, help="architecture start frame")
    parser.add_argument('--batchsize', type=int, default=BATCH_SIZE, help="total batch size for all GPUs (default:")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="init learning rate (default: 0.0001)")
    parser.add_argument('--tuning', action='store_true', help="no plot image for tuning")

    args = parser.parse_args()
    return args

def train(model, anchors, trainloader, optimizer, loss_function, best_ap, device='cpu'):
    model.train()
    loss_cls, loss_box, loss_pts = 0, 0, 0
    epoch_ap = 0
    for i, (input, targets) in enumerate(trainloader):
        # load data into cuda
        input, targets = input.to(device), targets.to(device)

        # forward
        predict = model(input)
        loss_l, loss_c, loss_landm = loss_function(predict, anchors, targets)
        loss = loss_l + loss_c + loss_landm

        # metric
        # TODO: log ap
        loss_cls += loss_c
        loss_box += loss_l 
        loss_pts += loss_landm

        # zero the gradient + backprpagation + step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
    
    # cls = classification; box = box regressionl; pts = landmark regression
    loss_cls = loss_cls/len(trainloader)
    loss_box = loss_box/len(trainloader)
    loss_pts = loss_pts/len(trainloader)

    wandb.log({'train': {'loss_cls': loss_cls, 
            'loss_box': loss_box, 
            'loss_landmark': loss_pts}})

    if epoch_ap>best_ap and not args.tuning:
        # export to onnx + pt
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        try:
            torch.onnx.export(model, input, os.path.join(SAVE_PATH+RUN_NAME+'.onnx'))
            torch.save(model.state_dict(), os.path.join(SAVE_PATH+RUN_NAME+'.pth'))
        except:
            print('Can export weights')

    return loss_cls, loss_box, loss_pts, epoch_ap

if __name__ == '__main__':
    args = parse_args()

    # init wandb
    config = dict(
        epoch           = args.epoch,
        weight_decay    = args.weight_decay,
        momentum        = args.momentum,
        lr              = args.lr,
        batchsize       = args.batchsize,
        startfm         = args.startfm,
    )
    
    # log experiments to
    RUN_NAME = args.run
    run = wandb.init(project="RetinaFace", config=config)
    artifact = wandb.Artifact(DATASET_NAME, type='RAW_DATASET')

    try:
        artifact.add_dir(DATA_PATH)
        run.log_artifact(artifact)
    except:
        artifact     = run.use_artifact(DATASET_NAME+DATASET_VER)
        artifact_dir = artifact.download(DATA_PATH)

    # train on device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Current device", device)

    # get dataloader
    train_set = WiderFaceDataset(TRAIN_PATH)
    valid_set = WiderFaceDataset(VALID_PATH)
    
    torch.manual_seed(RANDOM_SEED)
    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    validloader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # get model and define loss func, optimizer
    n_classes = N_CLASSES
    epochs = args.epoch

    model = RetinaFace().to(device)

    with torch.no_grad():
        # in Retina paper, they use anchor box which same as Prior box in SSD
        anchors = model.priors.to(device)

    # optimizer + citeration
    optimizer   = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    criterion   = MultiBoxLoss(N_CLASSES, 
                            overlap_thresh=OVERLAP_THRES, 
                            prior_for_matching=True, 
                            bkg_label=BKG_LABEL, neg_pos=True, 
                            neg_mining=NEG_MINING, neg_overlap=NEG_OVERLAP, 
                            encode_target=False, device=device)

    # wandb watch
    run.watch(models=model, criterion=criterion, log='all', log_freq=10)

    # training
    best_ap = -1

    for epoch in range(epochs):
        print(f'\tEpoch\tbox\t\tlandmarks\tcls\t\ttotal')
        t0 = time.time()
        loss_box, loss_pts, loss_cls, train_ap = train(model, anchors, trainloader, optimizer, criterion, best_ap, device)
        t1 = time.time()

        total_loss = loss_box + loss_pts + loss_cls
        print(f'\t{epoch}/{epochs}\t{loss_box}\t\t{loss_pts}\t\t{loss_cls:.5f}\t\t{():.5f}\t\t{(t1-t0):.2f}s')
        
        # summary
        # print(f'\tImages\tLabels\t\tP\t\tR\t\tmAP@.5\t\tmAP.5.95')
        # images, labels, P, R, map_5, map_95
        # print(f'\t{images}\t{labels}\t\t{P}\t\t{R}\t\t{map_5}\t\t{map_95}')
    
        # Wandb summary
        if train_ap > best_ap:
            best_ap = train_ap
            wandb.run.summary["best_accuracy"] = best_ap

    if not args.tuning:
        trained_weight = wandb.Artifact(RUN_NAME, type='weights')
        trained_weight.add_file(SAVE_PATH+RUN_NAME+'.onnx')
        trained_weight.add_file(SAVE_PATH+RUN_NAME+'.pth')
        wandb.log_artifact(trained_weight)