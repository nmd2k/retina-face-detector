import argparse
from model.multibox_loss import MultiBoxLoss

from model.config import *

import os 
import wandb
import time
import torch
import numpy as np
from torch import optim
from model.model import RetinaFace
from data.dataset import WiderFaceDataset


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train image segmentation')
    parser.add_argument('--run', type=str, default='demo', help="run name")
    parser.add_argument('--epoch', type=int, default=EPOCHS, help="number of epoch")
    parser.add_argument('--weight_decay', type=int, default=WEIGHT_DECAY, help="weight decay of optimizer")
    parser.add_argument('--momentum', type=int, default=MOMENTUM, help="momemtum of optimizer")
    parser.add_argument('--startfm', type=int, default=START_FRAME, help="architecture start frame")
    parser.add_argument('--batchsize', type=int, default=BATCH_SIZE, help="total batch size for all GPUs (default:")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="learning rate (default: 0.0001)")
    parser.add_argument('--tuning', action='store_true', help="no plot image for tuning")

    args = parser.parse_args()
    return args


def train(model, device, trainloader, optimizer, loss_function):
    model.train()
    running_loss = 0
    for i, (input, targets) in enumerate(trainloader):
        # load data into cuda
        input, targets = input.to(device), targets.to(device)

        # forward
        predict = model(input)
        loss = loss_function(predict, targets)

        # metric
        # TODO: log ap, loc_loss, class_loss, landmark_loss
        running_loss += (loss.item())
        
        # zero the gradient + backprpagation + step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
            
    # mean_iou = np.mean(iou)
    total_loss = running_loss/len(trainloader)
    
    # wandb.log({'Train loss': total_loss, 'Train IoU': })

    return total_loss
    
def test(model, device, testloader, loss_function, best_iou):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for i, (input, targets) in enumerate(testloader):
            input, targets = input.to(device), targets.to(device)

            predict = model(input)
            loss = loss_function(predict, targets)

            running_loss += loss.item()

    test_loss = running_loss/len(testloader)
    
    # log wandb
    # wandb.log({'Valid loss': test_loss, 'Valid IoU': mean_iou, 'Prediction': targets_list})
    epoch_ap = 0

    # TODO: cal epoch_ap
    if epoch_ap>best_iou and not args.tuning:
        # export to onnx + pt
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        try:
            torch.onnx.export(model, input, os.path.join(SAVE_PATH+RUN_NAME+'.onnx'))
            torch.save(model.state_dict(), os.path.join(SAVE_PATH+RUN_NAME+'.pth'))
        except:
            print('Can export weights')

    return test_loss

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


    # TODO: Workin' on it
    dataset = WiderFaceDataset(DATA_PATH)
    trainloader, validloader = get_dataloader(dataset=dataset, batch_size=args.batchsize)

    # get model and define loss func, optimizer
    n_classes = N_CLASSES
    epochs = args.epoch

    model = RetinaFace().to(device)

    # loss_func   = Weighted_Cross_Entropy_Loss()
    optimizer   = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    # citeration
    criterion = MultiBoxLoss(N_CLASSES, 
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
        t0 = time.time()
        train_loss, train_ap = train(model, device, trainloader, optimizer, criterion)
        t1 = time.time()
        print(f'Epoch: {epoch} | Train loss: {train_loss:.3f} | Train IoU: {train_iou:.3f} | Time: {(t1-t0):.1f}s')
        test_loss, test_ap = test(model, device, validloader, criterion, best_ap)
        print(f'Epoch: {epoch} | Valid loss: {test_loss:.3f} | Valid IoU: {test_ap:.3f} | Time: {(t1-t0):.1f}s')
        
        # Wandb summary
        if best_ap < test_ap:
            best_ap = test_ap
            wandb.run.summary["best_accuracy"] = best_ap

    if not args.tuning:
        trained_weight = wandb.Artifact(RUN_NAME, type='weights')
        trained_weight.add_file(SAVE_PATH+RUN_NAME+'.onnx')
        trained_weight.add_file(SAVE_PATH+RUN_NAME+'.pth')
        wandb.log_artifact(trained_weight)
    # evaluate