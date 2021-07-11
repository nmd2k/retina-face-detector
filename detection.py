import os
import cv2
import torch
import time
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

from model.anchor import Anchors
from model.config import *
from model.model import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms import py_cpu_nms

def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train image segmentation')
    parser.add_argument('--model', type=str, default='resnet50', help='select model')
    parser.add_argument('--path', type=str, default='./', help='path to file')
    parser.add_argument('--weight', type=str, default=None, help='path to pretrained weight')
    parser.add_argument('--device', type=int, default=0, help="no plot image for tuning")
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=7000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.65, type=float, help='visualization_threshold')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parse_args()

    # train on device
    device = torch.device("cpu")

    if args.device is not None:
        device = torch.device(args.device)
    print(f"\tCurrent training device {torch.cuda.get_device_name(device)}")
    
    # net & model
    model = RetinaFace(model_name=args.model, is_train=False).to(device)
    if args.weight is not None and os.path.isfile(args.weight):
        checkpoint = torch.load(args.weight, map_location=device)
        model.load_state_dict(checkpoint)
        print(f'\tWeight located in {args.weight} have been loaded')

    model.eval()

    cudnn.benchmark = True

    # load image
    img_raw = cv2.imread(args.path, cv2.IMREAD_COLOR)

    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    # scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    scale = torch.Tensor([im_height, im_width, im_height, im_width])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    
    resize = 1

    # prediction
    tic = time.time()
    loc, conf, landms = model(img)  # forward pass
    print('\tModel forward time: {:.4f}'.format(time.time() - tic))

    input_size = np.array([im_height, im_width])
    # input_size = np.array([im_width, im_height])
    anchors = Anchors(image_size=input_size, pyramid_levels=model.feature_map).forward().to(device)

    boxes = decode(loc.data.squeeze(), anchors, [0.1, 0.2])
    # from IPython import embed
    # embed()
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), anchors, [0.1, 0.2])
    # scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
    #                         img.shape[3], img.shape[2], img.shape[3], img.shape[2],
    #                         img.shape[3], img.shape[2]])
    scale1 = torch.Tensor([im_height, im_width, im_height, im_width,
                            im_height, im_width, im_height, im_width,
                            im_height, im_width])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    # show image
    if args.save_image:
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        # save image

        name = "test.jpg"
        cv2.imwrite(name, img_raw)

    