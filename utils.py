import os

import numpy as np
import torch
from torch import optim, nn

from models.AdaBins import AdaBins
from loss import SILogLoss, BinsChamferLoss

def model_setting(args, device, dataset, backbone):
    
    model = AdaBins(backbone=backbone, dataset=dataset).to(device)
    model = nn.DataParallel(model)
    
    siloss = SILogLoss()
    loss_bins = BinsChamferLoss()
    
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    return model, siloss, loss_bins, optimizer

class recording:
    def __init__(self):
        self.data = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, data, n=1):
        self.data = data
        self.sum += data * n
        self.count += n
        self.avg = self.sum / self.count

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def save_checkpoint(model, optimizer, epoch, filename, root="./checkpoints"):
    if not os.path.isdir(root):
        os.makedirs(root)

    fpath = os.path.join(root, filename)
    torch.save(
        {
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }
        , fpath)