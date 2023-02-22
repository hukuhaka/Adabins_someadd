import argparse
import time
import os

import numpy as np
from tqdm import tqdm
import aim

import torch
from torch import optim, nn

from utils import *
from dataloader import DepthDataLoader

import warnings
warnings.filterwarnings('ignore')


class AdaBinsTrainer:
    def __init__(self, args):

        self.args = args

        self.device = torch.device(
            "cuda" if self.args.device == "cuda" else "cpu")

        lt = time.localtime()
        time_config = str(lt.tm_mon).zfill(2)+str(lt.tm_mday).zfill(2) + \
            "-"+str(lt.tm_hour).zfill(2)+str(lt.tm_min).zfill(2)
        project_name = "AdaBins"+"-"+time_config

        self.run = aim.Run(
            repo="/home/dataset/EH",
            experiment="AdaBins"
        )
        self.run["hparams"] = {
            "Model": "AdaBins",
            "Backbone": self.args.backbone,
            "Batch Size": self.args.batch,
            "optimizer": self.args.optimizer,
            "DataSet": self.args.dataset_type,
            "Loss": self.args.loss
        }

        self.dir_name = self.args.save_path+"/"+time_config
        os.makedirs(self.dir_name, exist_ok=True)

        self.model, self.loss_ueff, self.loss_bins, self.optimizer = model_setting(
            self.args, self.device, self.args.dataset_type, self.args.backbone)

        self.TrainLoader = DepthDataLoader(args, mode="train").data
        self.TestLoader = DepthDataLoader(args, mode="online_eval").data

        if self.args.dataset_type == "nyu":
            self.min_depth = 1e-03
            self.max_depth = 10.0
        elif self.args.dataset_type == "kitti":
            self.min_depth = 1e-03
            self.max_depth = 80.0

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, epochs=self.args.epochs, steps_per_epoch=1, anneal_strategy ="cos", pct_start=10/30,
            cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, div_factor=1e+04*self.args.lr/1, final_div_factor=1/0.5)

    def train(self):
        print("\nData Loading is complete.\tStart Training...")

        for epoch in range(self.args.epochs):

            self.model.train()
            epoch_lr = self.optimizer.param_groups[0]["lr"]
            self.run.track(epoch_lr, name="lr")

            loss_meter = recording()

            print("\n"+"-"*50)
            print(f"{epoch+1}/{self.args.epochs} Epoch \n"
                  f"Train mode. Learning Rate: {epoch_lr:e}")
            with tqdm(self.TrainLoader, unit="batch") as trainloader:
                for idx, batch in enumerate(trainloader):

                    self.optimizer.zero_grad()

                    image = batch["image"].to(self.device)
                    depth = batch['depth'].to(self.device)
                    if 'has_valid_depth' in batch:
                        if not batch['has_valid_depth']:
                            continue

                    bin_edges, pred = self.model(image)

                    mask = depth > self.min_depth
                    l_dense = self.loss_ueff(pred, depth, mask=mask.to(
                        torch.bool), interpolate=True)
                    l_chamfer = self.loss_bins(bin_edges, depth)
                    total_loss = l_dense + 0.1 * l_chamfer
                    total_loss.backward()
                    # nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                    self.optimizer.step()

                    loss_meter.update(total_loss.item(), image.size(0))
                    num_iters = epoch * len(trainloader) + idx
                    self.run.track(loss_meter.data, name="loss", step=num_iters, epoch=epoch, context={
                                   "subset": "train", "model": "AdaBins"})
                    self.run.track(pred.max(), name="pred max")

                    trainloader.set_postfix(
                        loss=loss_meter.data, loss_avg=loss_meter.avg)

            self.scheduler.step()
            self.run.track(loss_meter.avg, name="avg_loss", epoch=epoch, context={
                           "subset": "train", "model": "AdaBins"})

            print("\nTest mode")
            self.model.eval()
            with torch.no_grad():

                test_meter = recording()
                a1_meter = recording()
                a2_meter = recording()
                a3_meter = recording()
                rmse_meter = recording()

                for idx, batch in enumerate(tqdm(self.TestLoader)):

                    if 'has_valid_depth' in batch:
                        if not batch['has_valid_depth']:
                            continue

                    image = batch["image"].to(self.device)
                    depth = batch['depth'].to(self.device)
                    depth = depth.squeeze().unsqueeze(0).unsqueeze(0)

                    _, pred = self.model(image)

                    mask = depth > self.min_depth
                    l_dense = self.loss_ueff(pred, depth, mask=mask.to(
                        torch.bool), interpolate=True)

                    test_meter.update(l_dense.item())

                    pred = nn.functional.interpolate(
                        pred, depth.shape[-2:], mode='bilinear', align_corners=True)
                    pred = pred.squeeze().cpu().numpy()
                    pred[pred < self.min_depth] = self.min_depth
                    pred[pred > self.max_depth] = self.max_depth
                    pred[np.isinf(pred)] = self.max_depth
                    pred[np.isnan(pred)] = self.min_depth

                    gt_depth = depth.squeeze().cpu().numpy()
                    valid_mask = np.logical_and(
                        gt_depth < self.max_depth, gt_depth > self.min_depth)

                    gt_height, gt_width = gt_depth.shape
                    eval_mask = np.zeros(valid_mask.shape)

                    if args.dataset_type == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                                  int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1

                    valid_mask = np.logical_and(valid_mask, eval_mask)
                    test_error_meters = compute_errors(
                        gt_depth[valid_mask], pred[valid_mask])

                    a1_meter.update(test_error_meters["a1"])
                    a2_meter.update(test_error_meters["a2"])
                    a3_meter.update(test_error_meters["a3"])
                    rmse_meter.update(test_error_meters["rmse"])

            self.run.track(test_meter.avg, name="test_loss",
                           epoch=epoch, context={"subset": "test", "model": "AdaBins"})
            self.run.track(a1_meter.avg, name="delta < 1.25",
                           epoch=epoch, context={"subset": "test", "model": "AdaBins"})
            self.run.track(a2_meter.avg, name="delta < 1.25 ** 2",
                           epoch=epoch, context={"subset": "test", "model": "AdaBins"})
            self.run.track(a3_meter.avg, name="delta < 1.25 ** 3",
                           epoch=epoch, context={"subset": "test", "model": "AdaBins"})
            self.run.track(rmse_meter.avg, name="RMSE",
                           epoch=epoch, context={"subset": "test", "model": "AdaBins"})

            print(f"Result\n"
                  f"Training Avg Loss: {loss_meter.avg:.4f}\tTest Avg Loss: {test_meter.avg}\n"
                  f"a1: {a1_meter.avg:.4f} a2: {a2_meter.avg:.4f} a3: {a3_meter.avg:.4f} RMSE: {rmse_meter.avg:.4f}")

            model_name = str(epoch).zfill(3)+"-"+self.args.backbone+".pt"
            save_checkpoint(model=self.model, optimizer=self.optimizer,
                            epoch=epoch, filename=model_name, root=self.dir_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="AdaBins Training File")
    # Basic Model Settings
    parser.add_argument("--epochs", default=30, type=int,
                        help="Number of epochs to train")
    parser.add_argument("--backbone", default="mobilevit", type=str, choices=[
                        "efficientnet", "mobilevit", "mobilevitv2"], help="Choose Encoder-Decoder Backbone")
    parser.add_argument("--lr", default=8e-05, type=float,
                        help="Base learning rate. We use OneCycleLR")
    parser.add_argument("--optimizer", default="adamw", type=str,
                        choices=["adam", "adamw"], help="Optimizer Setting")
    parser.add_argument("--loss", default="silogloss", type=str,
                        choices=["silogloss", "ssiloss", "ssilogloss"], help="Optimizer Setting")
    parser.add_argument("--batch", default=8, type=int,
                        help="Number of Batch Size")
    parser.add_argument("--device", default="cuda", type=str,
                        choices=["cpu", "cuda"], help="Choose Device to Train")
    
    # Dataset Settings
    parser.add_argument("--dataset_type", default="kitti", type=str,
                        choices=["nyu", "kitti"], help="Choose Dataset Type")
    parser.add_argument(
        "--filenames_file", default="./train_test_inputs/kitti_eigen_train_files_with_gt.txt", help="Eigen Split Train File. For groung truth. No Dense Map")
    parser.add_argument(
        "--filenames_file2", default="./train_test_inputs/kitti_eigen_train_files_with_gt_numpy.txt", help="Eigen Split Train File. For Dense Map")
    parser.add_argument("--filenames_file_eval",
                        default="./train_test_inputs/kitti_eigen_test_files_with_gt.txt", help="Eigen Split Test File")
    parser.add_argument("--image_path", default="/home/dataset/EH/DataSet/KITTI/KITTI_RGB_Image",
                        type=str, help="Image Path to train")
    parser.add_argument("--dense_depth_path", default="/home/dataset/EH/DataSet/KITTI/KITTI_DenseMap",
                        type=str, help="Depth Image Path to train")
    parser.add_argument("--truth_depth_path", default="/home/dataset/EH/DataSet/KITTI/KITTI_PointCloud",
                        type=str, help="Depth Image Path to train")
    parser.add_argument("--save_path", default="/home/dataset/EH/project/Model/Adabins",
                        type=str, help="Model Save Path")
    parser.add_argument("--dataset_mode", default=3, type=int, help="1: Only GT. 2: Only Dense Map. 3: GT+Dense Map.")

    # DataSet PreProcessing Settings
    parser.add_argument("--height", default=352, type=int)
    parser.add_argument("--width", default=704, type=int)
    parser.add_argument("--rotation", default=True, type=bool)
    parser.add_argument("--random_crop", default=False, type=bool)
    parser.add_argument("--rgb_swap", default=True, type=bool)
    parser.add_argument("--worker", default=8, type=int)
    parser.add_argument("--drop_last", default=True, type=bool)

    # Validation Settings
    parser.add_argument("--do_val", default=True, type=bool,
                        help="Doing Validation when finish one epoch")

    args = parser.parse_args()

    trainer = AdaBinsTrainer(args)
    trainer.train()
