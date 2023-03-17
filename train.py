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
            self.args, self.device, self.args.width_range, self.args.backbone)

        self.TrainLoader = DepthDataLoader(args, mode="train", base_data=args.dataset_type[0]).data
        self.TestLoader = DepthDataLoader(args, mode="online_eval", base_data=args.dataset_type[0]).data

        self.min_depth = 0.5
        self.max_depth = self.args.width_range

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, epochs=self.args.epochs, steps_per_epoch=1, anneal_strategy ="cos", pct_start=0.5,
            cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, div_factor=self.args.lr*1e+04/1, final_div_factor=1/0.5)

    def train(self):
        print(f"Backend: {self.args.backbone}\tOptimizer: {self.args.optimizer}\tMax lr: {self.args.lr}\n"
              f"Epochs: {self.args.epochs}\tDataset: {self.args.dataset_type}")
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

                    image = torch.tensor(batch["image"], dtype=torch.float32, device=self.device)
                    depth = torch.tensor(batch['depth'], dtype=torch.float32, device=self.device)

                    bin_edges, pred = self.model(image)
                    
                    mask = depth > self.min_depth
                    l_dense = self.loss_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)
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

                    image = torch.tensor(batch["image"], dtype=torch.float32, device=self.device)
                    depth = torch.tensor(batch['depth'], dtype=torch.float32, device=self.device)
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
    parser.add_argument("--epochs", default=100, type=int,
                        help="Number of epochs to train")
    parser.add_argument("--backbone", default="mobilevitv2_150", type=str, choices=[
                        "efficientnet", "mobilevit", "mobilevitv2_100", "mobilevitv2_150", "mobilevitv2_200"], help="Choose Encoder-Decoder Backbone")
    parser.add_argument("--lr", default=1e-04, type=float,
                        help="Base learning rate. We use OneCycleLR")
    parser.add_argument("--optimizer", default="adamw", type=str,
                        choices=["adam", "adamw", "lion"], help="Optimizer Setting")
    parser.add_argument("--loss", default="silogloss", type=str,
                        choices=["silogloss", "ssiloss", "ssilogloss"], help="Optimizer Setting")
    parser.add_argument("--batch", default=4, type=int,
                        help="Number of Batch Size")
    parser.add_argument("--device", default="cuda", type=str,
                        choices=["cpu", "cuda"], help="Choose Device to Train")
    parser.add_argument("--width_range", default=250, type=float,
                        choices=[10, 100, 250], help="Choose Device to Train")
    
    # Dataset Settings
    parser.add_argument("--dataset_type", default=["diode", "kitti"], type=list, action="store",
                        choices=["nyu", "kitti", "kitti_dense", "diode"], help="Choose Dataset Type")
    parser.add_argument("--train_file", type=list, action="store",
                        default=["./database/DIODE_outdoor_train.txt", "./database/kitti_eigen_train.txt"], help="Train List File")
    parser.add_argument("--test_file", type=str,
                        default="./database/DIODE_outdoor_test.txt", help="Test List File")
    parser.add_argument("--image_path", type=list, action="store",
                        default=["/home/dataset/EH/DataSet/DIODE", "/home/dataset/EH/DataSet/KITTI/KITTI_RGB_Image"], help="Image Path to train")
    parser.add_argument("--depth_path", type=list, action="store",
                        default=["/home/dataset/EH/DataSet/DIODE", "/home/dataset/EH/DataSet/KITTI/KITTI_PointCloud"], help="Depth Image Path to train")
    parser.add_argument("--test_image_path", type=str,
                        default="/home/dataset/EH/DataSet/DIODE", help="Image Path to train")
    parser.add_argument("--test_depth_path", type=str,
                        default="/home/dataset/EH/DataSet/DIODE", help="Depth Image Path to train")
    parser.add_argument("--save_path", default="../Model/Adabins",
                        type=str, help="Model Save Path")

    # DataSet PreProcessing Settings
    # parser.add_argument("--height", default=352, type=int)
    # parser.add_argument("--width", default=704, type=int)
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
