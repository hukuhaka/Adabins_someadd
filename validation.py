import os
import argparse

import torch
from torch import nn

import numpy as np
from tqdm import tqdm

from utils import compute_errors, recording
from dataloader import DepthDataLoader
from models.AdaBins import AdaBins

import warnings
warnings.filterwarnings('ignore')


def validation(model, dataloader, device):

    a1_meter = recording()
    a2_meter = recording()
    a3_meter = recording()
    rmse_meter = recording()

    with torch.no_grad():
        for batch in dataloader:
            image = torch.tensor(
                batch["image"], dtype=torch.float32, device=device)
            depth = torch.tensor(
                batch['depth'], dtype=torch.float32, device=device)

            _, pred = model(image)

            pred = nn.functional.interpolate(
                pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred = pred.squeeze().cpu().numpy()
            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(
                gt_depth < max_depth, gt_depth > min_depth)

            test_error_meters = compute_errors(
                gt_depth[valid_mask], pred[valid_mask])

            a1_meter.update(test_error_meters["a1"])
            a2_meter.update(test_error_meters["a2"])
            a3_meter.update(test_error_meters["a3"])
            rmse_meter.update(test_error_meters["rmse"])

    return a1_meter.avg, a2_meter.avg, a3_meter.avg, rmse_meter.avg


if __name__ == '__main__':

    min_depth = 0.5
    max_depth = 250.0

    device = torch.device("cuda:1")

    parser = argparse.ArgumentParser(description="AdaBins Validation File")
    parser.add_argument("--model_path", type=str,
                        default="/home/iasl/project/Model/Adabins/0204-0020/020-efficientnet.pt", help="Model Path. dir or file")

    # KITTI
    parser.add_argument("--test_file", type=str,
                        default="./database/kitti_eigen_test.txt", help="Test List File")
    parser.add_argument("--test_image_path", type=str,
                        default="/home/dataset/EH/DataSet/KITTI/KITTI_RGB_Image", help="Image Path to train")
    parser.add_argument("--test_depth_path", type=str,
                        default="/home/dataset/EH/DataSet/KITTI/KITTI_PointCloud", help="Depth Image Path to train")
    parser.add_argument("--dataset", type=str, default="kitti")

    # DIODE
    # parser.add_argument("--test_file", type=str,
    #                     default="./database/DIODE_outdoor_test.txt", help="Test List File")
    # parser.add_argument("--test_image_path", type=str,
    #                     default="/home/dataset/EH/DataSet/DIODE", help="Image Path to train")
    # parser.add_argument("--test_depth_path", type=str,
    #                     default="/home/dataset/EH/DataSet/DIODE", help="Depth Image Path to train")
    # parser.add_argument("--dataset", type=str, default="diode")

    args = parser.parse_args()

    model = model = AdaBins(backbone="efficientnet", width_range=80).to(device)
    TestLoader = DepthDataLoader(
        args, mode="online_eval", base_data=args.dataset).data

    a1_meter = []
    a2_meter = []
    a3_meter = []
    rmse_meter = []

    if os.path.isfile(args.model_path):

        ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        a1, a2, a3, rmse = validation(model, TestLoader, device)

        a1_meter.append(a1)
        a2_meter.append(a2)
        a3_meter.append(a3)
        rmse_meter.append(rmse)

    else:
        path_list = sorted(os.listdir(args.model_path))
        for name in tqdm(path_list):

            ckpt = torch.load(os.path.join(
                args.model_path, name), map_location=device)
            model.load_state_dict(ckpt["model"])
            model.eval()

            a1, a2, a3, rmse = validation(model, TestLoader, device)

            a1_meter.append(a1)
            a2_meter.append(a2)
            a3_meter.append(a3)
            rmse_meter.append(rmse)

    best_model = np.argmax(a1_meter)

    print(f"BestModel: {path_list[best_model]}\na1: {a1_meter[best_model]:.4f} a2: {a2_meter[best_model]:.4f} a3: {a3_meter[best_model]:.4f} RMSE: {rmse_meter[best_model]:.4f}")

    meter_dict = {"a1": a1_meter, "a2": a2_meter,
                  "a3": a3_meter, "RMSE": rmse_meter}
    np.save("TEST.npy", meter_dict)
