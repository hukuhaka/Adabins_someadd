import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence


class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


class SSILoss(nn.Module):
    def __init__(self):
        super(SSILoss, self).__init__()

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)

        device = torch.device("cpu")
        input = torch.tensor(input, device=device)
        target = torch.tensor(target, device=device)
        mask = torch.tensor(mask, device=device)

        mask_sum = torch.sum(mask, dim=[2, 3]).view(-1, 1, 1, 1)

        input_translation = torch.tensor(
            [torch.median(input[i][mask[i]], dim=0).values for i in range(input.shape[0])])
        target_translation = torch.tensor(
            [torch.median(target[i][mask[i]], dim=0).values for i in range(target.shape[0])])

        input_translation, target_translation = input - \
            input_translation.view(-1, 1, 1, 1), target - \
            target_translation.view(-1, 1, 1, 1)
        input_scale = torch.tensor([torch.sum(torch.abs(
            input_translation[i][mask[i]])) for i in range(input.shape[0])]).view(-1, 1, 1, 1)
        target_scale = torch.tensor([torch.sum(torch.abs(
            target_translation[i][mask[i]])) for i in range(target.shape[0])]).view(-1, 1, 1, 1)

        scaled_input = input_translation * mask_sum / input_scale
        scaled_target = target_translation * mask_sum / target_scale

        target_min = torch.tensor([target[i][mask[i]].min()
                                  for i in range(target.shape[0])]).view(-1, 1, 1, 1)
        target_max = torch.tensor([target[i][mask[i]].max()
                                  for i in range(target.shape[0])]).view(-1, 1, 1, 1)

        target_normalize = 100 * \
            (scaled_target-target_min) / (target_max-target_min)

        loss = torch.mean(torch.abs(scaled_input[mask]-target_normalize[mask]))

        return loss


class SSILogLoss(nn.Module):
    def __init__(self):
        super(SSILogLoss, self).__init__()

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)

        device = torch.device("cpu")
        input = torch.tensor(input, device=device)
        target = torch.tensor(target, device=device)
        mask = torch.tensor(mask, device=device)

        mask_sum = torch.sum(mask, dim=[2, 3]).view(-1, 1, 1, 1)

        input_translation = torch.tensor(
            [torch.median(input[i][mask[i]], dim=0).values for i in range(input.shape[0])])
        target_translation = torch.tensor(
            [torch.median(target[i][mask[i]], dim=0).values for i in range(target.shape[0])])

        input_translation, target_translation = input - \
            input_translation.view(-1, 1, 1, 1), target - \
            target_translation.view(-1, 1, 1, 1)
        input_scale, target_scale = torch.sum(torch.abs(input_translation), dim=[
                                              2, 3], keepdim=True), torch.sum(torch.abs(target_translation), dim=[2, 3], keepdim=True)

        scaled_input = input_translation * mask_sum / input_scale
        scaled_input[mask] = torch.log(
            scaled_input[mask] + 1 - scaled_input[mask].min())
        # scaled_input[mask] = torch.tensor([(scaled_input[i][mask[i]] - scaled_input[i][mask[i]].min())/(scaled_input[i][mask[i]].max()-scaled_input[i][mask[i]].min()) for i in range(scaled_input.shape[0])])

        scaled_target = target_translation * mask_sum / target_scale
        scaled_target[mask] = torch.log(
            scaled_target[mask] + 1 - scaled_target[mask].min())
        # scaled_target[mask] = torch.tensor([(scaled_target[i][mask[i]] - scaled_target[i][mask[i]].min())/(scaled_target[i][mask[i]].max()-scaled_target[i][mask[i]].min()) for i in range(scaled_target.shape[0])])

        loss = torch.mean(torch.abs(scaled_input[mask]-scaled_target[mask]))

        return loss


class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor(
            [len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(
            target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(
            x=input_points, y=target_points, y_lengths=target_lengths)
        return loss
