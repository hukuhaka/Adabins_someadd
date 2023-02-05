import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .miniViT import mViT


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features,
                                            kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(
            2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, backbone, decoded_features):
        super(DecoderBN, self).__init__()

        self.backbone = backbone

        if self.backbone == "efficientnet":
            num_features = 2048
            features = int(num_features * 0.5)

            self.conv2 = nn.Conv2d(
                in_channels=num_features, out_channels=features, kernel_size=1, stride=1, padding=1)

            self.up1 = UpSampleBN(skip_input=features //
                                  1 + 176, output_features=features // 2)
            self.up2 = UpSampleBN(skip_input=features //
                                  2 + 64, output_features=features // 4)
            self.up3 = UpSampleBN(skip_input=features //
                                  4 + 40, output_features=features // 8)
            self.up4 = UpSampleBN(skip_input=features //
                                  8 + 24, output_features=features // 16)

        elif self.backbone == "mobilevit":
            num_features = 640
            features = int(num_features * 0.5)

            self.conv2 = nn.Conv2d(
                in_channels=num_features, out_channels=features, kernel_size=1, stride=1, padding=1)

            self.up1 = UpSampleBN(skip_input=features //
                                  1 + 128, output_features=features // 2)
            self.up2 = UpSampleBN(skip_input=features //
                                  2 + 96, output_features=features // 4)
            self.up3 = UpSampleBN(skip_input=features //
                                  4 + 64, output_features=features // 8)
            self.up4 = UpSampleBN(skip_input=features //
                                  8 + 32, output_features=features // 16)
            
        elif self.backbone == "mobilevitv2":
            num_features = 512
            features = int(num_features * 0.5)

            self.conv2 = nn.Conv2d(
                in_channels=num_features, out_channels=features, kernel_size=1, stride=1, padding=1)

            self.up1 = UpSampleBN(skip_input=features //
                                  1 + 384, output_features=features // 2)
            self.up2 = UpSampleBN(skip_input=features //
                                  2 + 256, output_features=features // 4)
            self.up3 = UpSampleBN(skip_input=features //
                                  4 + 128, output_features=features // 8)
            self.up4 = UpSampleBN(skip_input=features //
                                  8 + 64, output_features=features // 16)

        self.conv3 = nn.Conv2d(in_channels=features // 16, out_channels=decoded_features,
                               kernel_size=3, stride=1, padding=1)

    def forward(self, features):

        if self.backbone == "efficientnet":
            x_block0 = features[3]
            x_block1 = features[4]
            x_block2 = features[5]
            x_block3 = features[7]
            x_block4 = features[10]

        elif self.backbone == "mobilevit" or self.backbone == "mobilevitv2":
            x_block0 = features[2]
            x_block1 = features[3]
            x_block2 = features[4]
            x_block3 = features[5]
            x_block4 = features[7]

        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        out = self.conv3(x_d4)

        return out


class Encoder(nn.Module):
    def __init__(self, backbone):
        super(Encoder, self).__init__()

        self.backbone = backbone


        if self.backbone == "efficientnet":
            self.backend = timm.create_model(
                "tf_efficientnet_b5_ap", pretrained=True)
        elif self.backbone == "mobilevit":
            self.backend = timm.create_model("mobilevit_s", pretrained=True)
        elif self.backbone == "mobilevitv2":
            self.backend = timm.create_model("mobilevitv2_100", pretrained=True)

    def forward(self, x):
        feature_maps = [x]

        if self.backbone == "efficientnet":
            for k, v in self.backend._modules.items():
                if (k == 'blocks'):
                    for ki, vi in v._modules.items():
                        feature_maps.append(vi(feature_maps[-1]))
                else:
                    feature_maps.append(v(feature_maps[-1]))
        elif self.backbone == "mobilevit" or self.backbone == "mobilevitv2":
            for key, value in self.backend._modules.items():
                if key == "stages":
                    for k, v in value._modules.items():
                        feature_maps.append(v(feature_maps[-1]))
                else:
                    feature_maps.append(value(feature_maps[-1]))

        return feature_maps


class AdaBins(nn.Module):
    def __init__(self, backbone="mobilevit", dataset="kitti"):
        super(AdaBins, self).__init__()

        self.encoder = Encoder(backbone=backbone)
        self.decoder = DecoderBN(
            backbone=backbone, decoded_features=128)
        self.mViT = mViT(in_channels=128, n_query_channels=128,
                         patch_size=16, dim_out=256,
                         embedding_dim=128, num_heads=4,
                         norm="relu")
        self.range_attention_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1))

        if dataset == "nyu":
            self.min = 1e-03
            self.max = 10.0
        elif dataset == "kitti":
            self.min = 1e-03
            self.max = 80.0
        self.width_range = self.max - self.min

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        bin_widths_normed, range_attention_maps = self.mViT(x)

        out = self.range_attention_conv(range_attention_maps)

        bins_widths = self.width_range * bin_widths_normed
        bins_widths = nn.functional.pad(
            bins_widths, (1, 0), mode="constant", value=self.min)
        bins_edges = torch.cumsum(bins_widths, dim=1)

        centers = 0.5 * (bins_edges[:, :-1] + bins_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        return bins_edges, pred
