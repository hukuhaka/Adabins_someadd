import os
import random
from PIL import Image

import numpy as np
import albumentations as A

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class DepthDataLoader(object):
    def __init__(self, args, mode, base_data):
        # dataset_profile = image_path, depth_path, image_list_file, image_type
        if mode == 'train':
            self.training_samples = ConcatDataset([DataLoadPreprocess(dataset_profile, mode, base_data) for dataset_profile in zip(
                args.image_path, args.depth_path, args.train_file, args.dataset_type)])

            self.data = DataLoader(self.training_samples, batch_size=args.batch,
                                   shuffle=True, num_workers=args.batch*2,
                                   pin_memory=True, drop_last=args.drop_last)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, base_data)

            self.data = DataLoader(self.testing_samples, batch_size=1,
                                   shuffle=False, num_workers=2, pin_memory=False)

        elif mode == 'validation':
            self.testing_samples = DataLoadPreprocess(args, "online_eval", base_data)

            self.data = DataLoader(self.testing_samples, batch_size=8,
                                   shuffle=False, num_workers=8, pin_memory=True)

        else:
            print(
                'mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self, dataset_profile, mode, base_data):
        if mode == 'train':
            self.image_path, self.depth_path, list_file, self.dataset_type = dataset_profile
        elif mode == "online_eval":
            self.image_path, self.depth_path, list_file, self.dataset_type = dataset_profile.test_image_path, dataset_profile.test_depth_path, dataset_profile.test_file, base_data
        with open(list_file, 'r') as f:
            self.filenames = f.readlines()
            
        self.transform = transforms.Compose([ToTensor(mode=self.dataset_type)])
        self.mode = mode
        self.base_data = base_data
        
        if self.base_data == "diode":
            self.height = 480
            self.width = 640
        elif self.base_data == "kitti":
            self.height = 352
            self.width = 704
        
        self.A_train_transform = A.Compose([
            A.ChannelShuffle(p=1.0),
            A.Resize(height=self.height*2, width=self.width*2),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.ShiftScaleRotate(rotate_limit=20),
            A.RandomResizedCrop(height=self.height, width=self.width, scale=(0.95, 1), ratio=(0.8, 1.25)),
        ])
        
        self.A_test_transform = A.Compose([
            A.Resize(height=self.height, width=self.width)
        ])

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        image_path = os.path.join(self.image_path, sample_path.split()[0])
        image = np.array(Image.open(image_path), dtype=np.float32) / 255.

        depth_path = os.path.join(self.depth_path, sample_path.split()[1])
        if depth_path.split(".")[-1] == "npy":
            depth_gt = np.load(depth_path).squeeze()
        else:
            depth_gt = np.array(Image.open(depth_path)).squeeze()
        depth_gt = np.expand_dims(depth_gt, axis=2).astype(np.float32)

        if self.dataset_type == "nyu":
            depth_gt = depth_gt / 1000.0
        elif (self.dataset_type == "kitti") or (self.dataset_type == "kitti_dense"):
            depth_gt = depth_gt / 256.0

        if self.mode == "train":
            sample = self.A_train_transform(image=image, mask=depth_gt)
        else:
            sample = self.A_test_transform(image=image, mask=depth_gt)
            
        sample = {'image': sample["image"], 'depth': sample["mask"]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        # assert img.shape[0] >= height
        # assert img.shape[1] >= width
        # assert img.shape[0] == depth.shape[0]
        # assert img.shape[1] == depth.shape[1]
        if img.shape[1] - width > 0:
            x = random.randint(0, img.shape[1] - width)
        else: x = 0
        
        if img.shape[0] - height > 0:
            y = random.randint(0, img.shape[0] - height)
        else: y = 0
        
        # x = random.randint(0, img.shape[1] - width)
        # y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.dataset_type == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image = sample['image']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image}

        depth = sample['depth']
        depth = self.to_tensor(depth)
        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
