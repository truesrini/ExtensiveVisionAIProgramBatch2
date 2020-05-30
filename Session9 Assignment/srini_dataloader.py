# This module contains the data loading functionality for Cifar10 dataset. This module returns the train loader and test loader

import torch
from torchvision import datasets, transforms
import torchvision

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose,Normalize
)
from albumentations.pytorch import ToTensor

def cifar10_loader() :

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
               
    return trainloader, testloader

    
def cifar10_loader_albumentation(HorizontalFlip_p = 0, IAAPerspective_p = 0, ShiftScaleRotate_p = 0, CLAHE_p = 0, RandomRotate90_p = 0,
    Transpose_p = 0, Blur_p = 0, OpticalDistortion_p = 0, GridDistortion_p = 0, HueSaturationValue_p = 0,
    IAAAdditiveGaussianNoise_p = 0, GaussNoise_p = 0, MotionBlur_p = 0, MedianBlur_p = 0, RandomBrightnessContrast_p = 0, IAAPiecewiseAffine_p = 0,
    IAASharpen_p = 0, IAAEmboss_p = 0, Flip_p = 0, OneOf_p = 0 , Compose_p = 1) :
    
    class albu_transforms:
        def __init__(self,transforms):
            self.transforms = transforms
            
        def __call__(self,img):
            img = np.array(img)
            img = self.transforms(image=img)["image"]
            return img

    transform_train = Compose([
        ToTensor(),
        Normalize(mean=(0.5,0.5,0.5), std=(0.5, 0.5, 0.5), always_apply=True),
        HorizontalFlip(p=HorizontalFlip_p),
        OneOf([
            IAAAdditiveGaussianNoise(p=IAAAdditiveGaussianNoise_p),
            GaussNoise(p=GaussNoise_p),
        ], p=(IAAAdditiveGaussianNoise_p+GaussNoise_p)),
        OneOf([
            MotionBlur(p=MotionBlur_p),
            MedianBlur(blur_limit=3, p=MedianBlur_p),
            Blur(blur_limit=3, p=Blur_p),
        ], p=(MotionBlur_p+MedianBlur_p+Blur_p)),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=ShiftScaleRotate_p),
        OneOf([
            OpticalDistortion(p=OpticalDistortion_p),
            GridDistortion(p=GridDistortion_p),
            IAAPiecewiseAffine(p=IAAPiecewiseAffine_p),
        ], p=(OpticalDistortion_p + GridDistortion_p + IAAPiecewiseAffine_p)),
        OneOf([
            CLAHE(clip_limit=2,p=CLAHE_p),
            IAASharpen(p=IAASharpen_p),
            IAAEmboss(p=IAAEmboss_p),
            RandomBrightnessContrast(p=RandomBrightnessContrast_p),            
        ], p=(CLAHE_p+IAASharpen_p+IAAEmboss_p+RandomBrightnessContrast_p)),
        HueSaturationValue(p=HueSaturationValue_p),
    ], p=Compose_p)

    transform_test = Compose([
        ToTensor(),
        Normalize(mean=(0.5,0.5,0.5), std=(0.5, 0.5, 0.5), always_apply=True),
    ], p=Compose_p)


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=albu_transforms(transform_train))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=albu_transforms(transform_test))
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
               
    return trainloader, testloader