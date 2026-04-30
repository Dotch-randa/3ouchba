"""
utils/augmentations.py — Train / Val / TTA transform pipelines.
No vertical flips — they break leaf anatomical realism.
"""

import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(p=0.5),
        # NO vertical flip — breaks leaf anatomy
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomPerspective(distortion_scale=0.3, p=0.3),  # simulates phone camera angles
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
        T.RandomGrayscale(p=0.1),  # forces color-independent disease features
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
    ])


def get_val_transforms(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms(image_size: int = 224) -> list:
    """
    Test Time Augmentation — returns a list of transforms.
    Run inference with each, average the probabilities.
    5 views: original + 4 augmented variants.
    """
    base = [
        T.Resize(256),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return [
        # 1. Original (center crop)
        T.Compose(base),
        # 2. Horizontal flip
        T.Compose([T.Resize(256), T.CenterCrop(image_size),
                   T.RandomHorizontalFlip(p=1.0),
                   T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        # 3. Slight zoom in
        T.Compose([T.Resize(288), T.CenterCrop(image_size),
                   T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        # 4. Slight color shift
        T.Compose([T.Resize(256), T.CenterCrop(image_size),
                   T.ColorJitter(brightness=0.2, contrast=0.2),
                   T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        # 5. Five-crop center
        T.Compose([T.Resize(256), T.FiveCrop(image_size),
                   T.Lambda(lambda crops: crops[0]),  # use center crop
                   T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    ]