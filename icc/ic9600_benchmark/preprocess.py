from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 512, augment: bool = True) -> transforms.Compose:
    ops = [transforms.Resize((img_size, img_size))]
    if augment:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transforms.Compose(ops)


def get_eval_transforms(img_size: int = 512) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
