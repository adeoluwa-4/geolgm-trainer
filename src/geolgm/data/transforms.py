from __future__ import annotations

from torchvision import transforms


def build_transforms(image_size: int, split: str):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    resize = transforms.Resize((image_size, image_size))

    if split == "train":
        return transforms.Compose(
            [
                resize,
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_size, padding=max(4, image_size // 16)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            resize,
            transforms.ToTensor(),
            normalize,
        ]
    )
