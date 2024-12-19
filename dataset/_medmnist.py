import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import Dataset

from .core import DatasetInfo, DatasetPack, register_dataset_factory


@register_dataset_factory(name="medmnist")
def dataset_factory(name: str = "pathmnist") -> DatasetPack:
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    target_transform = transforms.Lambda(lambda x: x.squeeze())
    train_dataset = DataClass(
        split="train", transform=transform, target_transform=target_transform, download=True
    )
    val_dataset = DataClass(
        split="val", transform=transform, target_transform=target_transform, download=True
    )
    test_dataset = DataClass(
        split="test", transform=transform, target_transform=target_transform, download=True
    )

    infos = DatasetInfo(
        name="medmnist",
        num_classes=9,
        image_size=(info["n_channels"], 28, 28),
    )
    return train_dataset, val_dataset, test_dataset, infos
