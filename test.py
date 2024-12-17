import medmnist
from medmnist import INFO

# from .core import DatasetInfo, DatasetPack, DATASET_DIR, register_dataset_factory
# from .utils import split_dataset


# @register_dataset_factory(name="medmnist")
# def dataset_factory(name: str = "pathmnist", val_ratio: float = 0.2) -> DatasetPack:
#     dataset = medmnist.load(name, DATASET_DIR)
#     train_dataset, val_dataset = split_dataset(dataset, val_ratio)
#     infos = DatasetInfo(
#         name=name,
#         num_classes=INFO[name]["label_num"],
#         image_size=INFO[name]["img_shape"],
#     )
#     return train_dataset, val_dataset, dataset, infos


if __name__ == "__main__":
    data_flag = "pathmnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    import torchvision.transforms as transforms

    data_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )
    train_dataset = DataClass(split="train", transform=data_transform, download=True)
    test_dataset = DataClass(split="test", transform=data_transform, download=True)

    print(len(train_dataset), train_dataset[0][0].shape, train_dataset[0][1].shape)
    print(len(test_dataset), type(test_dataset[0]))
