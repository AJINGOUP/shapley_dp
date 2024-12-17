import torch
import torch.nn as nn
from typing import Tuple
from dataset import DatasetInfo
from config import default_config


class SimpleConvNet(nn.Module):
    def __init__(self, image_size: Tuple[int, int, int], num_classes: int):
        super(SimpleConvNet, self).__init__()
        in_channels, width, height = image_size
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * width * height // 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def get_train_stuff(infos: DatasetInfo):
    if infos.name not in ["mnist", "adult", "medmnist"]:
        raise ValueError(f"not support for dataset: {infos.name}")

    config = default_config()
    if infos.name == "mnist":
        model = SimpleConvNet(infos.image_size, infos.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif infos.name == "adult":
        model = SimpleMLP(infos.vector_size, infos.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif infos.name == "medmnist":
        model = SimpleConvNet(infos.image_size, infos.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    return (
        model,
        optimizer,
        criterion,
    )
