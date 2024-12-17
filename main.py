import torch
import torch.nn as nn
from typing import Tuple, List
from config import default_config
from dataset import DatasetFactory
from model import get_train_stuff
from utils import get_per_sample_gradient, unflatten_gradient
import shapley
import metrics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def server_update(model: nn.Module, optimizer, per_sample_gradients):
    # now just use the mean gradients
    mean_gradients = per_sample_gradients.mean(dim=0)
    named_gradients = unflatten_gradient(model, mean_gradients)

    # update the model
    optimizer.zero_grad()
    for name, param in model.named_parameters():
        if name not in named_gradients:
            raise ValueError(f"gradient for {name} not found")
        param.grad = named_gradients[name]
    optimizer.step()


def server_warmup(model, optimizer, criterion, dataset):
    config = default_config()
    logger.info(f"start warmup with {config.warmup_type} for {config.warmup_epochs} epochs")
    dataset_for_train = DatasetFactory.get_warmup_dataset(
        dataset, config.target_label, exclusive=(config.warmup_type == "exclusive")
    )
    model.train()
    for epoch in range(config.warmup_epochs):
        logger.info(f"warmup epoch: {epoch}")
        for per_sample_gradients, _ in get_per_sample_gradient(model, dataset_for_train, criterion):
            server_update(model, optimizer, per_sample_gradients)


def compute_shapley_values_per_classes(
    model, per_sample_gradients, dataset_per_label, criterion
) -> List[torch.Tensor]:
    result = []
    for i, dataset in enumerate(dataset_per_label):
        true_gradients = shapley.compute_true_gradients(model, dataset, criterion)
        shapley_values = shapley.monte_carlo_shapley(
            per_sample_gradients, true_gradients, num_samples=25
        )
        result.append(shapley_values)
    return result


def compute_accuracy_over_classes(
    model: nn.Module, dataset, num_classes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config = default_config()
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024)
    count = torch.zeros(num_classes)
    correct = torch.zeros(num_classes)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(num_classes):
                count[i] += (labels == i).sum().item()
                correct[i] += ((predicted == labels) & (labels == i)).sum().item()
    return count, correct, correct / count


def main():
    config = default_config()
    logger.info(f"config: {config}")
    train_dataset, val_dataset, test_dataset, infos = DatasetFactory.get_dataset_by_name(
        config.dataset
    )
    if config.num_clients_per_run > len(train_dataset):
        raise ValueError("to many client in a single run")

    model, optimizer, criterion = get_train_stuff(infos)
    model = model.to(config.device)

    if all(x is not None for x in [config.warmup_type, config.warmup_epochs, config.target_label]):
        server_warmup(model, optimizer, criterion, val_dataset)
        count, correct, accuracy = compute_accuracy_over_classes(
            model, val_dataset, infos.num_classes
        )
        logger.info(f"after warmup, count: {count}")
        logger.info(f"after warmup, correct: {correct}")
        logger.info(f"after warmup, accuracy: {accuracy}")

    client_datasets = DatasetFactory.get_client_datasets(train_dataset, sample_per_client=1)
    dataset_per_label = DatasetFactory.group_dataset_by_label(val_dataset, infos.num_classes)
    logger.info(
        f"start training with {len(client_datasets)} clients and {config.num_epochs} epochs"
    )
    for epoch in range(config.num_epochs):
        logger.info(f"epoch: {epoch}")
        model.train()
        for per_sample_gradients, dataset_for_train in get_per_sample_gradient(
            model,
            torch.utils.data.ConcatDataset(client_datasets),
            criterion,
            batch_size=config.num_clients_per_run,
        ):
            shapley_values_per_classes = compute_shapley_values_per_classes(
                model, per_sample_gradients, dataset_per_label, criterion
            )
            # calculate roc
            roc_results = metrics.calculate_roc_over_classes(
                dataset_for_train, shapley_values_per_classes
            )
            metrics.plot_roc_curve_over_classes(roc_results)
            server_update(model, optimizer, per_sample_gradients)

        # evaluate accuracy
        count, correct, accuracy = compute_accuracy_over_classes(
            model, test_dataset, infos.num_classes
        )
        logger.info(f"count: {count}")
        logger.info(f"correct: {correct}")
        logger.info(f"accuracy: {accuracy}")


if __name__ == "__main__":
    main()
