import torch
import torch.nn as nn
from typing import Tuple, List, Union
from torch.utils.data import Dataset
from config import default_config
from dataset import DatasetFactory
from model import get_train_stuff
from utils import get_per_sample_gradient, unflatten_gradient
import shapley
import metrics
import logging
import profiler

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


# maybe we need diffrent learning rate in this process?
def server_warmup(model, optimizer, criterion, dataset):
    config = default_config()
    logger.info(f"start warmup with {config.warmup_type} for {config.warmup_epochs} epochs")
    dataset_for_train = DatasetFactory.get_warmup_dataset(
        dataset, config.target_label, exclusive=(config.warmup_type == "exclusive")
    )
    model.train()
    for epoch in range(config.warmup_epochs):
        logger.info(f"warmup epoch: {epoch}")
        for per_sample_gradients, _ in get_per_sample_gradient(
            model, dataset_for_train, criterion, batch_size=256
        ):
            server_update(model, optimizer, per_sample_gradients)


def compute_shapley_values_for_single_dataset(
    model: nn.Module, per_sample_gradients: torch.Tensor, dataset: Dataset, criterion
) -> torch.Tensor:
    true_gradients = shapley.compute_true_gradients(model, dataset, criterion)
    shapley_values = shapley.monte_carlo_shapley(
        per_sample_gradients, true_gradients, num_samples=25
    )
    return shapley_values


@profiler.with_time_profile
def compute_shapley_values(
    model: nn.Module,
    per_sample_gradients: torch.Tensor,
    dataset: Union[List[Dataset], Dataset],
    criterion,
) -> Union[List[torch.Tensor], torch.Tensor]:
    if isinstance(dataset, Dataset):
        return compute_shapley_values_for_single_dataset(
            model, per_sample_gradients, dataset, criterion
        )
    return [
        compute_shapley_values_for_single_dataset(
            model, per_sample_gradients, single_dataset, criterion
        )
        for single_dataset in dataset
    ]


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


def fix_all_randomness(seed: int):
    import numpy as np
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main():
    config = default_config()

    if config.seed is not None:
        fix_all_randomness(config.seed)

    logger.info(f"config: {config}")
    train_dataset, val_dataset, test_dataset, infos = DatasetFactory.get_dataset_by_name(
        config.dataset
    )
    if config.num_clients_per_round > len(train_dataset):
        raise ValueError("to many client in a single round")

    model, optimizer, criterion = get_train_stuff(infos)
    model = model.to(config.device)

    with_warmup = all(x is not None for x in [config.warmup_type, config.warmup_epochs])
    if with_warmup:
        RESULT_DIR = (
            default_config().result_dir
            / f"{infos.name}"
            / "unbiased"
            / (f"{config.warmup_type}" + "-" f"{config.target_label}")
        )
        server_warmup(model, optimizer, criterion, val_dataset)
        count, correct, accuracy = compute_accuracy_over_classes(
            model, val_dataset, infos.num_classes
        )
        logger.info(f"after warmup, count: {count}")
        logger.info(f"after warmup, correct: {correct}")
        logger.info(f"after warmup, accuracy: {accuracy}")
    else:
        RESULT_DIR = default_config().result_dir / f"{infos.name}" / "biased"

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "config.txt", "w") as f:
        f.write(str(config))

    client_datasets = DatasetFactory.get_client_datasets(train_dataset, sample_per_client=1)
    dataset_per_label = DatasetFactory.group_dataset_by_label(val_dataset, infos.num_classes)
    logger.info(
        f"start training with {len(client_datasets)} clients and {config.num_epochs} epochs"
    )
    round_so_far = 0
    for epoch in range(config.num_epochs):
        model.train()
        for round, (per_sample_gradients, dataset_for_train) in enumerate(
            get_per_sample_gradient(
                model,
                torch.utils.data.ConcatDataset(client_datasets),
                criterion,
                batch_size=config.num_clients_per_round,
            )
        ):
            logger.info(f"round: {round_so_far}")

            # we asume that when with_warmup is true, you would like to run experiment with unbiased shapley values
            # otherwise, you would like to run experiment with biased shapley values

            if with_warmup:
                # use val dataset to calculate unbiased shapley values
                unbaised_shapley_values = compute_shapley_values(
                    model, per_sample_gradients, val_dataset, criterion
                )

                # calculate unbiased roc result
                save_path = RESULT_DIR / f"roc_{round_so_far}.png"
                unbaised_roc_results = metrics.calculate_roc_over_classes(
                    dataset_for_train, unbaised_shapley_values, num_classes=infos.num_classes
                )
                metrics.plot_roc_curve_over_classes(unbaised_roc_results, save_path=save_path)
            else:
                # use dataset group by label to calculate biased shapley values
                biased_shapley_values = compute_shapley_values(
                    model, per_sample_gradients, dataset_per_label, criterion
                )

                # calculate biased roc result
                save_path = RESULT_DIR / f"roc_{round_so_far}.png"
                biased_roc_results = metrics.calculate_roc_over_classes(
                    dataset_for_train, biased_shapley_values, num_classes=infos.num_classes
                )
                metrics.plot_roc_curve_over_classes(biased_roc_results, save_path=save_path)

            # server update global model for this single round
            server_update(model, optimizer, per_sample_gradients)
            round_so_far += 1
            if config.max_rounds is not None and round_so_far >= config.max_rounds:
                break

    # evaluate accuracy
    count, correct, accuracy = compute_accuracy_over_classes(model, test_dataset, infos.num_classes)
    logger.info(f"count: {count}")
    logger.info(f"correct: {correct}")
    logger.info(f"accuracy: {accuracy}")


if __name__ == "__main__":
    main()
