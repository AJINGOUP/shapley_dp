import torch
from torch.func import functional_call, vmap, grad
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Mapping, Iterable, Tuple
from config import default_config


def flatten_gradient(gradient: Mapping[str, torch.Tensor]) -> torch.Tensor:
    flat_grad = []
    for gradient_value in gradient.values():
        flat_grad.append(gradient_value.view(-1))
    return torch.cat(flat_grad)


def unflatten_gradient(model: nn.Module, flat_gradient: torch.Tensor) -> Mapping[str, torch.Tensor]:
    grad_dict = {}
    start = 0
    for name, param in model.named_parameters():
        end = start + param.numel()
        grad_dict[name] = flat_gradient[start:end].view(param.shape)
        start = end
    return grad_dict


def get_per_sample_gradient(
    model: nn.Module,
    dataset: Dataset,
    criterion,
    *,
    batch_size: int = 1024,
    shuffle: bool = True,
    flatten: bool = True,
) -> Iterable[Tuple[torch.Tensor, Dataset]]:
    config = default_config()

    def compute_loss(parmas, sample, target):
        batch = sample.unsqueeze(0).to(config.device)
        targets = target.unsqueeze(0).to(config.device)
        predictions = functional_call(model, parmas, batch)
        loss = criterion(predictions, targets)
        return loss

    compute_grad = grad(compute_loss)
    compute_sample_grad = vmap(compute_grad, in_dims=(None, 0, 0))

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    params = {name: param.detach() for name, param in model.named_parameters()}
    for sample, target in data_loader:
        per_sample_grads = compute_sample_grad(params, sample, target)
        if flatten:
            per_sample_grads = vmap(flatten_gradient)(per_sample_grads)
        yield per_sample_grads, TensorDataset(sample, target)
