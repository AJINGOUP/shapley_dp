import torch
import torch.nn as nn
from torch.utils.data import Dataset
from utils import get_per_sample_gradient


def _cosine_similarity(gradients1: torch.Tensor, gradients2: torch.Tensor):
    # assert gradients1 and gradients2 are 1D tensors
    if len(gradients1.shape) != 1 or len(gradients2.shape) != 1:
        raise ValueError("gradients1 and gradients2 must be 1D tensors")

    # assert gradients1 and gradients2 have the same length
    if gradients1.shape != gradients2.shape:
        raise ValueError("gradients1 and gradients2 must have the same length")

    # calculate cosine similarity between two vector lists
    similarity = torch.dot(gradients1, gradients2) / (
        torch.linalg.norm(gradients1) * torch.linalg.norm(gradients2)
    )
    return similarity


def _evaluate_function(mean_gradients, true_gradients):
    # calculate the cosine similarity between the real denoised gradient and the subset
    similarity = _cosine_similarity(mean_gradients, true_gradients)
    return similarity


def compute_true_gradients(model: nn.Module, dataset: Dataset, criterion):
    true_gradients = []
    for per_sample_gradients, _ in get_per_sample_gradient(model, dataset, criterion):
        true_gradients.append(per_sample_gradients)
    true_gradients = torch.cat(true_gradients).mean(dim=0)
    return true_gradients


# Monte Carlo Shapley computation
def monte_carlo_shapley(values_list, true_gradients, num_samples):
    n = len(values_list)
    shapley_values = torch.zeros(n, device=values_list.device)
    cum_values_list = torch.cumsum(values_list, dim=0)
    for si in range(num_samples):
        permuted_indices = torch.randperm(n)
        subset_value = 0
        permuted_cum_values_list = cum_values_list[permuted_indices]
        for i in range(n):
            new_subset_value = _evaluate_function(
                (permuted_cum_values_list[i]) / (i + 1), true_gradients
            )
            marginal_contribution = new_subset_value - subset_value
            shapley_values[permuted_indices[i]] += marginal_contribution
            subset_value = new_subset_value

    shapley_values /= num_samples
    # set shapley_values to 0 if the value is negative
    shapley_values[shapley_values <= 0] = 0
    shapley_values /= shapley_values.sum()
    return shapley_values
