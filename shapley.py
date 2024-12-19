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

    norm_gradients1 = torch.linalg.norm(gradients1)
    norm_gradients2 = torch.linalg.norm(gradients2)

    # I don't know which station this condition will be happen, but I handle it like this way
    if norm_gradients1 == 0 or norm_gradients2 == 0 or (norm_gradients1 * norm_gradients2) == 0:
        return torch.tensor(0.0)

    # calculate cosine similarity between two vector lists
    similarity = torch.dot(gradients1, gradients2) / (norm_gradients1 * norm_gradients2)
    return similarity


def _evaluate_function(mean_gradients, true_gradients):
    # calculate the cosine similarity between the real denoised gradient and the subset
    similarity = _cosine_similarity(mean_gradients, true_gradients)
    return similarity


def compute_true_gradients(model: nn.Module, dataset: Dataset, criterion):
    num_samples = len(dataset)
    true_gradients_sum = None
    for per_sample_gradients, _ in get_per_sample_gradient(model, dataset, criterion):
        if true_gradients_sum is None:
            true_gradients_sum = per_sample_gradients.sum(dim=0)
        else:
            true_gradients_sum += per_sample_gradients.sum(dim=0)
    true_gradients = true_gradients_sum / num_samples
    return true_gradients


# Monte Carlo Shapley computation
def monte_carlo_shapley(values_list, true_gradients, num_samples):
    n = len(values_list)
    shapley_values = torch.zeros(n, device=values_list.device)
    for si in range(num_samples):
        permuted_indices = torch.randperm(n)
        subset_value = 0
        permuted_cum_values_list = torch.cumsum(values_list[permuted_indices], dim=0)
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
    sum_shapley_values = shapley_values.sum()
    # NOTE: when in the config with warmup_type = inclusive and validation set with target label being use to compute the true gradients, the sum of shapley values can be 0
    if sum_shapley_values == 0:
        return shapley_values
    shapley_values /= shapley_values.sum()
    # print(shapley_values.min(), shapley_values.max(), shapley_values.mean())
    return shapley_values
