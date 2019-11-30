import torch
import numpy as np

NUM_BINS = 10


def copy_weights(from_model, to_model, change_ratio: float = 1):
    """
    Move parameters of `from_model` to `to_model`

    when change_ratio = .9:
        10% of weights come from `to_model`
        90% of weights come from `from_model`
    when keep change_ratio = 1:
        all weights come from `from_model`
    """
    for other_weights, orig_weights in zip(from_model.parameters(), to_model.parameters()):
        if change_ratio == 1:
            result_weights = other_weights.data.clone()
        else:
            result_weights = (other_weights.data * change_ratio +
                              orig_weights.data * (1 - change_ratio))
        orig_weights.data.copy_(result_weights)


def softmax(x: np.array) -> np.array:
    """
    More numerically stable softmax

    Args:
        x: float array of shape [N,]

    Returns:
        float array of shape [N,]
    """
    x = np.exp(x - max(x))
    return x / (sum(x) + 1e-9)


def kl_divergence(mean, log_var):
    """ From N(mean, log_var^2) to U(0, 1) """
    return -(1 + log_var - mean ** 2 - log_var.exp())


def pmf(values: [float], num_bins=NUM_BINS, sigma=3.) -> [float]:
    """
    Soft histogram (gaussian)

    differentiable

    https://discuss.pytorch.org/t/differentiable-torch-histc/25865

    Args:
        values: float tensor of shape [n,]  # TODO make it take batch size

    Returns:
        float tensor of shape [num_bins,] pmf of `values`
    """
    with torch.no_grad():
        vmin = values.min()
        vmax = values.max()

    delta = (vmax - vmin) / num_bins
    centers = vmin + delta * (torch.arange(num_bins) + .5)

    x = values.unsqueeze(dim=0) - centers.unsqueeze(dim=1)
    x = (torch.sigmoid(sigma * (x + delta / 2)) -
         torch.sigmoid(sigma * (x - delta / 2)))
    x = x.sum(dim=1)

    x /= sum(x)  # make it sum one

    return x


def cross_entropy(p, q, epsilon=1e-9):
    """
    Args:
        p: float tensor of shape [batch_size, n]
        q: float tensor of shape [batch_size, n]
        epsilon: small float to combat zero probability

    Returns
        cross entropy batch-wise: float tensor of shape [batch_size,]

    """
    return -(p * (q + epsilon).log()).sum(1)


def entropy(p):
    """
    Args:
        p: float tensor of shape [batch_size, n]
    Returns
        float tensor of shape [batch_size,]
    """
    return cross_entropy(p, p)


def pmf_mi(inputs, latent) -> float:
    """
    Computes number of bits to encode underlying events using `latent` rather than `inputs`

    Args:
        inputs: float tensor of shape [n,]
        latent: float tensor of shape [m,] with grad
    """
    pmf_inputs = torch.stack([pmf(x.flatten()) for x in inputs])
    pmf_latent = torch.stack([pmf(x.flatten()) for x in latent])
    return cross_entropy(pmf_inputs, pmf_latent) - entropy(pmf_inputs)
