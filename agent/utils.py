import numpy as np


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
