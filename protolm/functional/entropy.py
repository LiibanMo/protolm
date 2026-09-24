import torch


def get_entropy_contributions(inputs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Calculate the entropy of the probability distribution from input scores.

    Args:
        inputs: The tensor of scores whose predictive entropy is calculated.

    Returns:
        The entropy contributions of the input tensor.
    """
    log_probs = torch.log_softmax(inputs, dim)
    probs = torch.softmax(inputs, dim)
    return -(probs * log_probs).sum(dim=dim)
