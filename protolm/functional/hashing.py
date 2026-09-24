import torch


def roll_poly_hash(
    n_grams: torch.Tensor, num_buckets: int, p: int = 1_000_000_007
) -> torch.Tensor:
    """Map each n-gram to a bucket using a rolling polynomial hash.

    Args:
        n_grams: The tensor of n-grams, with tokens along the last dimension.
        num_buckets: The number of buckets available for the hash values.
        p: The polynomial multiplier used to update the rolling hash.

    Returns:
        A tensor containing one bucket index for each n-gram.
    """
    if n_grams.dtype != torch.long:
        raise TypeError(
            f"n_grams must contain torch.long byte IDs. Got type {n_grams.dtype}"
        )

    if n_grams.ndim < 1:
        raise ValueError("n_grams must have an n-gram dimension")

    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")

    result = torch.zeros(n_grams.shape[:-1], dtype=torch.long, device=n_grams.device)

    multiplier = p % num_buckets
    for byte_ids_at_position in n_grams.unbind(-1):
        result = (result * multiplier + byte_ids_at_position) % num_buckets

    return result
