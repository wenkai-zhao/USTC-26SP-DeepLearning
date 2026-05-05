import torch
import torch.nn.functional as F


def info_nce_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    if z_i.shape != z_j.shape:
        raise ValueError("z_i and z_j must have the same shape.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    similarity = torch.matmul(z, z.T) / temperature

    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    similarity = similarity.masked_fill(self_mask, float("-inf"))

    positives = torch.arange(2 * batch_size, device=z.device)
    positives = (positives + batch_size) % (2 * batch_size)
    return F.cross_entropy(similarity, positives)


def contrastive_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.5,
    margin: float = 1.0,
) -> torch.Tensor:
    del temperature
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    positives = F.pairwise_distance(z_i, z_j).pow(2)
    negatives = F.pairwise_distance(z_i, z_j.roll(shifts=1, dims=0))
    negatives = F.relu(margin - negatives).pow(2)
    return 0.5 * (positives.mean() + negatives.mean())


def triplet_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.5,
    margin: float = 0.2,
) -> torch.Tensor:
    del temperature
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    negatives = z_j.roll(shifts=1, dims=0)
    return F.triplet_margin_loss(z_i, z_j, negatives, margin=margin)


def build_contrastive_loss(name: str):
    if name == "ntxent":
        return info_nce_loss
    if name == "contrastive":
        return contrastive_loss
    if name == "triplet":
        return triplet_loss
    raise ValueError("loss must be one of: ntxent, contrastive, triplet.")
