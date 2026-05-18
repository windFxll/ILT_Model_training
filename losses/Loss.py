import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# BCE Loss
# ==================================================

def bce_loss(pred, target):

    return F.binary_cross_entropy_with_logits(
        pred,
        target,
    )


# ==================================================
# Dice Loss
# ==================================================

def dice_loss(pred, target, eps=1e-6):

    pred = torch.sigmoid(pred)

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)

    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)

    return 1.0 - dice.mean()


# ==================================================
# MSE Loss
# ==================================================

def mse_loss(pred, target):

    pred = torch.sigmoid(pred)

    return F.mse_loss(pred, target)


# ==================================================
# TV Loss
# ==================================================

def tv_loss(pred):

    pred = torch.sigmoid(pred)

    dh = torch.mean(
        torch.abs(
            pred[:, :, 1:, :] - pred[:, :, :-1, :]
        )
    )

    dw = torch.mean(
        torch.abs(
            pred[:, :, :, 1:] - pred[:, :, :, :-1]
        )
    )

    return dh + dw


# ==================================================
# Laplacian Loss
# ==================================================

def laplacian_loss(pred):

    pred = torch.sigmoid(pred)

    lap = (
        -4 * pred
        + torch.roll(pred, 1, dims=2)
        + torch.roll(pred, -1, dims=2)
        + torch.roll(pred, 1, dims=3)
        + torch.roll(pred, -1, dims=3)
    )

    return lap.abs().mean()

# ==================================================
# Binary Loss
# ==================================================

def binary_loss(pred, eps=1e-6):
    
    pred = torch.sigmoid(pred)

    loss = -(
        pred * torch.log(pred + eps)
        + (1 - pred) * torch.log(1 - pred + eps)
    )

    return loss.mean()


# ==================================================
# Loss Registry
# ==================================================

LOSS_REGISTRY = {
    "bce": bce_loss,
    "dice": dice_loss,
    "mse": mse_loss,
    "tv": tv_loss,
    "lap": laplacian_loss,
    "binary": binary_loss,
}


# ==================================================
# Combined Loss
# ==================================================

class CombinedLoss(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.loss_items = []

        weights = config.get("weights", {})

        for name, weight in weights.items():

            if name not in LOSS_REGISTRY:

                raise ValueError(
                    f"Unknown loss: {name}"
                )

            self.loss_items.append(
                (LOSS_REGISTRY[name], weight, name)
            )

    def forward(
        self,
        pred,
        target,
        mask_prob=None,
    ):

        total_loss = 0.0

        loss_dict = {}

        for fn, weight, name in self.loss_items:

            # ==========================================
            # Regularization Loss
            # ==========================================

            if name in ["tv", "lap", "binary"]:

                loss = fn(pred)

            # ==========================================
            # Reconstruction Loss
            # ==========================================

            else:

                loss = fn(pred, target)

            total_loss += weight * loss

            loss_dict[name] = loss.item()

        return total_loss, loss_dict