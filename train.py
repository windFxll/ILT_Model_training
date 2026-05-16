import argparse
from pathlib import Path
import random
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import yaml

from datasets import ILTDataset
from losses import CombinedLoss
from models.registry import MODEL_REGISTRY


# ==================================================
# Logger
# ==================================================

def setup_logger(output_dir):

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "train.log"

    logger = logging.getLogger("train_logger")

    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger, log_file


# ==================================================
# Save config
# ==================================================

def save_config(config, log_dir):

    config_path = log_dir / "config.yaml"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    return config_path


# ==================================================
# Args
# ==================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_unet.yaml",
    )

    return parser.parse_args()


# ==================================================
# Load config
# ==================================================

def load_config(config_path):

    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==================================================
# Device
# ==================================================

def resolve_device(device_name):

    if device_name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return torch.device(device_name)


# ==================================================
# Seed
# ==================================================

def set_seed(seed):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================================================
# Build model
# ==================================================

def build_model(model_name):

    key = str(model_name).lower()

    if key not in MODEL_REGISTRY:

        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[key]()


# ==================================================
# Gaussian Blur
# ==================================================

def gaussian_blur(x):

    kernel = torch.tensor(
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ],
        dtype=x.dtype,
        device=x.device,
    )

    kernel = kernel / kernel.sum()

    kernel = kernel.view(1, 1, 3, 3)

    kernel = kernel.repeat(x.shape[1], 1, 1, 1)

    x = F.conv2d(
        x,
        kernel,
        padding=1,
        groups=x.shape[1],
    )

    return x


# ==================================================
# Main
# ==================================================

def main():

    print()

    args = parse_args()

    config = load_config(args.config)

    criterion = CombinedLoss(config["loss"])

    project_root = Path(__file__).resolve().parent

    seed = int(config.get("seed", 42))

    set_seed(seed)

    device = resolve_device(config.get("device", "auto"))

    epochs = int(config["epochs"])

    batch_size = int(config["batch_size"])

    log_every = int(config.get("log_every", 1))

    num_workers = int(config.get("num_workers", 0))

    lr = float(config["lr"])

    # ==================================================
    # Dataset
    # ==================================================

    data_root = Path(config["data_root"])

    dataset = ILTDataset(str(data_root))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # ==================================================
    # Output
    # ==================================================

    output_root = Path(config["output_root"])

    output_dir = output_root / config["output_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    logger, log_file = setup_logger(output_dir)

    log_dir = output_dir / "logs"

    config_path = save_config(config, log_dir)

    checkpoint_dir = output_dir / "checkpoints"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Init] Config saved to: {config_path}")

    logger.info(f"[Init] Device: {device}")

    logger.info(f"[Init] Dataset size: {len(dataset)}")

    logger.info(f"[Init] Batches/epoch: {len(loader)}")

    # ==================================================
    # Inverse Model
    # ==================================================

    inverse_model = build_model(
        config["model"]
    ).to(device)

    # ==================================================
    # Forward Model
    # ==================================================

    forward_model = build_model(
        config["forward_model"]
    ).to(device)

    forward_ckpt_path = (
        project_root / config["forward_checkpoint"]
    )

    logger.info(
        f"[Init] Loading Forward Model: {forward_ckpt_path}"
    )

    ckpt = torch.load(
        str(forward_ckpt_path),
        map_location=device,
    )

    # 支持两种保存格式
    if "model_state_dict" in ckpt:
        forward_model.load_state_dict(
            ckpt["model_state_dict"]
        )
    else:
        forward_model.load_state_dict(ckpt)

    forward_model.eval()

    for p in forward_model.parameters():
        p.requires_grad = False

    logger.info("[Init] Forward model frozen")

    # ==================================================
    # Optimizer
    # ==================================================

    optimizer = optim.Adam(
        inverse_model.parameters(),
        lr=lr,
        weight_decay=float(
            config.get("weight_decay", 1e-5)
        ),
    )

    # ==================================================
    # Training
    # ==================================================

    best_loss = float("inf")

    last_completed_epoch = -1

    last_epoch_loss = None

    def save_checkpoint(
        path,
        epoch_idx,
        epoch_loss,
        is_interrupt=False,
    ):

        torch.save(
            {
                "epoch": epoch_idx,
                "model_state_dict": inverse_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch_loss": epoch_loss,
                "is_interrupt": is_interrupt,
            },
            str(path),
        )
    # ==============================================
    # Save Initial Checkpoint (Epoch 0)
    # ==============================================

    init_ckpt_path = (
        checkpoint_dir / "checkpoint_epoch_000.pt"
    )

    save_checkpoint(
        init_ckpt_path,
        epoch_idx=-1,
        epoch_loss=None,
    )

    logger.info(
        f"[Save] Initial checkpoint -> "
        f"{init_ckpt_path}"
    )
    try:

        for epoch in range(epochs):

            inverse_model.train()

            total_loss = 0.0

            logger.info(
                f"\n[Epoch {epoch + 1}/{epochs}] Start"
            )

            for batch_idx, target in enumerate(
                loader,
                start=1,
            ):

                target = target.to(device)

                # ==========================================
                # Inverse Lithography
                # ==========================================

                # 网络输出 correction（增量修正）
                delta = inverse_model(target)
                delta = 0.5 * torch.tanh(delta)

                # ------------------------------------------
                # 初始化 mask logits
                # target:
                #   0 -> -5
                #   1 -> +5
                #
                # 这样 sigmoid 后：
                # 初始 mask ≈ target
                # ------------------------------------------

                base_logits = (target * 2.0 - 1.0) * 1.5

                # 最终 logits
                delta_scale = 1.0
                mask_logits = base_logits + delta_scale * delta
                delta_min = delta.min().item()
                delta_max = delta.max().item()
                delta_mean = delta.mean().item()

                # 日志统计
                mask_min = mask_logits.min().item()
                mask_max = mask_logits.max().item()
                mask_mean = mask_logits.mean().item()

                # sigmoid
                mask_prob = torch.sigmoid(mask_logits)

                prob_min = mask_prob.min().item()
                prob_max = mask_prob.max().item()
                prob_mean = mask_prob.mean().item()
                
                # # ==========================================
                # # Inverse Lithography
                # # ==========================================

                # mask_pred = inverse_model(target)
                
                # mask_min = mask_pred.min().item()
                # mask_max = mask_pred.max().item()
                # mask_mean = mask_pred.mean().item()

                # mask_prob = torch.sigmoid(mask_pred / 5.0)
                # prob_min = mask_prob.min().item()
                # prob_max = mask_prob.max().item()
                # prob_mean = mask_prob.mean().item()
                
                # mask_prob = gaussian_blur(mask_prob)

                resist_pred = forward_model(mask_prob)

                # ==========================================
                # Loss
                # ==========================================

                loss, loss_dict = criterion(
                    resist_pred,
                    target,
                    mask_prob,
                )

                # ==========================================
                # Backward
                # ==========================================

                optimizer.zero_grad()

                loss.backward()
                grad_mean = 0.0
                grad_count = 0

                for p in inverse_model.parameters():

                    if p.grad is not None:

                        grad_mean += p.grad.abs().mean().item()
                        grad_count += 1

                if grad_count > 0:
                    grad_mean /= grad_count
                    
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % log_every == 0:

                    log_str = " ".join(
                        [
                            f"{k}:{v:.4f}"
                            for k, v in loss_dict.items()
                        ]
                    )

                    logger.info(
                        f"[Epoch {epoch+1}] "
                        f"Batch {batch_idx}/{len(loader)} "
                        f"Total={loss.item():.4f} "
                        f"{log_str}"
                        f"| logits[min={mask_min:.2f}, "
                        f"max={mask_max:.2f}, "
                        f"mean={mask_mean:.2f}] "
                        f"| prob[min={prob_min:.2f}, "
                        f"max={prob_max:.2f}, "
                        f"mean={prob_mean:.2f}]"
                        f" | grad={grad_mean:.6f}"
                        f"| delta[min={delta_min:.4f}, "
                        f"max={delta_max:.4f}, "
                        f"mean={delta_mean:.4f}] "
                    )

            # ==============================================
            # Epoch End
            # ==============================================

            epoch_loss = total_loss / len(loader)

            logger.info(
                f"[Epoch {epoch + 1}/{epochs}] "
                f"Avg Loss = {epoch_loss:.6f}"
            )

            save_every = int(
                config.get("save_every", 5)
            )

            if (epoch + 1) % save_every == 0:

                epoch_ckpt_path = (
                    checkpoint_dir
                    / f"checkpoint_epoch_{epoch + 1:03d}.pt"
                )

                save_checkpoint(
                    epoch_ckpt_path,
                    epoch,
                    epoch_loss,
                )

                logger.info(
                    f"[Save] Epoch checkpoint -> "
                    f"{epoch_ckpt_path}"
                )

            if epoch_loss < best_loss:

                best_loss = epoch_loss

                best_ckpt_path = (
                    checkpoint_dir / "best_model.pt"
                )

                save_checkpoint(
                    best_ckpt_path,
                    epoch,
                    best_loss,
                )

                logger.info(
                    f"[Save] Best model updated "
                    f"(loss={best_loss:.6f}) "
                    f"-> {best_ckpt_path}"
                )

            last_completed_epoch = epoch

            last_epoch_loss = epoch_loss

    except KeyboardInterrupt:

        interrupt_ckpt_path = (
            checkpoint_dir
            / "interrupt_checkpoint.pt"
        )

        save_checkpoint(
            interrupt_ckpt_path,
            last_completed_epoch,
            last_epoch_loss,
            is_interrupt=True,
        )

        logger.info(
            f"\n[Interrupt] Training interrupted. "
            f"Checkpoint saved to "
            f"{interrupt_ckpt_path}"
        )

        raise

    finally:

        final_model_path = (
            checkpoint_dir / "last_model.pt"
        )

        torch.save(
            inverse_model.state_dict(),
            str(final_model_path),
        )

        logger.info(
            f"[Done] Last model weights saved to "
            f"{final_model_path}"
        )


if __name__ == "__main__":
    main()