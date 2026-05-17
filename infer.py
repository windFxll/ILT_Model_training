import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
import yaml

from models.registry import MODEL_REGISTRY
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name under experiments_ILT"
    )

    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(model_path, model_name, device):

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    model = MODEL_REGISTRY[model_name]()

    ckpt = torch.load(model_path, map_location=device)

    state_dict = ckpt.get("model_state_dict", ckpt)

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def preprocess(img_path):

    img = cv2.imread(str(img_path), 0)

    if img is None:
        raise RuntimeError(f"Failed to read {img_path}")

    img = cv2.resize(
        img,
        (480, 480),
        interpolation=cv2.INTER_NEAREST
    )

    img = img / 255.0

    x = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    return x

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

def postprocess(pred):

    pred = pred.squeeze().cpu().numpy()

    pred = (pred > 0.5).astype(np.uint8) * 255

    return pred


def main():

    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ==================================================
    # 路径
    # ==================================================

    project_root = Path(__file__).resolve().parent

    DRIVE_ROOT = Path(
        r"/content/drive/MyDrive/Colab Notebooks"
    )

    exp_name = args.exp_name

    exp_root = (
        DRIVE_ROOT
        / "experiments_ILT"
        / exp_name
    )

    checkpoint_dir = exp_root / "checkpoints"

    log_dir = exp_root / "logs"

    input_dir = (
        project_root
        / "mental_layer_test"
    )

    # ==================================================
    # 模式配置
    # ==================================================

    MODE = "batch"

    MODEL_NAME = "best_model.pt"

    MODEL_PATTERN = "checkpoint_epoch_*.pt"

    # ==================================================
    # config
    # ==================================================

    config_files = sorted(
        log_dir.glob("config.yaml")
    )

    if not config_files:
        raise FileNotFoundError(
            f"No config found in {log_dir}"
        )

    cfg_path = config_files[-1]

    cfg = load_config(cfg_path)

    print(f"[Init] Using config: {cfg_path}")
    
    # ==================================================
    # infer parameters
    # ==================================================

    base_logits_scale = cfg.get(
        "base_logits_scale",
        1.5
    )

    delta_scale = cfg.get(
        "delta_scale",
        0.5
    )

    sigmoid_scale = cfg.get(
        "sigmoid_scale",
        1.0
    )

    use_gaussian_blur = cfg.get(
        "use_gaussian_blur",
        False
    )

    print(
        f"[Infer Params] "
        f"base={base_logits_scale}, "
        f"delta={delta_scale}, "
        f"sigmoid={sigmoid_scale}, "
        f"blur={use_gaussian_blur}"
    )

    # ==================================================
    # 模型选择
    # ==================================================

    if MODE == "batch":

        model_paths = sorted(
            checkpoint_dir.glob(MODEL_PATTERN)
        )

        if not model_paths:
            raise FileNotFoundError(
                f"No models match {MODEL_PATTERN}"
            )

    elif MODE == "single":

        model_paths = [
            checkpoint_dir / MODEL_NAME
        ]

    else:
        raise ValueError(
            "MODE must be 'single' or 'batch'"
        )

    print(
        f"[Mode] {MODE}, total models: {len(model_paths)}"
    )

    # ==================================================
    # 输入图像
    # ==================================================

    img_paths = sorted(
        input_dir.glob("*.png")
    )

    print(f"[Init] Images: {len(img_paths)}")

    # ==================================================
    # 推理
    # ==================================================

    for model_path in model_paths:

        print(f"\n[Model] {model_path.name}")

        model = load_model(
            model_path,
            cfg["model"],
            device
        )

        # ----------------------------------------------
        # 输出目录
        # experiments_ILT/exp_name/checkpoint_epoch_xxx/infer/
        # ----------------------------------------------

        checkpoint_output_dir = (
            exp_root
            / model_path.stem
            / "infer"
        )

        checkpoint_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------
        # infer
        # ----------------------------------------------

        for i, path in enumerate(img_paths):

            x = preprocess(path).to(device)

            with torch.no_grad():

                # ==========================================
                # delta prediction
                # ==========================================

                delta = model(x)

                delta = torch.tanh(delta)

                # ==========================================
                # base logits
                # 与 train.py 保持一致
                # ==========================================

                base_logits = x * 2.0 - 1.0

                # ==========================================
                # final logits
                # ==========================================

                mask_logits = base_logits * base_logits_scale + delta_scale * delta

                mask_prob = torch.sigmoid(sigmoid_scale * mask_logits)
                
                mask_prob = gaussian_blur(mask_prob)

            result = postprocess(mask_prob)

            save_name = (
                f"mental_layer_{i:05d}.png"
            )

            save_path = (
                checkpoint_output_dir
                / save_name
            )

            cv2.imwrite(
                str(save_path),
                result
            )

            print(
                f"[{i+1}/{len(img_paths)}] {save_name}"
            )

    print("\nInference done!")


if __name__ == "__main__":
    main()