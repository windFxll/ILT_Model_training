import argparse
from pathlib import Path
import cv2
import torch
import numpy as np

# =========================
# 直接调用固定模型
# =========================
from models.unet_edge_v2 import UNetEdge_v2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="experiment name in experiments_ILT"
    )

    return parser.parse_args()


# ==================================================
# 加载 Forward Model
# ==================================================
def load_model(model_path, device):

    model = UNetEdge_v2()

    ckpt = torch.load(model_path, map_location=device)

    state_dict = ckpt.get("model_state_dict", ckpt)

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model


# ==================================================
# 图像预处理
# ==================================================
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
# 后处理
# ==================================================
def postprocess(pred):

    pred = torch.sigmoid(pred)

    pred = pred.squeeze().cpu().numpy()

    pred = (pred > 0.5).astype(np.uint8) * 255

    return pred


# ==================================================
# 主函数
# ==================================================
def main():

    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ==================================================
    # 路径
    # ==================================================
    DRIVE_ROOT = Path(
        "/content/drive/MyDrive/Colab Notebooks"
    )

    project_root = Path(__file__).resolve().parent

    exp_root = (
        DRIVE_ROOT
        / "experiments_ILT"
        / args.exp_name
    )

    # ==================================================
    # Forward Model
    # ==================================================
    model_path = (
        project_root
        / "Forward_Model.pt"
    )

    print(f"[Load Model] {model_path}")

    model = load_model(model_path, device)

    # ==================================================
    # 查找 checkpoint_epoch_0XX
    # ==================================================
    checkpoint_dirs = sorted(
        exp_root.glob("checkpoint_epoch_*")
    )

    print(f"[Checkpoint Folders] {len(checkpoint_dirs)}")

    # ==================================================
    # 遍历每个 checkpoint
    # ==================================================
    for checkpoint_dir in checkpoint_dirs:

        print(f"\n==========")
        print(f"[Checkpoint] {checkpoint_dir.name}")

        infer_dir = checkpoint_dir / "infer"

        if not infer_dir.exists():

            print(f"[Skip] infer folder not found")
            continue

        # ==================================================
        # 输出目录
        # ==================================================
        output_dir = checkpoint_dir / "forward_sim"

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ==================================================
        # 图像
        # ==================================================
        img_paths = sorted(
            infer_dir.glob("mental_layer_*.png")
        )

        print(f"[Images] {len(img_paths)}")

        # ==================================================
        # Forward Simulation
        # ==================================================
        for i, img_path in enumerate(img_paths):

            x = preprocess(img_path).to(device)

            with torch.no_grad():

                pred = model(x)

            result = postprocess(pred)

            save_path = output_dir / img_path.name

            cv2.imwrite(
                str(save_path),
                result
            )

            print(
                f"[{i+1}/{len(img_paths)}] "
                f"{img_path.name}"
            )

    print("\nForward Simulation Done!")


if __name__ == "__main__":
    main()