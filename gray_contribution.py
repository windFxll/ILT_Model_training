import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ==================================================
# 根目录
# ==================================================

ROOT_DIR = Path(
    r"\content\drive\MyDrive\Colab Notebooks\experiments_ILT\exp_ilt_unet_smooth_mse_0.4base_0.6delta_copy"
)

# ==================================================
# 找所有 checkpoint_epoch_xxx
# ==================================================

checkpoint_dirs = sorted(
    [
        p for p in ROOT_DIR.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint_epoch_")
    ]
)

print(f"Found {len(checkpoint_dirs)} checkpoints")


# ==================================================
# 保存所有epoch统计
# ==================================================

all_results = {}

# ==================================================
# 遍历checkpoint
# ==================================================

for checkpoint_dir in checkpoint_dirs:

    infer_dir = checkpoint_dir / "infer"

    if not infer_dir.exists():
        print(f"Skip: {infer_dir}")
        continue

    print(f"\nProcessing: {checkpoint_dir.name}")

    # 统计0~255灰度数量
    gray_count = np.zeros(256, dtype=np.int64)

    img_paths = sorted(
        infer_dir.glob("mental_layer_*.png")
    )

    print(f"Images: {len(img_paths)}")

    # ----------------------------------------------
    # 遍历图片
    # ----------------------------------------------

    for img_path in img_paths:

        img = cv2.imread(str(img_path), 0)

        if img is None:
            print(f"Failed: {img_path}")
            continue

        # 统计灰度直方图
        hist = np.bincount(
            img.flatten(),
            minlength=256
        )

        gray_count += hist

    all_results[checkpoint_dir.name] = gray_count


# ==================================================
# 保存CSV
# ==================================================

csv_dict = {
    "gray_value": np.arange(256)
}

for epoch_name, counts in all_results.items():
    csv_dict[epoch_name] = counts

df = pd.DataFrame(csv_dict)

csv_path = ROOT_DIR / "gray_distribution.csv"

df.to_csv(csv_path, index=False)

print(f"\nCSV saved to:")
print(csv_path)


# ==================================================
# 绘图
# ==================================================

plt.figure(figsize=(12, 8))

for epoch_name, counts in all_results.items():

    plt.plot(
        np.arange(256),
        counts,
        label=epoch_name
    )

plt.xlabel("Gray Value")
plt.ylabel("Pixel Count")

plt.title("Gray Distribution Across Epochs")

plt.legend(fontsize=8)

plt.grid(True)

plot_path = ROOT_DIR / "gray_distribution_plot.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nPlot saved to:")
print(plot_path)