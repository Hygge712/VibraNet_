import os
import copy

import cv2
import torch
import numpy as np

from src.loftr import LoFTR, default_cfg


# ============================================================
# Change only these paths
# ============================================================
IMAGE0_PATH = r"img1"
IMAGE1_PATH = r"img2"
WEIGHT_PATH = r"Your weight path"
OUTPUT_PATH = r"result.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(img_bgr, device):
    h, w = img_bgr.shape[:2]
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8
    img = img_bgr[:h8, :w8]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(gray)[None, None].float().div(255.0).to(device)


def load_model():
    cfg = copy.deepcopy(default_cfg)
    print("Using project default_cfg")
    print("backbone_type:", cfg.get("backbone_type", "N/A"))
    print("resolution:", cfg.get("resolution", "N/A"))
    print("match_coarse:", cfg.get("match_coarse", "N/A"))

    matcher = LoFTR(config=cfg)
    checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    matcher.load_state_dict(state_dict, strict=True)
    matcher.eval().to(DEVICE)
    return matcher


def main():
    for path in [IMAGE0_PATH, IMAGE1_PATH, WEIGHT_PATH]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    img0 = cv2.imread(IMAGE0_PATH)
    img1 = cv2.imread(IMAGE1_PATH)
    if img0 is None:
        raise RuntimeError(f"Cannot read: {IMAGE0_PATH}")
    if img1 is None:
        raise RuntimeError(f"Cannot read: {IMAGE1_PATH}")

    print("Image0:", img0.shape)
    print("Image1:", img1.shape)
    if img0.shape[:2] != img1.shape[:2]:
        raise ValueError("The two demo images must have the same HxW.")

    tensor0 = preprocess(img0, DEVICE)
    tensor1 = preprocess(img1, DEVICE)
    print("Tensor0:", tuple(tensor0.shape))
    print("Tensor1:", tuple(tensor1.shape))

    matcher = load_model()
    batch = {"image0": tensor0, "image1": tensor1}

    with torch.no_grad():
        matcher(batch)

    if "mkpts0_f" not in batch or "mkpts1_f" not in batch:
        raise RuntimeError("LoFTR did not return mkpts0_f/mkpts1_f.")

    mkpts0 = batch["mkpts0_f"].detach().cpu().numpy()
    mkpts1 = batch["mkpts1_f"].detach().cpu().numpy()

    print("\n========== FINAL MATCH ==========")
    print("Total matches:", len(mkpts0))

    if "mconf" in batch:
        conf = batch["mconf"].detach().cpu().numpy().reshape(-1)
        if len(conf):
            print("Confidence min :", float(conf.min()))
            print("Confidence max :", float(conf.max()))
            print("Confidence mean:", float(conf.mean()))

    if len(mkpts0) == 0:
        print("NO MATCHES.")
        print("This test fails before ROI/displacement calculation.")
        return

    print("First 10 matches:")
    for i in range(min(10, len(mkpts0))):
        print(
            f"{i:03d}: "
            f"({mkpts0[i,0]:.2f}, {mkpts0[i,1]:.2f}) -> "
            f"({mkpts1[i,0]:.2f}, {mkpts1[i,1]:.2f})"
        )

    # Visualization: image0 | image1 with match lines.
    rgb0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    H = max(rgb0.shape[0], rgb1.shape[0])
    W0 = rgb0.shape[1]
    W1 = rgb1.shape[1]
    canvas = np.zeros((H, W0 + W1, 3), dtype=np.uint8)
    canvas[:rgb0.shape[0], :W0] = rgb0
    canvas[:rgb1.shape[0], W0:W0 + W1] = rgb1

    nvis = min(len(mkpts0), 100)
    idx = np.linspace(0, len(mkpts0) - 1, nvis).astype(int)

    for k in idx:
        x0, y0 = mkpts0[k]
        x1, y1 = mkpts1[k]
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1 + W0)), int(round(y1)))
        cv2.circle(canvas, p0, 3, (0, 255, 0), -1)
        cv2.circle(canvas, p1, 3, (0, 255, 0), -1)
        cv2.line(canvas, p0, p1, (255, 0, 0), 1)

    cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    mean_dx = float(np.mean(mkpts1[:, 0] - mkpts0[:, 0]))
    mean_dy = float(np.mean(mkpts1[:, 1] - mkpts0[:, 1]))
    print(f"Mean dx = {mean_dx:.6f} px")
    print(f"Mean dy = {mean_dy:.6f} px")
    print("Visualization:", OUTPUT_PATH)
    print("MATCH TEST PASSED.")


if __name__ == "__main__":
    main()
