# experiments/exp10_imagenet_adv_pier.py
#
# Exp 10: Adversarial-dose PIER audit for an ImageNet ecosystem.
#
# Ecosystem (example choice, at least 5 models total):
#   - Standard ResNet-18
#   - Standard ResNet-50
#   - EfficientNet-B0
#   - ConvNeXt-Tiny
#   - ViT-B/16
#   - (Optional) Adversarially robust ResNet-50 (user-specified checkpoint)
#
# We treat each model as a potential target and the rest as peers.
# For each target, we:
#   1) Split a subset of ImageNet validation samples into:
#        - P_fit: used to learn a single global convex weight vector w_hat
#        - P_eval: used to evaluate PIER as a function of adversarial dose
#   2) Define an FGSM-style adversarial intervention at a fixed dose epsilon:
#        x_adv = x + epsilon * sign(grad_x L(ref_model(x), y))
#      where ref_model is a fixed standard model (ResNet-50).
#      The same perturbed input is fed to all models in the ecosystem.
#   3) On P_fit and a set of low doses, learn w_hat via DISCOSolver:
#        Y_t = target outputs (p_correct) on (x_adv, epsilon) pairs
#        Y_P = peer outputs
#   4) On P_eval and a grid of doses, compute PIER using the fixed w_hat:
#        PIER_t(epsilon) = E_{x in P_eval} |Y_t - sum_j w_hat_j * Y_j|
#
# Intuition:
#   - At low adversarial dose, a robust model should behave similarly to
#     standard models; PIER should be small.
#   - At higher doses, standard models collapse while the robust model
#     maintains confident predictions on the true label. The robust
#     model's outputs should then move outside the convex hull of its
#     standard peers, leading to high PIER.
#   - In contrast, standard models may remain relatively redundant with
#     respect to each other across the dose range.

import sys
import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as tv_models

# Ensure we can import the local `isqed` package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isqed.geometry import DISCOSolver
from isqed.real_world import ImageModelWrapperLogitMargin, AdversarialFGSMIntervention
from torch.hub import download_url_to_file


# ---------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------
# Add experiments/ for utils if needed later
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_dir)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DOSES_FIT = np.linspace(0.0, 0.1, 6)
DOSES_EVAL = np.linspace(0.0, 0.1, 6)


# ---------------------------------------------------------------------
# 2. Checkpoint Configuration
# ---------------------------------------------------------------------
# REPLACEMENT: Using MadryLab's standard robust ResNet-50 (ImageNet)
# Paper: "Robustness (Lib)" / Engstrom et al. (2019)
# Source: https://huggingface.co/madrylab/robust-imagenet-models
# This is a direct download link that works without authentication.
ROBUST_R50_URL = (
    "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/"
    "resnet50_l2_eps3.ckpt"
)
CKPT_FILENAME = "resnet50_l2_eps3.ckpt"

def get_robust_checkpoint_path(download_dir="./checkpoints") -> Optional[str]:
    """
    Checks if the robust ResNet-50 checkpoint exists locally.
    If not, downloads it automatically from MadryLab's Hugging Face repo.
    """
    os.makedirs(download_dir, exist_ok=True)
    ckpt_path = os.path.join(download_dir, CKPT_FILENAME)

    if not os.path.exists(ckpt_path):
        print(f"\n[INFO] Robust checkpoint not found at {ckpt_path}")
        print(f"[INFO] Downloading robust ResNet-50 (L2 eps=3.0) from Hugging Face...")
        print(f"       Source: {ROBUST_R50_URL}")
        try:
            # torch.hub downloads with a progress bar
            download_url_to_file(ROBUST_R50_URL, ckpt_path)
            print("[INFO] Download completed.")
        except Exception as e:
            print(f"[ERROR] Failed to download robust checkpoint: {e}")
            print("        Please check your internet connection or try downloading manually.")
            return None
    else:
        print(f"  [INFO] Found robust checkpoint: {ckpt_path}")

    return ckpt_path

# ---------------------------------------------------------------------
# 3. Data preparation
# ---------------------------------------------------------------------
def build_imagenet_subset(
    data_root: str,
    max_samples: int,
    rng: np.random.RandomState,
) -> List[Tuple[torch.Tensor, int]]:
    """
    Build a small subset of ImageNet-style images using torchvision.datasets.ImageFolder.

    We apply a common ImageNet preprocessing transform so that all models
    can share the same representation:
      - Resize(256) + CenterCrop(224)
      - ToTensor
      - Normalize(IMAGENET_MEAN, IMAGENET_STD)

    Returns a list of (x, y) pairs, where x is a normalized tensor.
    """
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    n_total = len(dataset)
    print(f"Found {n_total} images under {data_root}.")

    if max_samples is not None and max_samples < n_total:
        indices = rng.choice(n_total, size=max_samples, replace=False)
    else:
        indices = np.arange(n_total)

    samples: List[Tuple[torch.Tensor, int]] = []
    for idx in indices:
        x, y = dataset[idx]
        samples.append((x, y))

    print(f"Using {len(samples)} samples for Exp 10.")
    return samples


def split_fit_eval(
    samples: List[Tuple[torch.Tensor, int]],
    rng: np.random.RandomState,
    fit_frac: float = 0.5,
) -> Tuple[List[Tuple[torch.Tensor, int]], List[Tuple[torch.Tensor, int]]]:
    """
    Split samples deterministically into fit/eval sets.
    """
    n = len(samples)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_fit = int(n * fit_frac)
    fit_idx = idx[:n_fit]
    eval_idx = idx[n_fit:]

    fit_samples = [samples[i] for i in fit_idx]
    eval_samples = [samples[i] for i in eval_idx]

    print(f"Split into fit={len(fit_samples)}, eval={len(eval_samples)}.")
    return fit_samples, eval_samples


# ---------------------------------------------------------------------
# 4. Model loaders
# ---------------------------------------------------------------------
def load_standard_models(device: str) -> List[ImageModelWrapperLogitMargin]:
    """
    Load a set of standard ImageNet models from torchvision.

    We keep at least 5 models in total. One of them (ResNet-50) will also
    serve as the adversarial reference model.
    """
    models: List[ImageModelWrapperLogitMargin] = []

    # ResNet-18
    try:
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        resnet18 = tv_models.resnet18(weights=weights)
        models.append(ImageModelWrapperLogitMargin(resnet18, "ResNet18", device))
        print("  [OK] Loaded ResNet18.")
    except Exception as e:
        print(f"  [FAIL] ResNet18: {e}")

    # ResNet-50 (used later as reference model)
    try:
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
        resnet50 = tv_models.resnet50(weights=weights)
        models.append(ImageModelWrapperLogitMargin(resnet50, "ResNet50", device))
        print("  [OK] Loaded ResNet50.")
    except Exception as e:
        print(f"  [FAIL] ResNet50: {e}")

    # EfficientNet-B0
    try:
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
        effnet_b0 = tv_models.efficientnet_b0(weights=weights)
        models.append(ImageModelWrapperLogitMargin(effnet_b0, "EfficientNetB0", device))
        print("  [OK] Loaded EfficientNetB0.")
    except Exception as e:
        print(f"  [FAIL] EfficientNetB0: {e}")

    # ConvNeXt-Tiny
    try:
        weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        convnext_t = tv_models.convnext_tiny(weights=weights)
        models.append(ImageModelWrapperLogitMargin(convnext_t, "ConvNeXtTiny", device))
        print("  [OK] Loaded ConvNeXtTiny.")
    except Exception as e:
        print(f"  [FAIL] ConvNeXtTiny: {e}")

    # ViT-B/16
    try:
        weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1
        vit_b16 = tv_models.vit_b_16(weights=weights)
        models.append(ImageModelWrapperLogitMargin(vit_b16, "ViT_B16", device))
        print("  [OK] Loaded ViT_B16.")
    except Exception as e:
        print(f"  [FAIL] ViT_B16: {e}")

    return models


def load_robust_resnet50(
    device: str,
    robust_ckpt: Optional[str],
) -> Optional[ImageModelWrapperLogitMargin]:
    """
    Optionally load an adversarially trained ResNet-50.

    We assume:
      - The architecture is torchvision.models.resnet50.
      - `robust_ckpt` is a path to a state_dict (or checkpoint with a
        'state_dict' field).

    If loading fails or robust_ckpt is None, we return None.
    """
    if robust_ckpt is None:
        print("  [INFO] No robust_ckpt provided; skipping robust ResNet-50.")
        return None

    try:
        base_model = tv_models.resnet50(weights=None)
        ckpt = torch.load(robust_ckpt, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        # Strip any 'module.' prefixes if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v
        base_model.load_state_dict(new_state_dict, strict=False)
        print("  [OK] Loaded robust ResNet-50 from checkpoint.")
        return ImageModelWrapperLogitMargin(base_model, "RobustResNet50", device)
    except Exception as e:
        print(f"  [FAIL] Robust ResNet-50: {e}")
        return None


# ---------------------------------------------------------------------
# 5. Main experiment logic
# ---------------------------------------------------------------------
def run_imagenet_adv_pier_experiment(
    data_root: str,
    max_samples: int = 500,
    robust_ckpt: Optional[str] = None,
    output_filename: str = "exp10_imagenet_adv_pier.csv",
):
    print("--- Running Exp 10: Adversarial-dose PIER on ImageNet ecosystem ---")

    # Seeds for reproducibility
    rng = np.random.RandomState(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Data
    samples = build_imagenet_subset(data_root, max_samples=max_samples, rng=rng)
    fit_samples, eval_samples = split_fit_eval(samples, rng=rng, fit_frac=0.5)

    # 2) Models
    models = load_standard_models(device)

    if robust_ckpt == "AUTO":
        robust_ckpt = get_robust_checkpoint_path(
            download_dir=os.path.join(this_dir, "checkpoints")
        )

    robust_wrapper = load_robust_resnet50(device, robust_ckpt=robust_ckpt)
    if robust_wrapper is not None:
        models.append(robust_wrapper)

    if len(models) < 5:
        print(f"Only {len(models)} models loaded (<5). Aborting.")
        return

    # Identify a reference model for FGSM (standard ResNet-50 if available)
    ref_model = None
    for m in models:
        if "ResNet50" in m.name and "Robust" not in m.name:
            ref_model = m.model
            break
    if ref_model is None:
        ref_model = models[0].model
        print("  [WARN] Using first loaded model as adversarial reference.")

    adv_intervention = AdversarialFGSMIntervention(ref_model=ref_model, device=device)

    # 3) Dose design (in normalized units; user may tune these)
    doses_fit = DOSES_FIT
    doses_eval = DOSES_EVAL

    print(f"Fit doses:  {doses_fit}")
    print(f"Eval doses: {doses_eval}")

    # For grouping in plots
    def model_group(name: str) -> str:
        if "Robust" in name:
            return "Robust CNN"
        elif "ViT" in name:
            return "Transformer"
        else:
            return "Standard CNN"

    # 4) Main loop over targets
    results = []

    for t_idx, target in enumerate(models):
        target_name = target.name
        group_name = model_group(target_name)
        print(f"\n>>> Auditing target: {target_name}  (group: {group_name})")

        # Peers are all other models
        peers = [m for j, m in enumerate(models) if j != t_idx]
        n_peers = len(peers)

        # -----------------------------
        # 4.1 Fit phase on P_fit
        # -----------------------------
        print("  [Phase] Fitting global convex baseline on P_fit...")

        y_t_fit_list = []
        Y_p_fit_list = []

        for (x, y) in tqdm(fit_samples, desc="    P_fit samples", leave=False):
            sample = (x, y)
            for eps in doses_fit:
                adv_sample = adv_intervention.apply(sample, epsilon=float(eps))

                y_t = target._forward(adv_sample)
                y_ps = [p._forward(adv_sample) for p in peers]

                y_t_fit_list.append(y_t)
                Y_p_fit_list.append(y_ps)

        y_t_fit_vec = np.array(y_t_fit_list, dtype=float)
        if y_t_fit_vec.ndim == 1:
            y_t_fit_vec = y_t_fit_vec.reshape(-1, 1)

        Y_p_fit_mat = np.array(Y_p_fit_list, dtype=float)

        try:
            _, w_hat = DISCOSolver.solve_weights_and_distance(
                y_t_fit_vec,
                Y_p_fit_mat,
            )
        except Exception as e:
            print(f"  [ERROR] DISCOSolver failed for {target_name}: {e}")
            continue

        w_hat = np.asarray(w_hat, dtype=float).flatten()
        if w_hat.shape[0] != n_peers:
            print(
                f"  [WARN] w_hat length {w_hat.shape[0]} != num_peers {n_peers} "
                f"for {target_name}"
            )

        print(f"  [Fit] Learned w_hat (first 3 entries): {w_hat[:3]}")

        # -----------------------------
        # 4.2 Eval phase on P_eval
        # -----------------------------
        print("  [Phase] Evaluating PIER on P_eval...")

        for eps in doses_eval:
            eps = float(eps)
            residuals = []

            for (x, y) in eval_samples:
                sample = (x, y)
                adv_sample = adv_intervention.apply(sample, epsilon=eps)

                y_t = target._forward(adv_sample)
                y_ps = np.array([p._forward(adv_sample) for p in peers], dtype=float)

                y_mix = float(np.dot(w_hat, y_ps))
                residuals.append(abs(y_t - y_mix))

            if residuals:
                mean_pier = float(np.mean(residuals))
            else:
                mean_pier = float("nan")

            results.append(
                {
                    "TargetModel": target_name,
                    "Group": group_name,
                    "Dose": eps,
                    "MeanPIER": mean_pier,
                }
            )

    if not results:
        print("No results collected; aborting.")
        return

    df = pd.DataFrame(results)
    out_dir = os.path.join("results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    df.to_csv(out_path, index=False)
    print(f"\nSaved adversarial-dose PIER results to: {out_path}")
    print("Done.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 10: Adversarial-dose PIER on ImageNet ecosystem."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        # required=True,
        default="/work3/leiyo/imagenet",
        help="Path to ImageNet-style folder (ImageFolder) with images.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum number of images to use.",
    )
    parser.add_argument(
        "--robust_ckpt",
        type=str,
        default="AUTO",
        help="Path to robust ResNet-50 checkpoint, or 'AUTO' to download automatically.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="exp10_imagenet_adv_pier.csv",
        help="Output CSV filename under results/tables/.",
    )

    args = parser.parse_args()

    run_imagenet_adv_pier_experiment(
        data_root=args.data_root,
        max_samples=args.max_samples,
        robust_ckpt=args.robust_ckpt,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
