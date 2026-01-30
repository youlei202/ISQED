# experiments/exp11_shape_bias_texture_context.py
#
# Exp 11: Shape-biased CNN vs standard CNNs under texture vs shape contexts.
#
# Contexts:
#   - "natural": standard ImageNet-like validation images
#   - "shape_bias": cue-conflict / stylized images that encourage shape-based decisions
#
# For each context and each target model, we:
#   1) Sample up to `max_samples_per_context` images.
#   2) Split them into P_fit and P_eval (50/50).
#   3) On P_fit (theta = 0, identity intervention), learn a single global convex
#      baseline w_hat over the peer models using DISCOSolver.
#   4) On P_eval, compute PIER = |Y_t - sum_j w_hat[j] * Y_j| and average it.
#
# We use logits-based scalar responses g(f_j(x)) defined inside ImageModelWrapper:
#   g = (logit_true_class) - max_other_logit
#
# Expected qualitative pattern:
#   - On the natural context, shape-biased variants are not dramatically more
#     unique than other high-capacity models (depending on training).
#   - On the shape-biased cue-conflict context, shape-biased ResNet-50 variants
#     become harder to approximate by convex combinations of peers,
#     i.e. have higher PIER than standard CNNs / transformers.

import os
import sys
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# Make local package importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from isqed.ecosystem import Ecosystem
from isqed.geometry import DISCOSolver
from isqed.real_world import ImageModelWrapper, ImageIdentityIntervention

# Import deterministic seed helper
sys.path.append(os.path.join(ROOT_DIR, "experiments"))
from experiments.utils import make_stable_seed

SCALAR_MODE = "p_true"


# ============================================================
# 1. Model loading utilities
# ============================================================

def load_standard_models(device: str) -> Dict[str, ImageModelWrapper]:
    """
    Load a small ImageNet ecosystem of standard/high-capacity models.

    All models are wrapped by ImageModelWrapper so that they expose `_forward`.
    """
    model_wrappers: Dict[str, ImageModelWrapper] = {}

    # Standard ResNet50
    res50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model_wrappers["ResNet50"] = ImageModelWrapper(res50, "ResNet50", device, mode=SCALAR_MODE)

    # EfficientNet-B0
    eff_b0 = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model_wrappers["EfficientNetB0"] = ImageModelWrapper(
        eff_b0, "EfficientNetB0", device, mode=SCALAR_MODE
    )

    # ConvNeXt-Tiny
    convnext_tiny = models.convnext_tiny(
        weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    )
    model_wrappers["ConvNeXtTiny"] = ImageModelWrapper(
        convnext_tiny, "ConvNeXtTiny", device, mode=SCALAR_MODE
    )

    # ViT-B/16
    vit_b16 = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model_wrappers["ViT_B16"] = ImageModelWrapper(vit_b16, "ViT_B16", device, mode=SCALAR_MODE)

    return model_wrappers


def load_shape_biased_resnet50(
    device: str,
    variant: str = "C",
) -> ImageModelWrapper:
    """
    Load one of the shape-related ResNet-50 variants from Geirhos et al.:

      A: resnet50_trained_on_SIN
      B: resnet50_trained_on_SIN_and_IN
      C: resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN
    """
    from collections import OrderedDict
    from torch.hub import load_state_dict_from_url

    model_urls = {
        "A": (
            "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/"
            "raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/"
            "resnet50_train_60_epochs-c8e5653e.pth.tar"
        ),
        "B": (
            "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/"
            "raw/60b770e128fffcbd8562a3ab3546c1a735432d03/"
            "resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar"
        ),
        "C": (
            "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/"
            "raw/60b770e128fffcbd8562a3ab3546c1a735432d03/"
            "resnet50_finetune_60_epochs_lr_decay_after_30_"
            "start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar"
        ),
    }

    if variant not in model_urls:
        raise ValueError(f"Unknown shape-biased variant '{variant}', use 'A', 'B' or 'C'.")

    backbone = models.resnet50(weights=None)
    checkpoint = load_state_dict_from_url(model_urls[variant], map_location=device)
    state_dict = checkpoint["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        new_state_dict[new_k] = v

    backbone.load_state_dict(new_state_dict)
    backbone.to(device).eval()

    name = {
        "A": "ShapeResNet50_SIN",
        "B": "ShapeResNet50_SININ",
        "C": "ShapeResNet50_ShapeResNet",
    }[variant]
    return ImageModelWrapper(backbone, name, device, mode=SCALAR_MODE)


# ============================================================
# 2. Dataset utilities
# ============================================================

def build_imagenet_transform() -> transforms.Compose:
    """
    Standard ImageNet validation transform (resize-crop-to-tensor + normalize).
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_context_samples(
    root: str,
    max_samples: int,
    np_rng: np.random.RandomState,
    context_name: str,
) -> Tuple[List[Tuple[torch.Tensor, int]], List[str]]:
    """
    Load up to `max_samples` images from an ImageFolder root.

    Returns:
        samples: list of (x_tensor, y_int)
        ids:     list of string identifiers (e.g. relative file paths)
    """
    transform = build_imagenet_transform()
    dataset = datasets.ImageFolder(root=root, transform=transform)

    n_total = len(dataset)
    if n_total == 0:
        raise RuntimeError(f"Context '{context_name}' has no images under {root}.")

    if max_samples is not None and max_samples < n_total:
        indices = np_rng.choice(n_total, size=max_samples, replace=False)
    else:
        indices = np.arange(n_total)

    samples: List[Tuple[torch.Tensor, int]] = []
    ids: List[str] = []

    print(
        f"[Context={context_name}] Loading {len(indices)} samples "
        f"from {root} (n_total={n_total})."
    )

    for idx in tqdm(indices, desc=f"Loading {context_name} images"):
        x, y = dataset[idx]
        path, _ = dataset.samples[idx]
        samples.append((x, int(y)))
        ids.append(os.path.relpath(path, root))

    return samples, ids


def split_fit_eval_indices(
    n: int,
    np_rng: np.random.RandomState,
    fit_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly split indices [0, ..., n-1] into fit and eval sets.
    """
    idx = np.arange(n)
    np_rng.shuffle(idx)
    n_fit = int(n * fit_fraction)
    fit_idx = idx[:n_fit]
    eval_idx = idx[n_fit:]
    return fit_idx, eval_idx


# ============================================================
# 3. Main experiment
# ============================================================

def run_shape_texture_experiment(
    natural_root: str,
    shape_root: str,
    max_samples_per_context: int = 800,
    fit_fraction: float = 0.5,
    output_filename: str = "exp12_shape_texture_pier_separated.csv",
):
    print("=== Exp 11: Shape-biased vs standard CNNs (texture vs shape contexts) ===")

    # Fixed seeds for reproducibility
    np_rng = np.random.RandomState(0)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Load standard models once
    print("\nLoading standard ImageNet models...")
    models_std = load_standard_models(device=device)

    # 2) Pre-load context datasets and fixed splits (shared across variants)
    contexts = [
        ("texture_natural", natural_root),
        ("shape_bias", shape_root),
    ]

    contexts_cache = {}
    for context_name, root in contexts:
        print(f"\n=== Pre-loading context: {context_name} ===")
        samples_ctx, ids_ctx = load_context_samples(
            root=root,
            max_samples=max_samples_per_context,
            np_rng=np_rng,
            context_name=context_name,
        )

        n_ctx = len(samples_ctx)
        if n_ctx < 10:
            print(f"  [WARN] Very small context ({n_ctx} samples). Skipping this context.")
            continue

        fit_idx, eval_idx = split_fit_eval_indices(n_ctx, np_rng, fit_fraction)
        fit_samples = [samples_ctx[i] for i in fit_idx]
        fit_ids = [ids_ctx[i] for i in fit_idx]

        eval_samples = [samples_ctx[i] for i in eval_idx]
        eval_ids = [ids_ctx[i] for i in eval_idx]

        print(
            f"  -> Context '{context_name}': using {len(fit_samples)} for P_fit, "
            f"{len(eval_samples)} for P_eval (raw size={n_ctx})."
        )

        contexts_cache[context_name] = {
            "fit_samples": fit_samples,
            "fit_ids": fit_ids,
            "eval_samples": eval_samples,
            "eval_ids": eval_ids,
            "n_ctx": n_ctx,
        }

    intervention = ImageIdentityIntervention()
    DOSES = [0.0]  # single theta = 0 (no intervention)

    rows = []

    # 3) Loop over shape-biased variants: A, B, C
    shape_variants = ["A", "B", "C"]

    for variant in shape_variants:
        print(f"\n==============================")
        print(f"=== Shape variant: {variant} ===")
        print(f"==============================")

        # Load the corresponding shape-biased ResNet-50
        model_shape = load_shape_biased_resnet50(device=device, variant=variant)

        # Build ecosystem: standard models + this shape-biased variant
        all_models: Dict[str, ImageModelWrapper] = {}
        all_models.update(models_std)
        all_models[model_shape.name] = model_shape

        model_names = list(all_models.keys())
        print(f"Ecosystem models (variant {variant}): {model_names}")

        # For each context, run DISCO-style uniqueness audit
        for (context_name, _root) in contexts:
            if context_name not in contexts_cache:
                print(f"[WARN] Context '{context_name}' missing in cache, skipping.")
                continue

            cache = contexts_cache[context_name]
            fit_samples = cache["fit_samples"]
            fit_ids = cache["fit_ids"]
            eval_samples = cache["eval_samples"]
            eval_ids = cache["eval_ids"]
            n_ctx = cache["n_ctx"]

            print(f"\n--- Variant {variant}, Context: {context_name} ---")

            for target_name in model_names:
                print(f"\n  >> Target: {target_name} (variant {variant}, context {context_name})")

                target_model = all_models[target_name]
                peers = [m for name, m in all_models.items() if name != target_name]

                eco = Ecosystem(target=target_model, peers=peers)

                # -------------------------------
                # 3.1 FIT phase (learn w_hat)
                # -------------------------------
                fit_X = []
                fit_Theta = []
                fit_seeds = []

                for (sample, sample_id) in zip(fit_samples, fit_ids):
                    for theta in DOSES:
                        seed = make_stable_seed(
                            text=f"{context_name}|fit|{sample_id}|variant={variant}",
                            theta=float(theta),
                            context_type="dataset",
                            ctx_label=context_name,
                        )
                        fit_X.append(sample)
                        fit_Theta.append(float(theta))
                        fit_seeds.append(int(seed))

                y_t_fit, Y_p_fit = eco.batched_query(
                    X=fit_X,
                    Thetas=fit_Theta,
                    intervention=intervention,
                    seeds=fit_seeds,
                )

                y_t_fit_vec = y_t_fit.reshape(-1, 1)
                Y_p_fit_mat = Y_p_fit

                _, w_hat = DISCOSolver.solve_weights_and_distance(
                    y_t_fit_vec,
                    Y_p_fit_mat,
                )
                w_hat = np.asarray(w_hat, dtype=float).flatten()

                print(f"    [Fit] Learned convex weights (first 3): {w_hat[:3]}")

                # -------------------------------
                # 3.2 EVAL phase (compute PIER)
                # -------------------------------
                eval_X = []
                eval_Theta = []
                eval_seeds = []

                for (sample, sample_id) in zip(eval_samples, eval_ids):
                    for theta in DOSES:
                        seed = make_stable_seed(
                            text=f"{context_name}|eval|{sample_id}|variant={variant}",
                            theta=float(theta),
                            context_type="dataset",
                            ctx_label=context_name,
                        )
                        eval_X.append(sample)
                        eval_Theta.append(float(theta))
                        eval_seeds.append(int(seed))

                y_t_eval, Y_p_eval = eco.batched_query(
                    X=eval_X,
                    Thetas=eval_Theta,
                    intervention=intervention,
                    seeds=eval_seeds,
                )

                eval_Theta_arr = np.asarray(eval_Theta, dtype=float)
                y_mix_eval = Y_p_eval @ w_hat
                residuals_all = np.abs(y_t_eval - y_mix_eval)
                res_dir = os.path.join(ROOT_DIR, "results", "artifacts", "exp12")
                os.makedirs(res_dir, exist_ok=True)
                res_file = f"exp12_{context_name}_{target_name}_residuals.npz"
                res_path = os.path.join(res_dir, res_file)

                np.savez_compressed(
                    res_path,
                    residuals=residuals_all,
                    thetas=np.asarray(eval_Theta, dtype=float),
                )

                for theta in DOSES:
                    mask = np.isclose(eval_Theta_arr, float(theta), atol=1e-8)
                    vals = residuals_all[mask]
                    if vals.size == 0:
                        continue

                    mean_pier = float(np.mean(vals))
                    std_pier = float(np.std(vals))
                    rows.append(
                        {
                            "ContextType": "dataset",
                            "ContextLabel": context_name,
                            "TargetModel": target_name,
                            "Group": (
                                "Shape-biased CNN"
                                if target_name.startswith("ShapeResNet50")
                                else "Standard CNN"
                            ),
                            "Dose": float(theta),
                            "MeanPIER": mean_pier,
                            "StdPIER": std_pier,
                            "NumEvalPoints": int(vals.size),
                            "NumFitPoints": int(len(fit_X)),
                            "ContextRawSize": int(n_ctx),
                            "ShapeVariant": variant,  # A / B / C
                        }
                    )

    if not rows:
        print("No rows collected. Something went wrong.")
        return

    df = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT_DIR, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    df.to_csv(out_path, index=False)

    print(f"\nSaved Exp 11 shape/texture PIER results to: {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 11: Shape-biased vs standard CNNs under texture vs shape contexts."
    )
    parser.add_argument(
        "--natural_root",
        type=str,
        # required=True,
        default="~/work3/username/imagenet",
        help="Root of natural ImageNet-like validation images (ImageFolder).",
    )
    parser.add_argument(
        "--shape_root",
        type=str,
        # required=True,
        default="~/work3/username/texture-vs-shape/stimuli/style-transfer-preprocessed-512",
        help="Root of shape-biased cue-conflict / stylized images (ImageFolder).",
    )
    parser.add_argument(
        "--max_samples_per_context",
        type=int,
        default=800,
        help="Maximum number of images per context.",
    )
    parser.add_argument(
        "--fit_fraction",
        type=float,
        default=0.5,
        help="Fraction of samples per context used for P_fit.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="exp12_shape_texture_pier_separated.csv",
        help="Output CSV file under results/tables/.",
    )

    args = parser.parse_args()

    run_shape_texture_experiment(
        natural_root=args.natural_root,
        shape_root=args.shape_root,
        max_samples_per_context=args.max_samples_per_context,
        fit_fraction=args.fit_fraction,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
