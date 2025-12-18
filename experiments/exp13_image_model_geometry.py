# experiments/exp13_image_model_geometry.py
#
# Exp 13: 2D convex-hull geometry of PIER in model-output space.
#
# Setting (aligned with Exp12):
#   - We consider three small ecosystems:
#       Ecosystem A: {ResNet50, EfficientNetB0, ConvNeXtTiny, ViT_B16, ShapeResNet50_SIN}
#       Ecosystem B: {ResNet50, EfficientNetB0, ConvNeXtTiny, ViT_B16, ShapeResNet50_SININ}
#       Ecosystem C: {ResNet50, EfficientNetB0, ConvNeXtTiny, ViT_B16, ShapeResNet50_ShapeResNet}
#   - In each ecosystem, the *only* target is the corresponding shape-biased model
#     (A, B, or C). The four standard models always act as peers.
#
#   - For each ecosystem and each context ("texture_natural" vs "shape_bias"), we:
#       1) Build P_fit as in Exp11 (same max_samples_per_context and fit_fraction).
#       2) For each model in the ecosystem, collect a scalar output vector on P_fit:
#            v_j = [g_j(x_1), ..., g_j(x_N)]^T
#          where g_j is the scalar returned by ImageModelWrapper (e.g. p_true).
#       3) For the target, form y_t = v_target, and peers matrix Y_P = [v_p1, ..., v_p4].
#          Use DISCOSolver to find the optimal convex combination w_hat.
#          Define v_mix = sum_j w_hat_j * v_pj.
#       4) Stack all peer vectors, v_target, v_mix into a matrix X and apply PCA to 2D.
#       5) In 2D, compute the convex hull of peer points, and save:
#            - 2D coordinates of all models + mix
#            - peer mask, hull vertices, target index, mix index, w_hat, etc.
#
# The plotting is done in the companion notebook 13_image_model_geometry.ipynb.

import os
import sys
from typing import List, Tuple, Dict, Optional

import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

# Make local package importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from isqed.geometry import DISCOSolver
from isqed.real_world import ImageModelWrapper

SCALAR_MODE = "p_true"  # or "margin", depending on how your ImageModelWrapper is implemented


# ============================================================
# 0. Preprocessing
# ============================================================

def build_imagenet_transform() -> transforms.Compose:
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
    variant: str,
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

    new_state_dict = {}
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

def load_context_samples(
    root: str,
    max_samples: int,
    np_rng: np.random.RandomState,
    context_name: str,
):
    """
    Load up to `max_samples` images from an ImageFolder root.
    Returns:
        samples: list of (x_tensor, y_int)
        ids:     list of string identifiers (relative file paths)
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

    samples = []
    ids = []
    for idx in indices:
        x, y = dataset[idx]
        path, _ = dataset.samples[idx]
        samples.append((x, int(y)))
        ids.append(os.path.relpath(path, root))

    return samples, ids


def split_fit_eval_indices(
    n: int,
    np_rng: np.random.RandomState,
    fit_fraction: float = 0.5,
):
    idx = np.arange(n)
    np_rng.shuffle(idx)
    n_fit = int(n * fit_fraction)
    fit_idx = idx[:n_fit]
    eval_idx = idx[n_fit:]
    return fit_idx, eval_idx


# ============================================================
# 3. Core geometry experiment
# ============================================================

def collect_scalar_vector(
    wrapper: ImageModelWrapper,
    samples: List[Tuple[torch.Tensor, int]],
):
    vals = []
    for sample in samples:
        vals.append(wrapper._forward(sample))
    return np.asarray(vals, dtype=float)


def run_model_geometry_experiment(
    natural_root: str,
    shape_root: str,
    max_samples_per_context: int = 800,
    fit_fraction: float = 0.5,
    output_prefix: str = "exp13_geometry",
):
    """
    Run convex-hull geometry experiment for three ecosystems:

      Ecosystem A: target ShapeResNet50_SIN, peers = standard models
      Ecosystem B: target ShapeResNet50_SININ, peers = standard models
      Ecosystem C: target ShapeResNet50_ShapeResNet, peers = standard models

    For each ecosystem and each context (texture_natural / shape_bias),
    saves an npz under results/artifacts/exp13/.
    """
    from sklearn.decomposition import PCA
    from scipy.spatial import ConvexHull

    print("=== Exp 13: Model-output convex-hull geometry (A/B/C ecosystems) ===")

    np_rng = np.random.RandomState(0)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Load standard models once
    print("\nLoading standard models...")
    models_std = load_standard_models(device=device)
    standard_names = list(models_std.keys())
    print(f"  Standard models: {standard_names}")

    # Shape variants A/B/C
    shape_variants = ["A", "B", "C"]

    # 2) Prepare contexts
    contexts = [
        ("texture_natural", natural_root),
        ("shape_bias", shape_root),
    ]

    out_dir = os.path.join(ROOT_DIR, "results", "artifacts", "exp13")
    os.makedirs(out_dir, exist_ok=True)

    # Pre-load datasets and P_fit splits once per context (shared across ecosystems)
    contexts_cache = {}
    for context_name, root in contexts:
        print(f"\n[Data] Pre-loading context: {context_name} from {root}")
        samples_ctx, ids_ctx = load_context_samples(
            root=root,
            max_samples=max_samples_per_context,
            np_rng=np_rng,
            context_name=context_name,
        )
        n_ctx = len(samples_ctx)
        fit_idx, eval_idx = split_fit_eval_indices(n_ctx, np_rng, fit_fraction)
        fit_samples = [samples_ctx[i] for i in fit_idx]
        fit_ids = [ids_ctx[i] for i in fit_idx]

        print(f"  Using {len(fit_samples)} samples for P_fit (out of {n_ctx}).")

        contexts_cache[context_name] = {
            "fit_samples": fit_samples,
            "fit_ids": fit_ids,
            "n_ctx": n_ctx,
        }

    # 3) For each shape variant ecosystem
    for variant in shape_variants:
        print(f"\n==============================")
        print(f"=== Ecosystem variant {variant} ===")
        print(f"==============================")

        # load corresponding shape model
        shape_model = load_shape_biased_resnet50(device=device, variant=variant)
        target_name = shape_model.name

        # ecosystem models: 4 standard + this shape model
        eco_models: Dict[str, ImageModelWrapper] = {}
        eco_models.update(models_std)
        eco_models[target_name] = shape_model

        peers = standard_names  # peers are always the standard models

        for context_name, _root in contexts:
            cache = contexts_cache[context_name]
            fit_samples = cache["fit_samples"]
            fit_ids = cache["fit_ids"]
            n_ctx = cache["n_ctx"]

            print(f"\n  [Context={context_name}] Target={target_name}, peers={peers}")

            # 3.1 Collect scalar vectors for peers + target on P_fit
            vectors: Dict[str, np.ndarray] = {}
            for name in peers + [target_name]:
                vec = collect_scalar_vector(eco_models[name], fit_samples)
                vectors[name] = vec

            target_vec = vectors[target_name]
            peer_vecs = [vectors[name] for name in peers]

            y_t_fit_vec = target_vec.reshape(-1, 1)        # (N, 1)
            Y_p_fit_mat = np.stack(peer_vecs, axis=1)      # (N, num_peers)

            # 3.2 Solve convex weights
            _, w_hat = DISCOSolver.solve_weights_and_distance(y_t_fit_vec, Y_p_fit_mat)
            w_hat = np.asarray(w_hat, dtype=float).flatten()
            print(f"    Learned w_hat (len={len(w_hat)}), first 3 = {w_hat[:3]}")

            # 3.3 Form convex-mix vector
            v_mix = np.zeros_like(target_vec, dtype=float)
            for w, v in zip(w_hat, peer_vecs):
                v_mix += w * v

            # High dimension L2 distance
            l2_dist = float(np.linalg.norm(target_vec - v_mix))

            # 3.4 PCA to 2D
            all_vecs = peer_vecs + [target_vec, v_mix]
            labels = peers + [target_name, "peer_mix"]
            X = np.stack(all_vecs, axis=0)  # (num_peers+2, N_fit)

            pca = PCA(n_components=2)
            coords = pca.fit_transform(X)   # (num_peers+2, 2)
            explained = pca.explained_variance_ratio_

            peer_coords = coords[:len(peers)]
            target_coord = coords[len(peers)]
            mix_coord = coords[len(peers) + 1]

            hull = ConvexHull(peer_coords)
            hull_vertices = hull.vertices  # indices into peers

            # 3.5 Save npz
            peer_mask = np.array([i < len(peers) for i in range(len(labels))], dtype=bool)
            target_idx = len(peers)
            mix_idx = len(peers) + 1

            out_name = f"{output_prefix}_{target_name}_{context_name}.npz"
            out_path = os.path.join(out_dir, out_name)
            np.savez_compressed(
                out_path,
                coords=coords,
                labels=np.array(labels),
                peer_mask=peer_mask,
                hull_vertices=hull_vertices,
                target_idx=target_idx,
                mix_idx=mix_idx,
                context=context_name,
                target_name=target_name,
                w_hat=w_hat,
                explained_variance=explained,
                fit_ids=np.array(fit_ids),
                peers=np.array(peers),
                l2_dist=l2_dist,   
            )

            print(f"    Saved geometry to: {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 13: Model-output convex-hull geometry of PIER (A/B/C ecosystems)."
    )
    parser.add_argument(
        "--natural_root",
        type=str,
        default="~/work3/leiyo/imagenet",
        help="Root of natural ImageNet-like validation images (ImageFolder).",
    )
    parser.add_argument(
        "--shape_root",
        type=str,
        default="~/work3/leiyo/texture-vs-shape/stimuli/style-transfer-preprocessed-512",
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
        "--output_prefix",
        type=str,
        default="exp13_geometry",
        help="Prefix for geometry npz files under results/artifacts/exp13/.",
    )

    args = parser.parse_args()

    run_model_geometry_experiment(
        natural_root=args.natural_root,
        shape_root=args.shape_root,
        max_samples_per_context=args.max_samples_per_context,
        fit_fraction=args.fit_fraction,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
