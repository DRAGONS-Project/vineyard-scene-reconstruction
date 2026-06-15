"""Train a basic 3D Gaussian Splatting model from a COLMAP sparse reconstruction using gsplat."""

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap
import torch
from gsplat import rasterization
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import load_config


def load_colmap_scene(sparse_dir: Path, image_dir: Path, device: torch.device) -> dict:
    """Load camera intrinsics/extrinsics, images, and the sparse point cloud.

    Images are returned sorted by filename for a deterministic train/test split.
    """
    recon = pycolmap.Reconstruction(str(sparse_dir))

    viewmats, Ks, images, names = [], [], [], []
    width = height = None

    for image in sorted(recon.images.values(), key=lambda im: im.name):
        cam = recon.cameras[image.camera_id]
        width, height = cam.width, cam.height

        K = np.eye(3, dtype=np.float32)
        params = cam.params
        if cam.model.name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            f, cx, cy = params[0], params[1], params[2]
            K[0, 0] = K[1, 1] = f
        else:  # PINHOLE, OPENCV, ...
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            K[0, 0], K[1, 1] = fx, fy
        K[0, 2], K[1, 2] = cx, cy
        Ks.append(K)

        cam_from_world = image.cam_from_world()
        world_to_cam = np.eye(4, dtype=np.float32)
        world_to_cam[:3, :3] = cam_from_world.rotation.matrix()
        world_to_cam[:3, 3] = cam_from_world.translation
        viewmats.append(world_to_cam)

        img = Image.open(image_dir / image.name).convert("RGB").resize((width, height))
        images.append(np.asarray(img, dtype=np.float32) / 255.0)
        names.append(image.name)

    xyz = np.stack([p.xyz for p in recon.points3D.values()]).astype(np.float32)
    rgb = np.stack([p.color for p in recon.points3D.values()]).astype(np.float32) / 255.0

    return {
        "viewmats": torch.tensor(np.stack(viewmats), device=device),
        "Ks": torch.tensor(np.stack(Ks), device=device),
        "images": torch.tensor(np.stack(images), device=device),
        "names": names,
        "width": width,
        "height": height,
        "points_xyz": torch.tensor(xyz, device=device),
        "points_rgb": torch.tensor(rgb, device=device),
    }


def split_train_test(n_views: int, holdout_every: int) -> tuple[list[int], list[int]]:
    """Hold out every Nth view (by sorted order) for evaluation."""
    if holdout_every <= 0:
        return list(range(n_views)), []
    test = list(range(0, n_views, holdout_every))
    train = [i for i in range(n_views) if i not in test]
    return train, test


class GaussianModel(torch.nn.Module):
    """Minimal set of optimizable Gaussian primitives, initialized from a sparse point cloud."""

    def __init__(self, points_xyz: torch.Tensor, points_rgb: torch.Tensor, init_scale: float = 0.01):
        super().__init__()
        n = points_xyz.shape[0]
        device = points_xyz.device

        self.means = torch.nn.Parameter(points_xyz.clone())
        self.scales = torch.nn.Parameter(torch.full((n, 3), init_scale, device=device).log())

        quats = torch.zeros((n, 4), device=device)
        quats[:, 0] = 1.0
        self.quats = torch.nn.Parameter(quats)

        self.opacities = torch.nn.Parameter(torch.logit(torch.full((n,), 0.5, device=device)))
        self.colors = torch.nn.Parameter(points_rgb.clone())

    def render(self, viewmat: torch.Tensor, K: torch.Tensor, width: int, height: int):
        renders, alphas, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales.exp(),
            opacities=self.opacities.sigmoid(),
            colors=self.colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
        )
        return renders[0], alphas[0]


def train(scene: dict, train_idx: list[int], output_dir: Path, num_iters: int, save_every: int) -> GaussianModel:
    device = scene["points_xyz"].device
    model = GaussianModel(scene["points_xyz"], scene["points_rgb"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    renders_dir = output_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    for it in range(num_iters):
        view_idx = train_idx[it % len(train_idx)]
        rendered, _ = model.render(
            scene["viewmats"][view_idx], scene["Ks"][view_idx], scene["width"], scene["height"]
        )
        target = scene["images"][view_idx]
        loss = torch.nn.functional.l1_loss(rendered, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (it + 1) % save_every == 0 or it == num_iters - 1:
            print(f"iter {it + 1}/{num_iters}  loss={loss.item():.4f}")
            img = (rendered.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(renders_dir / f"iter_{it + 1:06d}_view{view_idx:03d}.png")

    return model


@torch.no_grad()
def evaluate(model: GaussianModel, scene: dict, test_idx: list[int], output_dir: Path) -> dict:
    """Render held-out views and compute PSNR/SSIM against ground truth."""
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    per_view = []
    for view_idx in test_idx:
        rendered, _ = model.render(
            scene["viewmats"][view_idx], scene["Ks"][view_idx], scene["width"], scene["height"]
        )
        pred = rendered.clamp(0, 1).cpu().numpy()
        target = scene["images"][view_idx].cpu().numpy()

        psnr = float(peak_signal_noise_ratio(target, pred, data_range=1.0))
        ssim = float(structural_similarity(target, pred, data_range=1.0, channel_axis=2))
        per_view.append({"name": scene["names"][view_idx], "psnr": psnr, "ssim": ssim})

        Image.fromarray((pred * 255).astype(np.uint8)).save(
            eval_dir / f"pred_{scene['names'][view_idx]}.png"
        )
        Image.fromarray((target * 255).astype(np.uint8)).save(
            eval_dir / f"gt_{scene['names'][view_idx]}.png"
        )

    metrics = {
        "n_test_views": len(test_idx),
        "psnr_mean": float(np.mean([v["psnr"] for v in per_view])) if per_view else None,
        "ssim_mean": float(np.mean([v["ssim"] for v in per_view])) if per_view else None,
        "per_view": per_view,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def save_ply(model: GaussianModel, output_path: Path) -> None:
    from plyfile import PlyData, PlyElement

    means = model.means.detach().cpu().numpy()
    colors = (model.colors.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

    vertex = np.empty(
        means.shape[0],
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = means[:, 0], means[:, 1], means[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(output_path))


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sparse_dir", type=Path, help="COLMAP sparse reconstruction directory")
    parser.add_argument("image_dir", type=Path, help="Directory of input images")
    parser.add_argument(
        "output_dir", type=Path, help="Directory to write renders, checkpoint, and point cloud"
    )
    parser.add_argument("--iters", type=int, default=config.train.iters, help="Number of training iterations")
    parser.add_argument(
        "--save-every", type=int, default=config.train.save_every, help="Save a render every N iterations"
    )
    parser.add_argument(
        "--holdout-every",
        type=int,
        default=config.train.holdout_every,
        help="Hold out every Nth view (sorted order) for evaluation; 0 disables holdout",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = load_colmap_scene(args.sparse_dir, args.image_dir, device)
    n_views = scene["viewmats"].shape[0]
    train_idx, test_idx = split_train_test(n_views, args.holdout_every)
    print(
        f"Loaded {n_views} views ({len(train_idx)} train / {len(test_idx)} test), "
        f"{scene['points_xyz'].shape[0]} points"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = train(scene, train_idx, args.output_dir, args.iters, args.save_every)
    save_ply(model, args.output_dir / "point_cloud.ply")
    torch.save(model.state_dict(), args.output_dir / "model.pt")

    if test_idx:
        metrics = evaluate(model, scene, test_idx, args.output_dir)
        print(f"Held-out PSNR: {metrics['psnr_mean']:.2f}  SSIM: {metrics['ssim_mean']:.4f}")


if __name__ == "__main__":
    main()
