"""Train a basic 3D Gaussian Splatting model from a COLMAP sparse reconstruction using gsplat."""

import argparse
from pathlib import Path

import numpy as np
import pycolmap
import torch
from gsplat import rasterization
from PIL import Image


def load_colmap_scene(sparse_dir: Path, image_dir: Path, device: torch.device) -> dict:
    """Load camera intrinsics/extrinsics, images, and the sparse point cloud."""
    recon = pycolmap.Reconstruction(str(sparse_dir))

    viewmats, Ks, images = [], [], []
    width = height = None

    for image in recon.images.values():
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

        world_to_cam = np.eye(4, dtype=np.float32)
        world_to_cam[:3, :3] = image.cam_from_world.rotation.matrix()
        world_to_cam[:3, 3] = image.cam_from_world.translation
        viewmats.append(world_to_cam)

        img = Image.open(image_dir / image.name).convert("RGB").resize((width, height))
        images.append(np.asarray(img, dtype=np.float32) / 255.0)

    xyz = np.stack([p.xyz for p in recon.points3D.values()]).astype(np.float32)
    rgb = np.stack([p.color for p in recon.points3D.values()]).astype(np.float32) / 255.0

    return {
        "viewmats": torch.tensor(np.stack(viewmats), device=device),
        "Ks": torch.tensor(np.stack(Ks), device=device),
        "images": torch.tensor(np.stack(images), device=device),
        "width": width,
        "height": height,
        "points_xyz": torch.tensor(xyz, device=device),
        "points_rgb": torch.tensor(rgb, device=device),
    }


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


def train(scene: dict, output_dir: Path, num_iters: int, save_every: int) -> GaussianModel:
    device = scene["points_xyz"].device
    model = GaussianModel(scene["points_xyz"], scene["points_rgb"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    n_views = scene["viewmats"].shape[0]
    renders_dir = output_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    for it in range(num_iters):
        view_idx = it % n_views
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sparse_dir", type=Path, help="COLMAP sparse reconstruction directory")
    parser.add_argument("image_dir", type=Path, help="Directory of input images")
    parser.add_argument(
        "output_dir", type=Path, help="Directory to write renders, checkpoint, and point cloud"
    )
    parser.add_argument("--iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--save-every", type=int, default=100, help="Save a render every N iterations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = load_colmap_scene(args.sparse_dir, args.image_dir, device)
    print(f"Loaded {scene['viewmats'].shape[0]} views, {scene['points_xyz'].shape[0]} points")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = train(scene, args.output_dir, args.iters, args.save_every)
    save_ply(model, args.output_dir / "point_cloud.ply")
    torch.save(model.state_dict(), args.output_dir / "model.pt")


if __name__ == "__main__":
    main()
