"""Render Gaussian splat with YOLO segmentation masks."""
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from plyfile import PlyData
import json, math, argparse

from gsplat import rasterization
from ultralytics import YOLO

def load_ply(path):
    v = PlyData.read(path)['vertex']
    return {
        'means': torch.from_numpy(np.stack([v['x'], v['y'], v['z']], 1).astype(np.float32)),
        'scales': torch.from_numpy(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1).astype(np.float32)),
        'quats': torch.from_numpy(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], 1).astype(np.float32)),
        'opacities': torch.from_numpy(v['opacity'].astype(np.float32)),
        'sh': torch.from_numpy(np.stack([v[f'f_dc_{i}'] for i in range(3)], 1).astype(np.float32))
    }

def camera_matrix(pos, look_at):
    fwd = (look_at - pos) / np.linalg.norm(look_at - pos)
    right = np.cross([0, 1, 0], fwd)
    right = right / (np.linalg.norm(right) + 1e-6)
    up = np.cross(fwd, right)
    vm = np.eye(4)
    vm[0, :3], vm[1, :3], vm[2, :3] = right, up, fwd
    vm[:3, 3] = -vm[:3, :3] @ pos
    return torch.from_numpy(vm).float()

def render(data, cam_pos, look_at, w=1920, h=1080, bg=(1, 1, 1)):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    means = data['means'].to(dev)
    scales = torch.exp(data['scales']).to(dev)
    quats = data['quats'].to(dev) / torch.norm(data['quats'].to(dev), dim=-1, keepdim=True)
    opacities = torch.sigmoid(data['opacities']).to(dev)
    colors = torch.clamp(data['sh'].to(dev) * 0.28209479177387814 + 0.5, 0, 1)

    vm = camera_matrix(cam_pos, look_at).to(dev)
    focal = w / (2 * np.tan(np.radians(50) / 2))
    K = torch.tensor([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=torch.float32, device=dev)

    rgb = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=vm.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=w,
        height=h,
        backgrounds=torch.tensor(bg, dtype=torch.float32, device=dev).unsqueeze(0)
    )[0].squeeze(0).cpu().numpy()

    return {
        'rgb': (rgb * 255).clip(0, 255).astype(np.uint8),
        'camera': {'pos': cam_pos.tolist(), 'look_at': look_at.tolist(), 'K': K.cpu().tolist(),
                   'focal': float(focal), 'size': [w, h]}
    }

def yolo_mask(rgb, model, conf=0.25, combine=True):
    results = model(rgb, conf=conf, verbose=False)[0]
    if results.masks is None or len(results.masks) == 0:
        return np.zeros(rgb.shape[:2], dtype=np.uint8)

    masks = results.masks.data.cpu().numpy()

    if combine:
        # Combine all masks
        combined = np.any(masks > 0.5, axis=0)
        return (combined * 255).astype(np.uint8)

    # Single largest mask
    areas = [mask.sum() for mask in masks]
    return (masks[np.argmax(areas)] * 255).astype(np.uint8)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ply', default='segmentedPLY.ply')
    p.add_argument('--out', default='renders_yolo')
    p.add_argument('--views', type=int, default=1)
    p.add_argument('--size', type=int, nargs=2, default=[1920, 1080])
    p.add_argument('--model', default='yolov8n-seg.pt', help='YOLO model')
    p.add_argument('--conf', type=float, default=0.1, help='YOLO confidence')
    p.add_argument('--combine', action='store_true', default=True, help='Combine all masks')
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    print(f"Loading {args.ply}...")
    data = load_ply(args.ply)
    center = data['means'].numpy().mean(0)
    radius = np.max(data['means'].numpy().max(0) - data['means'].numpy().min(0)) * 2.5

    print(f"Loading YOLO {args.model}...")
    yolo = YOLO(args.model)

    print(f"Rendering {args.views} views...")
    for i in range(args.views):
        theta = 0 if args.views == 1 else i / args.views * 2 * math.pi
        phi = math.radians(15) if args.views == 1 else math.radians(30 * math.sin(i / args.views * 2 * math.pi))

        pos = center + radius * np.array([math.cos(phi) * math.sin(theta),
                                          math.sin(phi),
                                          math.cos(phi) * math.cos(theta)])

        result = render(data, pos, center, *args.size)
        name = "render" if args.views == 1 else f"render_{i:04d}"

        Image.fromarray(result['rgb']).save(out / f"{name}.png")
        mask = yolo_mask(result['rgb'], yolo, args.conf, args.combine)
        Image.fromarray(mask, 'L').save(out / f"{name}_mask.png")
        json.dump(result['camera'], open(out / f"{name}_camera.json", 'w'), indent=2)

        print(f"  ✓ {name}")

    print(f"\n✓ Done: {out}/")

if __name__ == '__main__':
    main()
