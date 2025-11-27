"""
Proper Gaussian Splatting renderer using the gsplat library.
This actually renders the gaussians correctly with their shapes and rotations.
"""

import numpy as np
import torch
from pathlib import Path
import argparse
from PIL import Image
from plyfile import PlyData
import math

try:
    from gsplat import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("ERROR: gsplat library not installed!")
    print("Install it with: pip install gsplat")
    exit(1)


def load_gaussian_splat_ply(ply_path):
    """Load gaussian splat data from .ply file."""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    # Positions (means)
    means = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)

    # Scales (log space)
    scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1).astype(np.float32)

    # Rotations (quaternions: w, x, y, z)
    quats = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1).astype(np.float32)

    # Opacity (pre-sigmoid)
    opacities = vertex['opacity'].astype(np.float32)

    # Spherical harmonics (colors)
    sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1).astype(np.float32)

    # Optional: Higher order SH coefficients
    sh_rest = []
    for i in range(45):  # Up to 3rd order SH (45 coefficients)
        key = f'f_rest_{i}'
        if key in vertex:
            sh_rest.append(vertex[key])

    if sh_rest:
        sh_rest = np.stack(sh_rest, axis=1).astype(np.float32)
        sh_features = np.concatenate([sh_dc, sh_rest], axis=1)
    else:
        sh_features = sh_dc

    print(f"Loaded {len(means)} gaussians")
    print(f"  Means shape: {means.shape}")
    print(f"  Scales shape: {scales.shape}")
    print(f"  Quats shape: {quats.shape}")
    print(f"  SH features shape: {sh_features.shape}")

    return {
        'means': torch.from_numpy(means),
        'scales': torch.from_numpy(scales),
        'quats': torch.from_numpy(quats),
        'opacities': torch.from_numpy(opacities),
        'sh': torch.from_numpy(sh_features),
    }


def create_camera_matrix(camera_pos, look_at, up=np.array([0, 1, 0])):
    """Create view and projection matrices."""
    # View matrix
    forward = look_at - camera_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up, forward)
    if np.linalg.norm(right) < 0.001:
        right = np.array([1, 0, 0])
    else:
        right = right / np.linalg.norm(right)

    camera_up = np.cross(forward, right)
    camera_up = camera_up / np.linalg.norm(camera_up)

    # Create view matrix (world to camera)
    view_matrix = np.eye(4)
    view_matrix[0, :3] = right
    view_matrix[1, :3] = camera_up
    view_matrix[2, :3] = forward
    view_matrix[:3, 3] = -view_matrix[:3, :3] @ camera_pos

    return torch.from_numpy(view_matrix).float()


def render_gaussian_splat(gaussian_data, camera_pos, look_at, width=800, height=600, background_color=(1.0, 1.0, 1.0)):
    """Render gaussian splat using gsplat library.

    Args:
        background_color: RGB tuple (0-1 range), e.g. (1, 1, 1) for white, (0, 0, 0) for black
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move data to device
    means = gaussian_data['means'].to(device)
    scales = torch.exp(gaussian_data['scales']).to(device)  # Convert from log space
    quats = gaussian_data['quats'].to(device)
    opacities = torch.sigmoid(gaussian_data['opacities']).to(device)  # Apply sigmoid
    sh = gaussian_data['sh'].to(device)

    # Create camera parameters
    view_matrix = create_camera_matrix(camera_pos, look_at).to(device)

    # Projection matrix (perspective)
    fov = 50  # degrees
    focal = width / (2 * np.tan(np.radians(fov) / 2))

    K = torch.tensor([
        [focal, 0, width / 2],
        [0, focal, height / 2],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)

    # Convert SH to RGB (use DC component only for simplicity)
    SH_C0 = 0.28209479177387814
    colors_rgb = torch.clamp(sh[:, :3] * SH_C0 + 0.5, 0, 1)

    # Create background color tensor
    bg_color = torch.tensor(background_color, dtype=torch.float32, device=device)

    # Render using gsplat
    try:
        rendered = rasterization(
            means=means,
            quats=quats / torch.norm(quats, dim=-1, keepdim=True),  # Normalize quaternions
            scales=scales,
            opacities=opacities,
            colors=colors_rgb,
            viewmats=view_matrix.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            backgrounds=bg_color.unsqueeze(0),  # Set background color
        )

        # Extract RGB image
        rgb = rendered[0].squeeze(0).cpu().numpy()  # [H, W, 3]
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        return rgb
    except Exception as e:
        print(f"Error during rendering: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Render Gaussian Splat using gsplat library')
    parser.add_argument('--ply-file', type=str, default='segmentedsplat.ply')
    parser.add_argument('--output-dir', type=str, default='gsplat_renders')
    parser.add_argument('--num-views', type=int, default=100)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--bg-mode', type=str, default='white',
                        choices=['white', 'black', 'random', 'cycle'],
                        help='Background color mode')

    args = parser.parse_args()

    # Define color palette for cycling
    color_palette = [
        (1.0, 1.0, 1.0),  # White
        (0.0, 0.0, 0.0),  # Black
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0),  # Magenta
        (0.0, 1.0, 1.0),  # Cyan
        (0.5, 0.5, 0.5),  # Gray
        (1.0, 0.5, 0.0),  # Orange
    ]

    if not GSPLAT_AVAILABLE:
        return

    ply_path = Path(args.ply_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Gaussian Splat from {ply_path}...")
    gaussian_data = load_gaussian_splat_ply(ply_path)

    # Calculate object center and size
    means_np = gaussian_data['means'].numpy()
    center = means_np.mean(axis=0)
    bbox_size = means_np.max(axis=0) - means_np.min(axis=0)
    radius = np.max(bbox_size) * 2.5

    print(f"Object center: {center}")
    print(f"Camera radius: {radius:.3f}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Render from multiple views - front hemisphere with all angles
    print(f"\nRendering {args.num_views} views around front hemisphere...")
    for i in range(args.num_views):
        progress = i / args.num_views

        # Horizontal: 180° arc around front (left → center → right)
        theta = (progress * math.pi) - math.pi/2  # -90° to +90°

        # Vertical: Stay in front hemisphere only
        # Bottom (-90°) to eye level (0°), not going to back/top
        phi = -math.pi/2 + (progress * math.pi/2)  # -90° (bottom) to 0° (eye level)

        # Spherical to Cartesian - front hemisphere only
        # Using the same coordinate system as view_20 (vertical_front)
        cam_x = center[0] + radius * math.cos(phi) * math.sin(theta)
        cam_y = center[1] + radius * math.sin(phi)
        cam_z = center[2] - radius * math.cos(phi) * math.cos(theta)
        camera_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        # Select background color based on mode
        if args.bg_mode == 'white':
            bg_color = (1.0, 1.0, 1.0)
        elif args.bg_mode == 'black':
            bg_color = (0.0, 0.0, 0.0)
        elif args.bg_mode == 'random':
            bg_color = (np.random.rand(), np.random.rand(), np.random.rand())
        elif args.bg_mode == 'cycle':
            bg_color = color_palette[i % len(color_palette)]

        image = render_gaussian_splat(
            gaussian_data,
            camera_pos,
            center,
            args.width,
            args.height,
            background_color=bg_color
        )

        if image is not None:
            output_path = output_dir / f"render_{i:04d}.png"
            Image.fromarray(image).save(output_path)

            if (i + 1) % 10 == 0:
                print(f"Rendered {i + 1}/{args.num_views} views")
        else:
            print(f"Failed to render view {i}")
            if i == 0:  # If first view fails, stop
                print("First view failed, stopping...")
                break

    print(f"\nDone! Renders saved to {output_dir}")


if __name__ == '__main__':
    main()
