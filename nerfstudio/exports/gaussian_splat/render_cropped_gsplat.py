"""
Render a cropped Gaussian Splat (.ply) from multiple camera angles.
This creates clean renders of just the object for dataset generation.
"""

import numpy as np
import torch
from pathlib import Path
import argparse
from PIL import Image
from plyfile import PlyData
import math


def load_ply_gaussian_splat(ply_path):
    """Load gaussian splat from .ply file."""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    # Extract positions
    positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

    # Extract other properties if available
    data = {
        'positions': torch.from_numpy(positions).float(),
    }

    # Try to extract other gaussian properties
    try:
        # Scales (if available)
        if 'scale_0' in vertex:
            scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
            data['scales'] = torch.from_numpy(scales).float()

        # Rotations (quaternions, if available)
        if 'rot_0' in vertex:
            rotations = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)
            data['rotations'] = torch.from_numpy(rotations).float()

        # Colors (if available as SH coefficients or RGB)
        if 'f_dc_0' in vertex:
            # SH coefficients
            sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)
            data['sh_dc'] = torch.from_numpy(sh_dc).float()
        elif 'red' in vertex:
            # RGB colors
            colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1) / 255.0
            data['colors'] = torch.from_numpy(colors).float()

        # Opacity
        if 'opacity' in vertex:
            data['opacity'] = torch.from_numpy(vertex['opacity']).float()

    except Exception as e:
        print(f"Warning: Could not extract some gaussian properties: {e}")

    print(f"Loaded {len(positions)} gaussians from {ply_path}")
    return data


def create_orbit_cameras(center, radius, num_views=100, height_variation=0.3):
    """
    Create camera poses orbiting around a center point.

    Args:
        center: 3D point to orbit around
        radius: Distance from center
        num_views: Number of camera positions
        height_variation: How much to vary height (0-1)

    Returns:
        List of camera matrices (4x4)
    """
    cameras = []

    for i in range(num_views):
        # Angle around the circle
        theta = (i / num_views) * 2 * math.pi

        # Vary height sinusoidally
        phi = math.sin(theta * 2) * height_variation

        # Camera position
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * height_variation * phi
        z = center[2] + radius * math.sin(theta)

        camera_pos = np.array([x, y, z])

        # Look at center
        forward = center - camera_pos
        forward = forward / np.linalg.norm(forward)

        # Up vector
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Camera matrix
        camera_matrix = np.eye(4)
        camera_matrix[:3, 0] = right
        camera_matrix[:3, 1] = up
        camera_matrix[:3, 2] = -forward
        camera_matrix[:3, 3] = camera_pos

        cameras.append(camera_matrix)

    return cameras


def render_gaussian_splat_simple(gaussian_data, camera_pos, look_at, width=800, height=600):
    """
    Render gaussian splat with proper camera transform.

    Args:
        gaussian_data: Dict with positions, colors, opacity
        camera_pos: Camera position (3D point)
        look_at: Point camera is looking at (3D point)
        width, height: Image dimensions
    """
    # Get positions
    positions = gaussian_data['positions'].numpy()

    # Get colors
    if 'colors' in gaussian_data:
        colors = gaussian_data['colors'].numpy()
    elif 'sh_dc' in gaussian_data:
        # Convert SH DC (0th order) to RGB
        SH_C0 = 0.28209479177387814
        colors = gaussian_data['sh_dc'].numpy() * SH_C0 + 0.5
        colors = np.clip(colors, 0, 1)
    else:
        colors = np.ones((len(positions), 3))

    # Get opacity
    if 'opacity' in gaussian_data:
        opacity = 1.0 / (1.0 + np.exp(-gaussian_data['opacity'].numpy()))  # sigmoid
    else:
        opacity = np.ones(len(positions))

    # Build camera coordinate system
    forward = look_at - camera_pos
    forward = forward / np.linalg.norm(forward)

    # Use negative Y as world up to fix orientation
    world_up = np.array([0, -1, 0])
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 0.001:  # Handle case when forward is parallel to world_up
        right = np.array([1, 0, 0])
    else:
        right = right / np.linalg.norm(right)

    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)

    # Transform points to camera space
    positions_cam = positions - camera_pos
    x_cam = np.dot(positions_cam, right)
    y_cam = np.dot(positions_cam, up)
    z_cam = np.dot(positions_cam, forward)

    # Perspective projection
    focal = width * 0.8  # Adjust field of view

    # Filter points behind camera
    valid = z_cam > 0.01

    # Project
    proj_x = (x_cam / z_cam * focal + width / 2).astype(int)
    proj_y = (-y_cam / z_cam * focal + height / 2).astype(int)  # Flip Y for image coords

    # Filter valid screen coordinates
    valid = valid & (proj_x >= 0) & (proj_x < width) & (proj_y >= 0) & (proj_y < height)

    # Create image
    image = np.ones((height, width, 3), dtype=np.float32)
    depth_buffer = np.full((height, width), np.inf)

    # Get scale for splat size (use scales if available, otherwise use distance)
    if 'scales' in gaussian_data:
        scales = np.exp(gaussian_data['scales'].numpy())  # Scales are in log space
        splat_sizes = scales.mean(axis=1) * focal / z_cam * 3  # Increase multiplier for larger splats
        splat_sizes = np.clip(splat_sizes, 3, 20).astype(int)
    else:
        # Default splat size based on distance
        splat_sizes = np.clip(focal / z_cam * 0.05, 5, 15).astype(int)

    # Sort by depth (far to near)
    valid_idx = np.where(valid)[0]
    depths = z_cam[valid_idx]
    sorted_indices = valid_idx[np.argsort(-depths)]  # Sort descending

    # Render gaussians
    for idx in sorted_indices:
        x, y = proj_x[idx], proj_y[idx]
        depth = z_cam[idx]
        alpha = opacity[idx]
        kernel_size = splat_sizes[idx]

        if alpha < 0.01:
            continue

        # Splat with gaussian kernel
        for dy in range(-kernel_size, kernel_size + 1):
            for dx in range(-kernel_size, kernel_size + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    dist_sq = dx*dx + dy*dy
                    # Gaussian falloff with wider spread
                    sigma = kernel_size / 2.0
                    weight = np.exp(-dist_sq / (2 * sigma**2)) * alpha
                    # Alpha blending
                    image[ny, nx] = image[ny, nx] * (1 - weight) + colors[idx] * weight

    return (image * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Render cropped Gaussian Splat from multiple angles')
    parser.add_argument('--ply-file', type=str, default='segmentedsplat.ply',
                        help='Path to cropped .ply file')
    parser.add_argument('--output-dir', type=str, default='cropped_renders',
                        help='Output directory for rendered images')
    parser.add_argument('--num-views', type=int, default=100,
                        help='Number of camera views')
    parser.add_argument('--width', type=int, default=800,
                        help='Image width')
    parser.add_argument('--height', type=int, default=600,
                        help='Image height')
    parser.add_argument('--radius', type=float, default=3.0,
                        help='Camera orbit radius')

    args = parser.parse_args()

    ply_path = Path(args.ply_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Gaussian Splat from {ply_path}...")
    gaussian_data = load_ply_gaussian_splat(ply_path)

    # Calculate center of the object
    positions = gaussian_data['positions'].numpy()
    center = positions.mean(axis=0)

    # Calculate bounding box and automatic radius
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    bbox_size = bbox_max - bbox_min
    max_extent = np.max(bbox_size)

    # Auto-calculate radius if not specified or if default value
    if args.radius == 3.0:  # Default value
        auto_radius = max_extent * 2.5  # 2.5x the largest dimension
        print(f"Object bounding box: {bbox_size}")
        print(f"Object max extent: {max_extent:.3f}")
        print(f"Auto-calculated camera radius: {auto_radius:.3f}")
        radius = auto_radius
    else:
        radius = args.radius

    print(f"Object center: {center}")
    print(f"Using camera radius: {radius:.3f}")

    # Render from each camera
    print(f"Rendering {args.num_views} views...")
    for i in range(args.num_views):
        # Calculate camera position (orbit around object)
        theta = (i / args.num_views) * 2 * math.pi
        phi = math.sin(theta * 2) * 0.3  # Slight height variation

        # Camera position
        cam_x = center[0] + radius * math.cos(theta)
        cam_y = center[1] + radius * 0.3 * phi
        cam_z = center[2] + radius * math.sin(theta)
        camera_pos = np.array([cam_x, cam_y, cam_z])

        # Render
        image = render_gaussian_splat_simple(
            gaussian_data,
            camera_pos,
            center,  # Look at object center
            args.width,
            args.height
        )

        # Save image
        output_path = output_dir / f"render_{i:04d}.png"
        Image.fromarray(image).save(output_path)

        if (i + 1) % 10 == 0:
            print(f"Rendered {i + 1}/{args.num_views} views")

    print(f"\nDone! Rendered images saved to {output_dir}")


if __name__ == '__main__':
    main()
