"""
Create 20 test views around the object to find the best starting angle.
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from plyfile import PlyData
import math

from render_with_gsplat import load_gaussian_splat_ply, render_gaussian_splat

# Load the gaussian splat
print("Loading...")
gaussian_data = load_gaussian_splat_ply('segmentedsplat.ply')

# Calculate object center and size
means_np = gaussian_data['means'].numpy()
center = means_np.mean(axis=0)
bbox_size = means_np.max(axis=0) - means_np.min(axis=0)
radius = np.max(bbox_size) * 2.5

print(f"Object center: {center}")
print(f"Camera radius: {radius:.3f}")

# Create output directory
output_dir = Path('test_views')
output_dir.mkdir(exist_ok=True)

# Render views from multiple axes
view_count = 0

# Axis 1: Horizontal circle (around Y axis) - 20 views
print(f"\nRendering horizontal circle (around Y axis)...")
for i in range(20):
    theta = (i / 20) * 2 * math.pi

    cam_x = center[0] + radius * math.cos(theta)
    cam_y = center[1]  # Same height
    cam_z = center[2] + radius * math.sin(theta)
    camera_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

    image = render_gaussian_splat(gaussian_data, camera_pos, center, 800, 600)

    if image is not None:
        angle = int((theta * 180 / math.pi) % 360)
        output_path = output_dir / f"view_{view_count:02d}_horizontal_{angle:03d}.png"
        Image.fromarray(image).save(output_path)
        print(f"View {view_count:02d}: Horizontal {angle:03d}°")
        view_count += 1

# Axis 2: Vertical arc over front (around X axis) - 10 views
print(f"\nRendering vertical arc over front (around X axis)...")
for i in range(10):
    phi = (i / 10) * math.pi - math.pi/2  # -90° to +90° (bottom to top)

    cam_x = center[0]  # Stay at center X
    cam_y = center[1] + radius * math.sin(phi)
    cam_z = center[2] - radius * math.cos(phi)  # Front side
    camera_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

    image = render_gaussian_splat(gaussian_data, camera_pos, center, 800, 600)

    if image is not None:
        angle = int((phi * 180 / math.pi) + 90)
        output_path = output_dir / f"view_{view_count:02d}_vertical_front_{angle:03d}.png"
        Image.fromarray(image).save(output_path)
        print(f"View {view_count:02d}: Vertical front {angle:03d}°")
        view_count += 1

# Axis 3: Vertical arc over side (around Z axis) - 10 views
print(f"\nRendering vertical arc over side (around Z axis)...")
for i in range(10):
    phi = (i / 10) * math.pi - math.pi/2  # -90° to +90°

    cam_x = center[0] + radius * math.cos(phi)  # Side
    cam_y = center[1] + radius * math.sin(phi)
    cam_z = center[2]  # Stay at center Z
    camera_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

    image = render_gaussian_splat(gaussian_data, camera_pos, center, 800, 600)

    if image is not None:
        angle = int((phi * 180 / math.pi) + 90)
        output_path = output_dir / f"view_{view_count:02d}_vertical_side_{angle:03d}.png"
        Image.fromarray(image).save(output_path)
        print(f"View {view_count:02d}: Vertical side {angle:03d}°")
        view_count += 1

print(f"\nDone! Check the images in {output_dir}")
print("Tell me which view number (00-19) shows the front, and I'll set that as the starting point!")
