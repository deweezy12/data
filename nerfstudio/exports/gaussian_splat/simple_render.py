"""Simple point cloud renderer to debug"""

import numpy as np
from plyfile import PlyData
from PIL import Image, ImageDraw

# Load PLY
print("Loading...")
ply = PlyData.read('segmentedsplat.ply')
vertex = ply['vertex']

positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)

# Convert colors
SH_C0 = 0.28209479177387814
colors = np.clip(sh_dc * SH_C0 + 0.5, 0, 1)

print(f"Loaded {len(positions)} points")
print(f"Position center: {positions.mean(axis=0)}")
print(f"Position range: {positions.max(axis=0) - positions.min(axis=0)}")

# Simple orthographic projection from front view
width, height = 800, 600

# Center object
center = positions.mean(axis=0)
positions_centered = positions - center

# Find scale
max_extent = np.abs(positions_centered).max()
scale = min(width, height) * 0.4 / max_extent  # Use 40% of image

print(f"Max extent: {max_extent:.3f}")
print(f"Scale: {scale:.1f}")

# Project: X -> image X, Y -> image Y (ignore Z for now)
proj_x = (positions_centered[:, 0] * scale + width / 2).astype(int)
proj_y = (positions_centered[:, 1] * scale + height / 2).astype(int)

# Create image
image = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)

# Draw points
valid_mask = (proj_x >= 0) & (proj_x < width) & (proj_y >= 0) & (proj_y < height)
print(f"Valid points: {valid_mask.sum()} / {len(positions)}")

for i in np.where(valid_mask)[0]:
    x, y = proj_x[i], proj_y[i]
    color = (int(colors[i, 0] * 255), int(colors[i, 1] * 255), int(colors[i, 2] * 255))  # RGB
    draw.ellipse([x-2, y-2, x+2, y+2], fill=color)

# Save
output_path = 'test_simple_render.png'
image.save(output_path)
print(f"Saved to {output_path}")

# Also try from different angles
for angle_name, axis_idx in [("side", 2), ("top", 1)]:
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    if axis_idx == 2:  # Side view (X, Z)
        proj_x = (positions_centered[:, 0] * scale + width / 2).astype(int)
        proj_y = (positions_centered[:, 2] * scale + height / 2).astype(int)
    else:  # Top view (X, Y)
        proj_x = (positions_centered[:, 0] * scale + width / 2).astype(int)
        proj_y = (positions_centered[:, 1] * scale + height / 2).astype(int)

    valid_mask = (proj_x >= 0) & (proj_x < width) & (proj_y >= 0) & (proj_y < height)

    for i in np.where(valid_mask)[0]:
        x, y = proj_x[i], proj_y[i]
        color = (int(colors[i, 0] * 255), int(colors[i, 1] * 255), int(colors[i, 2] * 255))  # RGB
        draw.ellipse([x-2, y-2, x+2, y+2], fill=color)

    output_path = f'test_render_{angle_name}.png'
    image.save(output_path)
    print(f"Saved to {output_path}")

print("\nDone! Check the test_*.png files")
