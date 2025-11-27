"""Debug script to check what's in the .ply file"""

import numpy as np
from plyfile import PlyData

ply = PlyData.read('segmentedsplat.ply')
vertex = ply['vertex']

print(f"Number of gaussians: {len(vertex)}")
print(f"\nPosition stats:")
positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
print(f"  Min: {positions.min(axis=0)}")
print(f"  Max: {positions.max(axis=0)}")
print(f"  Mean: {positions.mean(axis=0)}")

print(f"\nColor stats (SH DC):")
sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)
print(f"  Min: {sh_dc.min(axis=0)}")
print(f"  Max: {sh_dc.max(axis=0)}")
print(f"  Mean: {sh_dc.mean(axis=0)}")

# Convert to RGB
SH_C0 = 0.28209479177387814
colors_rgb = sh_dc * SH_C0 + 0.5
colors_rgb = np.clip(colors_rgb, 0, 1)
print(f"\nConverted RGB colors:")
print(f"  Min: {colors_rgb.min(axis=0)}")
print(f"  Max: {colors_rgb.max(axis=0)}")
print(f"  Mean: {colors_rgb.mean(axis=0)}")
print(f"  First 5 colors:\n{colors_rgb[:5]}")

print(f"\nOpacity stats:")
opacity_raw = vertex['opacity']
print(f"  Raw min: {opacity_raw.min()}")
print(f"  Raw max: {opacity_raw.max()}")
print(f"  Raw mean: {opacity_raw.mean()}")

opacity = 1.0 / (1.0 + np.exp(-opacity_raw))
print(f"  After sigmoid min: {opacity.min()}")
print(f"  After sigmoid max: {opacity.max()}")
print(f"  After sigmoid mean: {opacity.mean()}")
print(f"  Number with opacity > 0.5: {(opacity > 0.5).sum()}")

print(f"\nScale stats:")
scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
print(f"  Min: {scales.min(axis=0)}")
print(f"  Max: {scales.max(axis=0)}")
print(f"  Mean: {scales.mean(axis=0)}")
