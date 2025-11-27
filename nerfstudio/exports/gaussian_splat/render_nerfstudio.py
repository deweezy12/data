"""
Use nerfstudio's own rendering to generate views.
This properly renders the gaussian splat using the trained model.
"""

import subprocess
import json
from pathlib import Path
import math
import numpy as np


def create_camera_path_json(center, radius, num_views, output_file):
    """Create a camera path JSON file for nerfstudio."""

    camera_path = {
        "camera_type": "perspective",
        "render_height": 600,
        "render_width": 800,
        "camera_path": [],
        "fps": 24,
        "seconds": num_views / 24.0,
        "smoothness_value": 0,
        "is_cycle": True,
    }

    for i in range(num_views):
        theta = (i / num_views) * 2 * math.pi
        phi_variation = math.sin(theta * 2) * 0.3

        # Camera position
        cam_x = float(center[0] + radius * math.cos(theta))
        cam_y = float(center[1] + radius * 0.3 * phi_variation)
        cam_z = float(center[2] + radius * math.sin(theta))

        # Look at center
        forward = center - np.array([cam_x, cam_y, cam_z])
        forward = forward / np.linalg.norm(forward)

        # Create camera transform matrix
        up = np.array([0.0, -1.0, 0.0])
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)

        camera_up = np.cross(forward, right)

        # Camera to world matrix
        camera_to_world = [
            [float(right[0]), float(camera_up[0]), float(forward[0]), float(cam_x)],
            [float(right[1]), float(camera_up[1]), float(forward[1]), float(cam_y)],
            [float(right[2]), float(camera_up[2]), float(forward[2]), float(cam_z)],
            [0.0, 0.0, 0.0, 1.0]
        ]

        camera_path["camera_path"].append({
            "camera_to_world": camera_to_world,
            "fov": 50,
            "aspect": 800 / 600,
        })

    with open(output_file, 'w') as f:
        json.dump(camera_path, f, indent=2)

    print(f"Created camera path with {num_views} views: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Render using nerfstudio')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to nerfstudio config.yml')
    parser.add_argument('--num-views', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='nerfstudio_renders')
    parser.add_argument('--center', type=float, nargs=3,
                        default=[0.021, -0.062, 0.003],
                        help='Object center coordinates')
    parser.add_argument('--radius', type=float, default=1.0,
                        help='Camera orbit radius')

    args = parser.parse_args()

    # Create camera path
    center = np.array(args.center)
    camera_path_file = 'camera_path_orbit.json'

    print(f"Creating camera path...")
    print(f"  Center: {center}")
    print(f"  Radius: {args.radius}")
    print(f"  Views: {args.num_views}")

    create_camera_path_json(center, args.radius, args.num_views, camera_path_file)

    # Render using ns-render
    output_path = Path(args.output_dir) / 'video.mp4'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ns-render', 'camera-path',
        '--load-config', args.config,
        '--camera-path-filename', camera_path_file,
        '--output-path', str(output_path),
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)

    print(f"\nDone! Video saved to {output_path}")
    print(f"Extract frames with:")
    print(f"  ffmpeg -i {output_path} {args.output_dir}/frame_%04d.png")


if __name__ == '__main__':
    main()
