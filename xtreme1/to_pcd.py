#!/usr/bin/env python3
import os
import sys
import glob
import struct
import numpy as np
from tqdm import tqdm

def depth_to_point_cloud(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert a depth map to an Nx7 point cloud using camera intrinsics.
    Returns array with columns: [index, x, y, z, i, t, d]
    """
    assert depth.ndim == 2, "depth must be HxW"
    assert K.shape == (3, 3), "intrinsics must be 3x3"

    H, W = depth.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Create a grid of pixel coordinates
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v, indexing='xy')

    Z = depth.astype(np.float32)
    # Mask invalid/zero depths
    mask = np.isfinite(Z) & (Z > 0)
    
    if not np.any(mask):
        return np.zeros((0, 7), dtype=np.float32)

    uu = uu[mask]
    vv = vv[mask]
    Z = Z[mask]

    # Convert to 3D points
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    # Create structured array with all required fields
    num_points = len(Z)
    points = np.zeros((num_points, 7), dtype=np.float32)
    
    # Fill in the data
    points[:, 0] = np.arange(num_points)  # index
    points[:, 1] = X                     # x
    points[:, 2] = Y                     # y
    points[:, 3] = Z                     # z
    points[:, 4] = 1.0                   # intensity (i)
    points[:, 5] = 0.0                   # timestamp (t)
    points[:, 6] = 1.0                   # additional data (d)
    
    return points

def write_pcd(points: np.ndarray, filename: str):
    """Write point cloud to PCD file in binary format"""
    num_points = points.shape[0]
    
    # Create PCD header
    header = f"""VERSION 0.7
FIELDS index x y z i t d
SIZE 4 4 4 4 4 4 4
TYPE F F F F F F F
COUNT 1 1 1 1 1 1 1
WIDTH 1
HEIGHT 1
VIEWPOINT 0.0 0.0 0.0 1.0 0.0 0.0 0.0
POINTS {num_points}
DATA binary
"""
    # Write header
    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        # Write binary data
        points.astype(np.float32).tofile(f)

def process_all(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    depth_files = sorted(glob.glob(os.path.join(input_dir, "*_depth.npy")))
    
    if not depth_files:
        print(f"[WARN] No depth npy files found in {input_dir}")
        return

    for dpath in tqdm(depth_files, desc="Processing depth maps"):
        base = os.path.basename(dpath)
        stem = base.replace("_depth.npy", "")
        kpath = os.path.join(input_dir, f"{stem}_intrinsics.npy")
        
        if not os.path.isfile(kpath):
            print(f"[SKIP] Missing intrinsics for {base}")
            continue

        try:
            # Load depth and intrinsics
            depth = np.load(dpath)
            K = np.load(kpath)
            
            # Handle different depth map formats
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth.ndim != 2:
                print(f"[SKIP] Depth array for {base} must be HxW; got shape {depth.shape}")
                continue
                
            # Convert to point cloud
            points = depth_to_point_cloud(depth, K)
            if points.shape[0] == 0:
                print(f"[SKIP] No valid points for {base}")
                continue
                
            # Save as PCD
            out_path = os.path.join(output_dir, f"{stem}.pcd")
            write_pcd(points, out_path)
            print(f"[OK] Wrote PCD: {out_path} (points: {points.shape[0]})")
            
        except Exception as e:
            print(f"[ERROR] Failed processing {base}: {str(e)}")
            continue

def main():
    # Default paths relative to this script
    in_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(in_dir, "point_cloud")

    # Handle command line arguments
    if len(sys.argv) >= 2:
        in_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        out_dir = sys.argv[2]

    process_all(in_dir, out_dir)
    print("\nConversion complete!")

if __name__ == "__main__":
    main()
