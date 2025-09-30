#!/usr/bin/env python3
import os
import glob
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate point_cloud_list.yml with a base URL (e.g., http://localhost:8000)')
    parser.add_argument('--ply_dir', type=str, default=None, help='Directory containing .ply files (default: ./ply relative to this script)')
    parser.add_argument('--base_url', type=str, required=True, help='Base URL prefix, e.g., http://localhost:8000')
    parser.add_argument('--out', type=str, default=None, help='Output YAML path (default: <ply_dir>/point_cloud_list_http.yml)')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    ply_dir = Path(args.ply_dir) if args.ply_dir else (base_dir / 'ply')
    ply_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out) if args.out else (ply_dir / 'point_cloud_list_http.yml')

    # Ensure no trailing slash mess-ups
    base_url = args.base_url.rstrip('/')

    ply_files = sorted(glob.glob(str(ply_dir / '*.ply')))
    if not ply_files:
        print(f'No .ply files found in {ply_dir}. Convert first.')
        return

    with open(out_path, 'w') as f:
        for p in ply_files:
            url = f"{base_url}/{Path(p).name}"
            f.write(f'- {{\n    url: "{url}",\n  }}\n')

    print(f'Wrote {out_path} with {len(ply_files)} entries using base URL {base_url}.')

if __name__ == '__main__':
    main()
