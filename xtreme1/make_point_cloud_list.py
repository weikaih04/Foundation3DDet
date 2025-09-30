#!/usr/bin/env python3
import os
import glob
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent
    ply_dir = base_dir / 'ply'
    ply_dir.mkdir(parents=True, exist_ok=True)
    out_path = ply_dir / 'point_cloud_list.yml'

    ply_files = sorted(glob.glob(str(ply_dir / '*.ply')))
    if not ply_files:
        print(f'No .ply files found in {ply_dir}. Run convert_depth_to_ply.py first.')
        return

    with open(out_path, 'w') as f:
        for p in ply_files:
            url = Path(p).resolve().as_uri()  # file:// URL
            f.write(f'- {{\n    url: "{url}",\n  }}\n')

    print(f'Wrote {out_path} with {len(ply_files)} entries.')

if __name__ == '__main__':
    main()
