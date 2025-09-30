#!/usr/bin/env python3
"""
创建符合Xtreme1要求的LiDAR目录结构
"""

import os
import shutil
import zipfile

def create_lidar_structure():
    """创建正确的LiDAR目录结构"""
    
    # 创建目录结构
    point_cloud_dir = "point_cloud"
    lidar_dir = os.path.join(point_cloud_dir, "lidar_point_cloud_0")
    
    # 删除旧目录（如果存在）
    if os.path.exists(point_cloud_dir):
        shutil.rmtree(point_cloud_dir)
    
    # 创建新目录
    os.makedirs(lidar_dir, exist_ok=True)
    print(f"创建目录: {lidar_dir}")
    
    # 复制PCD文件
    pcd_source_dir = "pcd_files"
    if os.path.exists(pcd_source_dir):
        pcd_files = [f for f in os.listdir(pcd_source_dir) if f.endswith('.pcd')]
        print(f"找到 {len(pcd_files)} 个PCD文件")
        
        for pcd_file in pcd_files:
            src = os.path.join(pcd_source_dir, pcd_file)
            dst = os.path.join(lidar_dir, pcd_file)
            shutil.copy2(src, dst)
            print(f"复制: {pcd_file}")
        
        print(f"\n目录结构:")
        print(f"point_cloud/")
        print(f"└── lidar_point_cloud_0/")
        for pcd_file in sorted(pcd_files)[:5]:  # 显示前5个文件
            print(f"    ├── {pcd_file}")
        if len(pcd_files) > 5:
            print(f"    └── ... ({len(pcd_files)-5} more files)")
        
        # 创建zip文件
        zip_filename = "point_cloud_lidar.zip"
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(point_cloud_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, '.')
                    zipf.write(file_path, arcname)
                    print(f"添加到zip: {arcname}")
        
        print(f"\n✅ 创建完成!")
        print(f"📁 目录结构: {point_cloud_dir}/")
        print(f"📦 Zip文件: {zip_filename}")
        print(f"📊 文件数量: {len(pcd_files)}")
        
        # 显示zip文件大小
        zip_size = os.path.getsize(zip_filename) / (1024*1024)  # MB
        print(f"📏 Zip文件大小: {zip_size:.1f} MB")
        
    else:
        print(f"错误: 找不到源目录 {pcd_source_dir}")

if __name__ == "__main__":
    create_lidar_structure()
