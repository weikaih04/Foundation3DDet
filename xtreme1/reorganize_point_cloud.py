#!/usr/bin/env python3
"""
重新组织point_cloud目录结构
"""

import os
import shutil

def reorganize_point_cloud():
    """重新组织point_cloud目录结构"""
    
    point_cloud_dir = "point_cloud"
    lidar_dir = os.path.join(point_cloud_dir, "lidar_point_cloud_0")
    
    print("🔄 重新组织point_cloud目录结构...")
    
    # 检查当前目录结构
    if not os.path.exists(point_cloud_dir):
        print(f"❌ 目录不存在: {point_cloud_dir}")
        return False
    
    # 获取所有PCD文件
    pcd_files = [f for f in os.listdir(point_cloud_dir) if f.endswith('.pcd')]
    print(f"📋 找到 {len(pcd_files)} 个PCD文件")
    
    if not pcd_files:
        print("❌ 没有找到PCD文件")
        return False
    
    # 确保lidar_point_cloud_0目录存在
    os.makedirs(lidar_dir, exist_ok=True)
    
    # 移动PCD文件到子目录
    moved_count = 0
    for pcd_file in pcd_files:
        src = os.path.join(point_cloud_dir, pcd_file)
        dst = os.path.join(lidar_dir, pcd_file)
        
        try:
            shutil.move(src, dst)
            moved_count += 1
            print(f"  📄 移动: {pcd_file}")
        except Exception as e:
            print(f"  ❌ 移动失败 {pcd_file}: {e}")
    
    print(f"✅ 成功移动 {moved_count} 个文件")
    
    # 验证最终结构
    final_pcd_count = len([f for f in os.listdir(lidar_dir) if f.endswith('.pcd')])
    print(f"📊 最终PCD文件数量: {final_pcd_count}")
    
    # 显示目录结构
    print("\n📁 最终目录结构:")
    print("point_cloud/")
    print("└── lidar_point_cloud_0/")
    sample_files = [f for f in os.listdir(lidar_dir) if f.endswith('.pcd')][:3]
    for file in sample_files:
        print(f"    ├── {file}")
    if final_pcd_count > 3:
        print(f"    └── ... ({final_pcd_count-3} more files)")
    
    return True

if __name__ == "__main__":
    reorganize_point_cloud()
