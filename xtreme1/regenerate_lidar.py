#!/usr/bin/env python3
"""
重新生成LiDAR PCD文件并创建正确的目录结构
"""

import os
import sys
import glob
import shutil
import zipfile
import subprocess

def regenerate_pcd_files():
    """重新生成PCD文件"""
    print("🔄 重新生成PCD文件...")
    
    # 清理旧文件
    if os.path.exists('pcd_files'):
        shutil.rmtree('pcd_files')
    if os.path.exists('point_cloud'):
        shutil.rmtree('point_cloud')
    
    # 运行转换脚本
    try:
        result = subprocess.run([sys.executable, 'to_pcd.py', '.', 'pcd_files'], 
                               capture_output=True, text=True, check=True)
        print("✅ PCD文件生成成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PCD文件生成失败: {e}")
        return False

def create_lidar_structure():
    """创建LiDAR目录结构"""
    print("📁 创建目录结构...")
    
    # 创建目录
    point_cloud_dir = "point_cloud"
    lidar_dir = os.path.join(point_cloud_dir, "lidar_point_cloud_0")
    os.makedirs(lidar_dir, exist_ok=True)
    
    # 复制PCD文件
    pcd_files = glob.glob("pcd_files/*.pcd")
    if not pcd_files:
        print("❌ 没有找到PCD文件")
        return False
    
    print(f"📋 找到 {len(pcd_files)} 个PCD文件")
    
    for pcd_file in pcd_files:
        filename = os.path.basename(pcd_file)
        dst = os.path.join(lidar_dir, filename)
        shutil.copy2(pcd_file, dst)
    
    print(f"✅ 复制了 {len(pcd_files)} 个文件到 {lidar_dir}")
    return True

def create_zip_file():
    """创建zip文件"""
    print("📦 创建zip文件...")
    
    zip_filename = "point_cloud_lidar.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("point_cloud"):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, '.')
                zipf.write(file_path, arcname)
                print(f"  📄 添加: {arcname}")
    
    # 获取文件大小
    zip_size = os.path.getsize(zip_filename) / (1024*1024)  # MB
    print(f"✅ 创建完成: {zip_filename} ({zip_size:.1f} MB)")
    
    return zip_filename

def verify_structure():
    """验证目录结构"""
    print("🔍 验证目录结构...")
    
    # 检查目录结构
    if not os.path.exists("point_cloud/lidar_point_cloud_0"):
        print("❌ 目录结构不正确")
        return False
    
    pcd_count = len(glob.glob("point_cloud/lidar_point_cloud_0/*.pcd"))
    print(f"📊 PCD文件数量: {pcd_count}")
    
    # 检查zip文件
    if os.path.exists("point_cloud_lidar.zip"):
        zip_size = os.path.getsize("point_cloud_lidar.zip") / (1024*1024)
        print(f"📦 Zip文件大小: {zip_size:.1f} MB")
        return True
    else:
        print("❌ Zip文件不存在")
        return False

def main():
    print("🚀 开始重新生成LiDAR数据...")
    print("=" * 50)
    
    # 步骤1: 生成PCD文件
    if not regenerate_pcd_files():
        return
    
    # 步骤2: 创建目录结构
    if not create_lidar_structure():
        return
    
    # 步骤3: 创建zip文件
    zip_file = create_zip_file()
    if not zip_file:
        return
    
    # 步骤4: 验证结果
    if verify_structure():
        print("\n🎉 完成! 目录结构:")
        print("point_cloud/")
        print("└── lidar_point_cloud_0/")
        print("    ├── 000000000139.pcd")
        print("    ├── 000000000285.pcd")
        print("    └── ... (更多PCD文件)")
        print(f"\n📦 最终zip文件: {zip_file}")
        print("✅ 现在可以上传到Xtreme1了!")
    else:
        print("❌ 验证失败")

if __name__ == "__main__":
    main()
