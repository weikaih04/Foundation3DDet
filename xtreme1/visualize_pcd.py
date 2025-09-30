#!/usr/bin/env python3
"""
PCD文件可视化脚本
用于检查PCD文件是否正确生成
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import argparse

def read_pcd_header(filename):
    """读取PCD文件头部信息"""
    with open(filename, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline()
            if line.startswith(b'DATA'):
                header_lines.append(line.decode('ascii').strip())
                break
            header_lines.append(line.decode('ascii').strip())
    return header_lines

def read_pcd_binary(filename):
    """读取二进制PCD文件"""
    # 读取头部信息
    with open(filename, 'rb') as f:
        header_lines = []
        data_start = 0
        while True:
            line = f.readline()
            if line.startswith(b'DATA'):
                header_lines.append(line.decode('ascii').strip())
                data_start = f.tell()
                break
            header_lines.append(line.decode('ascii').strip())
    
    # 解析头部信息
    points = 0
    fields = []
    for line in header_lines:
        if line.startswith('POINTS'):
            points = int(line.split()[1])
        elif line.startswith('FIELDS'):
            fields = line.split()[1:]
    
    print(f"PCD文件信息:")
    print(f"  点数: {points}")
    print(f"  字段: {fields}")
    print(f"  头部行数: {len(header_lines)}")
    
    # 读取二进制数据
    with open(filename, 'rb') as f:
        f.seek(data_start)
        # 每个点7个float32值 (index, x, y, z, i, t, d)
        data = np.frombuffer(f.read(points * 7 * 4), dtype=np.float32)
        data = data.reshape(points, 7)
    
    return data, header_lines

def visualize_point_cloud(points, title="Point Cloud", max_points=10000):
    """可视化点云"""
    # 如果点太多，随机采样
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        print(f"随机采样到 {max_points} 个点进行可视化")
    
    # 提取坐标
    x = points[:, 1]  # x
    y = points[:, 2]  # y  
    z = points[:, 3]  # z
    
    # 创建3D图
    fig = plt.figure(figsize=(15, 5))
    
    # 3D散点图
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'{title} - 3D View')
    plt.colorbar(scatter, ax=ax1, shrink=0.5)
    
    # XY投影
    ax2 = fig.add_subplot(132)
    ax2.scatter(x, y, c=z, cmap='viridis', s=1, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.set_aspect('equal')
    
    # XZ投影
    ax3 = fig.add_subplot(133)
    ax3.scatter(x, z, c=y, cmap='viridis', s=1, alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection')
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def analyze_point_cloud(points):
    """分析点云数据"""
    print(f"\n点云分析:")
    print(f"  总点数: {len(points)}")
    
    # 坐标统计
    x, y, z = points[:, 1], points[:, 2], points[:, 3]
    print(f"  X范围: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Y范围: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Z范围: [{z.min():.3f}, {z.max():.3f}]")
    
    # 距离统计
    distances = np.sqrt(x**2 + y**2 + z**2)
    print(f"  距离范围: [{distances.min():.3f}, {distances.max():.3f}]")
    print(f"  平均距离: {distances.mean():.3f}")
    
    # 检查异常值
    valid_points = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    print(f"  有效点数: {valid_points.sum()}")
    print(f"  无效点数: {(~valid_points).sum()}")
    
    return {
        'x_range': (x.min(), x.max()),
        'y_range': (y.min(), y.max()),
        'z_range': (z.min(), z.max()),
        'distance_range': (distances.min(), distances.max()),
        'valid_points': valid_points.sum(),
        'total_points': len(points)
    }

def main():
    parser = argparse.ArgumentParser(description='可视化PCD文件')
    parser.add_argument('--pcd_file', type=str, help='单个PCD文件路径')
    parser.add_argument('--pcd_dir', type=str, default='pcd_files', help='PCD文件目录')
    parser.add_argument('--max_points', type=int, default=10000, help='最大可视化点数')
    parser.add_argument('--save_plots', action='store_true', help='保存图片')
    
    args = parser.parse_args()
    
    if args.pcd_file:
        # 可视化单个文件
        if not os.path.exists(args.pcd_file):
            print(f"文件不存在: {args.pcd_file}")
            return
        
        print(f"正在分析: {args.pcd_file}")
        points, header = read_pcd_binary(args.pcd_file)
        
        # 打印头部信息
        print("\nPCD头部信息:")
        for line in header:
            print(f"  {line}")
        
        # 分析点云
        stats = analyze_point_cloud(points)
        
        # 可视化
        fig = visualize_point_cloud(points, os.path.basename(args.pcd_file), args.max_points)
        
        if args.save_plots:
            output_name = os.path.splitext(os.path.basename(args.pcd_file))[0] + '_visualization.png'
            fig.savefig(output_name, dpi=150, bbox_inches='tight')
            print(f"图片已保存: {output_name}")
        
        plt.show()
        
    else:
        # 批量分析目录中的所有PCD文件
        pcd_files = glob.glob(os.path.join(args.pcd_dir, "*.pcd"))
        if not pcd_files:
            print(f"在目录 {args.pcd_dir} 中没有找到PCD文件")
            return
        
        print(f"找到 {len(pcd_files)} 个PCD文件")
        
        # 分析每个文件
        for i, pcd_file in enumerate(sorted(pcd_files)[:5]):  # 只分析前5个文件
            print(f"\n{'='*60}")
            print(f"分析文件 {i+1}/5: {os.path.basename(pcd_file)}")
            print(f"{'='*60}")
            
            try:
                points, header = read_pcd_binary(pcd_file)
                stats = analyze_point_cloud(points)
                
                # 可视化
                fig = visualize_point_cloud(points, os.path.basename(pcd_file), args.max_points)
                
                if args.save_plots:
                    output_name = f"visualization_{os.path.splitext(os.path.basename(pcd_file))[0]}.png"
                    fig.savefig(output_name, dpi=150, bbox_inches='tight')
                    print(f"图片已保存: {output_name}")
                
                plt.show()
                
            except Exception as e:
                print(f"处理文件 {pcd_file} 时出错: {e}")
                continue

if __name__ == "__main__":
    main()
