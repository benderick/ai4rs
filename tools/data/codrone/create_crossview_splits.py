#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨视角数据划分脚本 - 针对原始CODrone数据集

根据论文中定义的跨视角评估协议，分析和创建不同的训练/测试划分。
本脚本针对原始未分块的CODrone数据集，统计真实的图像数量。

协议说明：
1. 高度泛化协议 (Altitude Generalization, AG)
   - AG-30: 在60m+100m上训练，在30m上测试
   - AG-60: 在30m+100m上训练，在60m上测试
   - AG-100: 在30m+60m上训练，在100m上测试

2. 角度泛化协议 (Angle Generalization, AngG)
   - AngG-30: 在90°上训练，在30°上测试
   - AngG-90: 在30°上训练，在90°上测试

3. 联合泛化协议 (Joint Generalization, JG)
   - JG-Extreme: 在(30m,90°)上训练，在(100m,30°)上测试
   - JG-LOO: Leave-One-Out，在5种视角组合上训练，在剩余1种上测试

使用方法：
    python tools/data/codrone/create_crossview_splits.py \
        --data-root data/CODrone \
        --output-root data/crossview_splits \
        --protocol all \
        --save-lists
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import json


def parse_viewpoint_from_filename(filename: str) -> Tuple[str, str, str]:
    """
    从文件名中解析视角信息（高度、角度、时间）
    
    文件命名格式示例：
    - chenhuachengpark_day_30m_30c_frame_750.jpg
    - day_canton_road_30m_90c_frame_200.jpg
    - night_xxx_60m_30c_frame_xxx.jpg
    
    Args:
        filename: 图像文件名
        
    Returns:
        (altitude, angle, time): 高度（30m/60m/100m）、角度（30c/90c）、时间（day/night）
    """
    # 匹配高度模式：30m, 60m, 100m
    altitude_match = re.search(r'_(30m|60m|100m)_', filename)
    # 匹配角度模式：30c, 90c（c表示camera angle）
    angle_match = re.search(r'_(30c|90c)', filename)
    # 匹配时间模式：day, night
    time_match = re.search(r'(day|night)', filename, re.IGNORECASE)
    
    altitude = altitude_match.group(1) if altitude_match else None
    angle = angle_match.group(1) if angle_match else None
    time = time_match.group(1).lower() if time_match else None
    
    return altitude, angle, time


def get_all_images_by_viewpoint(data_root: str, split: str = 'train') -> Dict:
    """
    按视角组合分组获取所有图像，并统计标注信息
    
    Args:
        data_root: 数据集根目录
        split: 数据划分（train/val/test）
        
    Returns:
        Dict: 包含图像分组和统计信息
    """
    images_dir = Path(data_root) / split / 'images'
    annfiles_dir = Path(data_root) / split / 'annfile'
    
    viewpoint_images = defaultdict(list)
    viewpoint_instances = defaultdict(int)
    time_images = defaultdict(list)
    unknown_images = []
    
    total_instances = 0
    
    for img_file in sorted(images_dir.iterdir()):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            altitude, angle, time = parse_viewpoint_from_filename(img_file.name)
            
            if altitude and angle:
                viewpoint_images[(altitude, angle)].append(img_file.name)
                
                # 统计该图像的标注实例数
                ann_file = annfiles_dir / (img_file.stem + '.txt')
                if ann_file.exists():
                    with open(ann_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            # DOTA格式：每行一个目标
                            num_instances = len(content.split('\n'))
                            viewpoint_instances[(altitude, angle)] += num_instances
                            total_instances += num_instances
                
                # 按时间分组
                if time:
                    time_images[time].append(img_file.name)
            else:
                unknown_images.append(img_file.name)
    
    if unknown_images:
        print(f"  警告: {len(unknown_images)} 个图像无法解析视角信息")
        if len(unknown_images) <= 5:
            for img in unknown_images:
                print(f"    - {img}")
    
    return {
        'viewpoint_images': dict(viewpoint_images),
        'viewpoint_instances': dict(viewpoint_instances),
        'time_images': dict(time_images),
        'total_instances': total_instances,
        'unknown_images': unknown_images
    }


def print_viewpoint_statistics(data: Dict, title: str = ""):
    """打印详细的视角统计信息"""
    viewpoint_images = data['viewpoint_images']
    viewpoint_instances = data['viewpoint_instances']
    time_images = data['time_images']
    
    print(f"\n{'='*60}")
    print(f"视角统计 {title}")
    print(f"{'='*60}")
    
    # 按视角组合统计
    print("\n【按视角组合统计】")
    print(f"{'高度':<8} {'角度':<8} {'图像数':>10} {'目标数':>12} {'平均目标/图':>12}")
    print("-" * 50)
    
    total_images = 0
    total_instances = 0
    
    # 按高度和角度排序
    for (alt, ang) in sorted(viewpoint_images.keys()):
        images = viewpoint_images[(alt, ang)]
        instances = viewpoint_instances.get((alt, ang), 0)
        count = len(images)
        avg = instances / count if count > 0 else 0
        total_images += count
        total_instances += instances
        print(f"{alt:<8} {ang:<8} {count:>10,} {instances:>12,} {avg:>12.1f}")
    
    print("-" * 50)
    avg_total = total_instances / total_images if total_images > 0 else 0
    print(f"{'总计':<17} {total_images:>10,} {total_instances:>12,} {avg_total:>12.1f}")
    
    # 按时间统计
    if time_images:
        print("\n【按光照条件统计】")
        for time_type, images in sorted(time_images.items()):
            print(f"  {time_type}: {len(images):,} 张图像")
    
    print()


def create_split_info(protocol_name: str, 
                      train_viewpoints: Set[Tuple[str, str]], 
                      test_viewpoints: Set[Tuple[str, str]],
                      train_data: Dict,
                      val_data: Dict = None,
                      test_data: Dict = None) -> Dict:
    """
    创建协议的划分信息（不创建实际文件，只生成统计）
    
    对于跨视角评估：
    - 训练集：从原始train中选取训练视角的数据
    - 验证集：从原始val中选取测试视角的数据（用于调参监控）
    - 测试集：从原始test中选取测试视角的数据
    
    Args:
        protocol_name: 协议名称
        train_viewpoints: 训练集包含的视角组合
        test_viewpoints: 测试/验证集包含的视角组合
        train_data: 原始训练集数据
        val_data: 原始验证集数据
        test_data: 原始测试集数据
        
    Returns:
        划分信息字典
    """
    # 统计训练集（从原始train中选取训练视角）
    train_images = 0
    train_instances = 0
    for vp, images in train_data['viewpoint_images'].items():
        if vp in train_viewpoints:
            train_images += len(images)
            train_instances += train_data['viewpoint_instances'].get(vp, 0)
    
    # 统计验证集（从原始val中选取测试视角）
    val_images = 0
    val_instances = 0
    if val_data:
        for vp, images in val_data['viewpoint_images'].items():
            if vp in test_viewpoints:
                val_images += len(images)
                val_instances += val_data['viewpoint_instances'].get(vp, 0)
    
    # 统计测试集（从原始test中选取测试视角）
    test_images = 0
    test_instances = 0
    if test_data:
        for vp, images in test_data['viewpoint_images'].items():
            if vp in test_viewpoints:
                test_images += len(images)
                test_instances += test_data['viewpoint_instances'].get(vp, 0)
    
    return {
        'protocol': protocol_name,
        'train_viewpoints': sorted([list(vp) for vp in train_viewpoints]),
        'test_viewpoints': sorted([list(vp) for vp in test_viewpoints]),
        'train_images': train_images,
        'train_instances': train_instances,
        'val_images': val_images,
        'val_instances': val_instances,
        'test_images': test_images,
        'test_instances': test_instances,
    }


def create_altitude_generalization_splits(train_data: Dict, val_data: Dict, test_data: Dict, 
                                          all_viewpoints: Set) -> List[Dict]:
    """创建高度泛化协议划分 (AG)"""
    
    angles = ['30c', '90c']
    
    protocols = {
        'AG-30': {'train_alt': ['60m', '100m'], 'test_alt': ['30m']},
        'AG-60': {'train_alt': ['30m', '100m'], 'test_alt': ['60m']},
        'AG-100': {'train_alt': ['30m', '60m'], 'test_alt': ['100m']},
    }
    
    results = []
    for protocol_name, config in protocols.items():
        train_vps = {(alt, ang) for alt in config['train_alt'] for ang in angles 
                     if (alt, ang) in all_viewpoints}
        test_vps = {(alt, ang) for alt in config['test_alt'] for ang in angles 
                    if (alt, ang) in all_viewpoints}
        
        info = create_split_info(protocol_name, train_vps, test_vps, 
                                 train_data, val_data, test_data)
        results.append(info)
    
    return results


def create_angle_generalization_splits(train_data: Dict, val_data: Dict, test_data: Dict,
                                       all_viewpoints: Set) -> List[Dict]:
    """创建角度泛化协议划分 (AngG)"""
    
    altitudes = ['30m', '60m', '100m']
    
    protocols = {
        'AngG-30': {'train_ang': ['90c'], 'test_ang': ['30c']},
        'AngG-90': {'train_ang': ['30c'], 'test_ang': ['90c']},
    }
    
    results = []
    for protocol_name, config in protocols.items():
        train_vps = {(alt, ang) for alt in altitudes for ang in config['train_ang'] 
                     if (alt, ang) in all_viewpoints}
        test_vps = {(alt, ang) for alt in altitudes for ang in config['test_ang'] 
                    if (alt, ang) in all_viewpoints}
        
        info = create_split_info(protocol_name, train_vps, test_vps,
                                 train_data, val_data, test_data)
        results.append(info)
    
    return results


def create_joint_generalization_splits(train_data: Dict, val_data: Dict, test_data: Dict,
                                       all_viewpoints: Set) -> List[Dict]:
    """创建联合泛化协议划分 (JG)"""
    
    results = []
    
    # JG-Extreme: 近景视角训练，远景视角测试
    train_vps = {('30m', '90c')} & all_viewpoints
    test_vps = {('100m', '30c')} & all_viewpoints
    
    if train_vps and test_vps:
        info = create_split_info('JG-Extreme', train_vps, test_vps,
                                 train_data, val_data, test_data)
        results.append(info)
    
    # JG-LOO: Leave-One-Out
    for test_vp in sorted(all_viewpoints):
        train_vps = all_viewpoints - {test_vp}
        test_vps = {test_vp}
        
        protocol_name = f"JG-LOO-{test_vp[0]}-{test_vp[1]}"
        info = create_split_info(protocol_name, train_vps, test_vps,
                                 train_data, val_data, test_data)
        results.append(info)
    
    return results

def create_std_split(train_data: Dict, val_data: Dict, test_data: Dict) -> Dict:
    """创建标准评估协议划分 (STD)"""
    all_viewpoints = set(train_data['viewpoint_images'].keys()) | set(val_data['viewpoint_images'].keys()) | set(test_data['viewpoint_images'].keys())
    return create_split_info('STD', all_viewpoints, all_viewpoints, train_data, val_data, test_data)

def print_protocol_summary(protocols: List[Dict], title: str):
    """打印协议摘要表格"""
    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")
    print(f"{'协议':<20} {'训练视角':<20} {'测试视角':<12} {'训练':>8} {'验证':>8} {'测试':>8}")
    print("-" * 100)
    
    for p in protocols:
        train_vps = ', '.join([f"{v[0]}/{v[1]}" for v in p['train_viewpoints']])
        test_vps = ', '.join([f"{v[0]}/{v[1]}" for v in p['test_viewpoints']])
        
        # 截断过长的字符串
        if len(train_vps) > 18:
            train_vps = train_vps[:15] + "..."
        if len(test_vps) > 10:
            test_vps = test_vps[:7] + "..."
            
        print(f"{p['protocol']:<20} {train_vps:<20} {test_vps:<12} "
              f"{p['train_images']:>8,} {p['val_images']:>8,} {p['test_images']:>8,}")


def save_split_lists(output_root: Path, protocols: List[Dict], 
                     train_data: Dict, val_data: Dict, test_data: Dict):
    """
    保存每个协议的图像文件列表（用于后续生成分块数据）
    """
    for p in protocols:
        protocol_dir = output_root / p['protocol']
        protocol_dir.mkdir(parents=True, exist_ok=True)
        
        train_vps = {tuple(vp) for vp in p['train_viewpoints']}
        test_vps = {tuple(vp) for vp in p['test_viewpoints']}
        
        # 收集训练图像（从原始train中选取训练视角）
        train_images = []
        for vp, images in train_data['viewpoint_images'].items():
            if vp in train_vps:
                train_images.extend(images)
        
        # 收集验证图像（从原始val中选取测试视角）
        val_images = []
        for vp, images in val_data['viewpoint_images'].items():
            if vp in test_vps:
                val_images.extend(images)
        
        # 收集测试图像（从原始test中选取测试视角）
        test_images = []
        for vp, images in test_data['viewpoint_images'].items():
            if vp in test_vps:
                test_images.extend(images)
        
        # 保存图像列表
        with open(protocol_dir / 'train_images.txt', 'w') as f:
            f.write('\n'.join(sorted(train_images)))
        
        with open(protocol_dir / 'val_images.txt', 'w') as f:
            f.write('\n'.join(sorted(val_images)))
        
        with open(protocol_dir / 'test_images.txt', 'w') as f:
            f.write('\n'.join(sorted(test_images)))
        
        # 保存完整信息
        with open(protocol_dir / 'split_info.json', 'w') as f:
            json.dump(p, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='创建跨视角评估数据划分（原始CODrone数据集）')
    parser.add_argument('--data-root', type=str, default='data/CODrone',
                        help='原始CODrone数据集根目录')
    parser.add_argument('--output-root', type=str, default='data/CODrone/crossview_splits',
                        help='输出目录')
    parser.add_argument('--protocol', type=str, default='all',
                        choices=['all', 'AG', 'AngG', 'JG', 'STD'],
                        help='要创建的协议类型')
    parser.add_argument('--save-lists', action='store_true',
                        help='保存图像文件列表')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    
    print("="*70)
    print("CODrone 跨视角数据划分分析脚本")
    print("="*70)
    print(f"数据源: {data_root}")
    print(f"输出目录: {output_root}")
    print(f"协议类型: {args.protocol}")
    
    # 分析各数据划分
    all_data = {}
    for split in ['train', 'val', 'test']:
        split_dir = data_root / split / 'images'
        if split_dir.exists():
            print(f"\n正在分析 {split} 集...")
            all_data[split] = get_all_images_by_viewpoint(data_root, split)
            print_viewpoint_statistics(all_data[split], f"({split}集)")
    
    # 检查必要的数据集
    if 'train' not in all_data:
        print("错误: 未找到训练集！")
        return
    if 'val' not in all_data:
        print("错误: 未找到验证集！")
        return
    if 'test' not in all_data:
        print("错误: 未找到测试集！")
        return
    
    train_data = all_data['train']
    val_data = all_data['val']
    test_data = all_data['test']
    all_viewpoints = set(train_data['viewpoint_images'].keys())
    
    print(f"\n检测到的视角组合: {sorted(all_viewpoints)}")
    
    # 创建输出目录
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 收集所有协议
    all_protocols = []
    
    if args.protocol in ['all', 'AG']:
        ag_protocols = create_altitude_generalization_splits(train_data, val_data, test_data, all_viewpoints)
        all_protocols.extend(ag_protocols)
        print_protocol_summary(ag_protocols, "高度泛化协议 (Altitude Generalization)")
    
    if args.protocol in ['all', 'AngG']:
        ang_protocols = create_angle_generalization_splits(train_data, val_data, test_data, all_viewpoints)
        all_protocols.extend(ang_protocols)
        print_protocol_summary(ang_protocols, "角度泛化协议 (Angle Generalization)")
    
    if args.protocol in ['all', 'JG']:
        jg_protocols = create_joint_generalization_splits(train_data, val_data, test_data, all_viewpoints)
        # 分开打印JG-Extreme和JG-LOO
        jg_extreme = [p for p in jg_protocols if 'LOO' not in p['protocol']]
        jg_loo = [p for p in jg_protocols if 'LOO' in p['protocol']]
        
        if jg_extreme:
            print_protocol_summary(jg_extreme, "联合泛化协议 - 极端测试 (JG-Extreme)")
        if jg_loo:
            print_protocol_summary(jg_loo, "联合泛化协议 - 留一法 (JG-LOO)")
        
        all_protocols.extend(jg_protocols)
    
    if args.protocol in ['all', 'STD']:
        "STD协议即原始CODrone划分，训练集包含所有视角，测试集包含所有视角"
        std_protocol = create_std_split(train_data, val_data, test_data)
        print_protocol_summary([std_protocol], "标准评估协议 (STD)")
        all_protocols.append(std_protocol)
        
    
    # 保存图像列表
    if args.save_lists:
        print(f"\n保存图像列表到: {output_root}")
        save_split_lists(output_root, all_protocols, train_data, val_data, test_data)
    
    # 打印总体统计
    print("\n" + "="*70)
    print("数据集总体统计")
    print("="*70)
    
    total_images = 0
    total_instances = 0
    for split, data in all_data.items():
        images = sum(len(imgs) for imgs in data['viewpoint_images'].values())
        instances = data['total_instances']
        total_images += images
        total_instances += instances
        print(f"  {split}: {images:,} 张图像, {instances:,} 个目标实例")
    
    print(f"  {'─'*40}")
    print(f"  总计: {total_images:,} 张图像, {total_instances:,} 个目标实例")
    
    # 生成论文用的LaTeX表格数据
    print("\n" + "="*70)
    print("论文用数据（可复制到LaTeX）")
    print("="*70)
    
    print("\n% 数据集统计表格数据")
    for (alt, ang) in sorted(train_data['viewpoint_images'].keys()):
        images = len(train_data['viewpoint_images'][(alt, ang)])
        instances = train_data['viewpoint_instances'].get((alt, ang), 0)
        alt_num = alt.replace('m', '')
        ang_num = ang.replace('c', '°')
        print(f"        {alt_num}m & {ang_num} & {images:,} & {instances:,} \\\\")
    
    print("\n完成！")


if __name__ == '__main__':
    main()
