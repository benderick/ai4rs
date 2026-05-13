#!/usr/bin/env python3
"""
跨视角数据划分的分块脚本

根据crossview_splits中的图像列表，对CODrone数据集进行分块处理。

用法:
    python create_crossview_patches.py --protocol AG-30 --split train
    python create_crossview_patches.py --protocol AG-30 --split all
"""

import argparse
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Split crossview data')
    parser.add_argument('--protocol', type=str, required=True,
                        help='Crossview protocol name, e.g., AG-30, AngG-90')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to process')
    parser.add_argument('--data-root', type=str, 
                        default='data/CODrone',
                        help='CODrone data root')
    parser.add_argument('--splits-root', type=str,
                        default='data/CODrone/crossview_splits',
                        help='Crossview splits root')
    parser.add_argument('--output-root', type=str,
                        default='data/CODrone/crossview_patches',
                        help='Output root for split images')
    parser.add_argument('--size', type=int, default=1024,
                        help='Split patch size')
    parser.add_argument('--gap', type=int, default=200,
                        help='Split gap/overlap')
    parser.add_argument('--nproc', type=int, default=10,
                        help='Number of processes')
    parser.add_argument('--use-symlink', action='store_true',
                        help='Use symlink instead of copy')
    return parser.parse_args()


def prepare_split_data(protocol: str, split: str, data_root: str, 
                       splits_root: str, temp_dir: str, use_symlink: bool = True):
    """准备分块所需的数据（创建临时目录结构）
    
    Args:
        protocol: 协议名称，如 AG-30
        split: 划分类型，train/val/test
        data_root: CODrone数据根目录
        splits_root: 跨视角划分文件根目录
        temp_dir: 临时目录
        use_symlink: 是否使用软链接
        
    Returns:
        tuple: (临时图像目录, 临时标注目录, 图像数量)
    """
    # 读取图像列表
    list_file = osp.join(splits_root, protocol, f'{split}_images.txt')
    if not osp.exists(list_file):
        raise FileNotFoundError(f'Image list file not found: {list_file}')
    
    with open(list_file, 'r') as f:
        image_names = [line.strip() for line in f if line.strip()]
    
    print(f'  找到 {len(image_names)} 张图像')
    
    # 创建临时目录
    temp_img_dir = osp.join(temp_dir, 'images')
    temp_ann_dir = osp.join(temp_dir, 'annfile')
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(temp_ann_dir, exist_ok=True)
    
    # 确定源数据目录（CODrone的train/val/test）
    # 需要从所有原始split中查找图像
    source_splits = ['train', 'val', 'test']
    
    linked_count = 0
    missing_images = []
    missing_anns = []
    
    for img_name in image_names:
        # 查找图像文件
        img_found = False
        for src_split in source_splits:
            src_img_path = osp.join(data_root, src_split, 'images', img_name)
            if osp.exists(src_img_path):
                dst_img_path = osp.join(temp_img_dir, img_name)
                if not osp.exists(dst_img_path):
                    if use_symlink:
                        os.symlink(osp.abspath(src_img_path), dst_img_path)
                    else:
                        shutil.copy2(src_img_path, dst_img_path)
                img_found = True
                
                # 查找对应的标注文件
                ann_name = osp.splitext(img_name)[0] + '.txt'
                src_ann_path = osp.join(data_root, src_split, 'annfile', ann_name)
                dst_ann_path = osp.join(temp_ann_dir, ann_name)
                
                if osp.exists(src_ann_path):
                    if not osp.exists(dst_ann_path):
                        if use_symlink:
                            os.symlink(osp.abspath(src_ann_path), dst_ann_path)
                        else:
                            shutil.copy2(src_ann_path, dst_ann_path)
                    linked_count += 1
                else:
                    missing_anns.append(ann_name)
                break
        
        if not img_found:
            missing_images.append(img_name)
    
    if missing_images:
        print(f'  警告: {len(missing_images)} 张图像未找到')
        if len(missing_images) <= 5:
            for m in missing_images:
                print(f'    - {m}')
    
    if missing_anns:
        print(f'  警告: {len(missing_anns)} 个标注文件未找到')
    
    print(f'  成功链接 {linked_count} 个文件')
    
    return temp_img_dir, temp_ann_dir, linked_count


def create_split_config(img_dir: str, ann_dir: str, save_dir: str,
                        size: int, gap: int, nproc: int) -> str:
    """创建分块配置文件
    
    Returns:
        配置文件路径
    """
    config = {
        "nproc": nproc,
        "img_dirs": [img_dir],
        "ann_dirs": [ann_dir],
        "sizes": [size],
        "gaps": [gap],
        "rates": [1.0],
        "img_rate_thr": 0.6,
        "iof_thr": 0.7,
        "no_padding": False,
        "padding_value": [104, 116, 124],
        "save_dir": save_dir,
        "save_ext": ".jpg"
    }
    
    config_path = osp.join(osp.dirname(save_dir), 'split_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def run_split(config_path: str):
    """运行分块脚本"""
    script_path = 'tools/data/codrone/split/img_split.py'
    
    cmd = [sys.executable, script_path, '--base-json', config_path]
    print(f'  执行: {" ".join(cmd)}')
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    args = parse_args()
    
    # 确定要处理的split
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]
    
    print(f'=' * 60)
    print(f'跨视角数据分块')
    print(f'协议: {args.protocol}')
    print(f'分块大小: {args.size}, 间隔: {args.gap}')
    print(f'=' * 60)
    
    # 检查协议目录是否存在
    protocol_dir = osp.join(args.splits_root, args.protocol)
    if not osp.exists(protocol_dir):
        print(f'错误: 协议目录不存在: {protocol_dir}')
        available = os.listdir(args.splits_root)
        print(f'可用协议: {", ".join(available)}')
        return 1
    
    for split in splits:
        print(f'\n处理 {split} 集...')
        
        # 输出目录
        output_dir = osp.join(args.output_root, args.protocol, split)
        
        # 检查是否已存在
        if osp.exists(output_dir):
            print(f'  输出目录已存在: {output_dir}')
            print(f'  跳过此split，如需重新处理请先删除目录')
            continue
        
        # 临时目录
        temp_dir = osp.join(args.output_root, args.protocol, f'_temp_{split}')
        
        try:
            # 准备数据
            print(f'  准备数据...')
            img_dir, ann_dir, count = prepare_split_data(
                args.protocol, split, args.data_root, args.splits_root,
                temp_dir, args.use_symlink
            )
            
            if count == 0:
                print(f'  没有找到有效数据，跳过')
                continue
            
            # 创建配置文件
            config_path = create_split_config(
                img_dir, ann_dir, output_dir,
                args.size, args.gap, args.nproc
            )
            print(f'  配置文件: {config_path}')
            
            # 运行分块
            print(f'  开始分块...')
            success = run_split(config_path)
            
            if success:
                print(f'  ✓ {split} 集分块完成')
            else:
                print(f'  ✗ {split} 集分块失败')
                
        finally:
            # 清理临时目录
            if osp.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f'  清理临时目录')
    
    print(f'\n完成!')
    return 0


if __name__ == '__main__':
    sys.exit(main())
