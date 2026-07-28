#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 20:47:58 2026

@author: ubuntu
"""

import sys
from modules import separate_by_repeater, process_frb_data, quick_summary, list_saved_files, load_and_display
from modules import download_canfar_file
import argparse
import numpy as np

#得到6类数据集合：1. chimefrbcat2_duplicates.npy代表存在具有重复tns_name的数据（864个），
#这些数据的tns_name重复数与其sub_num即峰数相同；
#2. chimefrbcat2_first_duplicates.npy代表第一个重复tns_name的数据（340个）；
#3. chimefrbcat2_unique.npy代表第一个重复tns_name的数据（340个）➕唯一tns_name数据（4199个），
#组成了catalog2中所有的数据集合（4539个）；
#4. chimefrbcat2_unique_first_repeaters.npy代表重复暴的第一个数据（83个）,
#有一些重复暴的tns_name名字与repeater_name不同，即表示不为catalog2中首次发现的重复暴；
#5. chimefrbcat2_unique_non_repeater.npy代表所有非重复暴（3558个）；
#6. chimefrbcat2_unique_repeater.npy代表所有重复暴（981个）。
def data_spearate():
    """主程序函数"""
    # 原始数据文件路径
    original_file = 'FRB_data/CHIME_cat2_frb/chimefrbcat2.npy'
    
    print("=" * 60)
    print("FRB数据处理程序")
    print("=" * 60)
    
    # 快速检查原始数据
    print("\n1. 原始数据检查:")
    quick_summary(original_file)
    print("\n" + "=" * 60)
    
    # 处理tns_name数据
    print("\n2. 处理tns_name数据:")
    unique_data, duplicates_data, first_duplicates_data = process_frb_data(original_file)
    print("\n" + "=" * 60)
    
    # 对原始数据直接进行repeater_name分离
    print("\n3. 对4539个全样本数据进行repeater_name分离:")
    repeater_data, non_repeater_data = separate_by_repeater('FRB_data/CHIME_cat2_frb/chimefrbcat2_unique.npy')
    
    # 显示重复暴数据
    print("\n重复暴数据:")
    load_and_display('FRB_data/CHIME_cat2_frb/chimefrbcat2_unique_first_repeaters.npy')
    print("\n" + "=" * 60)
    
    # 列出所有保存的文件
    print("\n4. 保存的文件列表:")
    list_saved_files(original_file)
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    
data_spearate()


#在catalog2网站https://www.canfar.net/storage/list/AstroDataCitationDOI/CISTI.CANFAR/25.0066/data下载对应文件
def data_load():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description='批量下载CANFAR天文数据文件')
    parser.add_argument('--npy-file', type=str, 
                       default='FRB_data/CHIME_cat2_frb/chimefrbcat2_first_duplicates.npy',
                       help='包含tns_name列表的npy文件路径')
    #动态谱图pdf保存路径
    parser.add_argument('--output', '-o', type=str, default='Figures/canfar_downloads', 
                       help='保存文件的文件夹路径')
    
    #动态谱hdf5数据保存路径
    #parser.add_argument('--output', '-o', type=str, default='FRB_data/canfar_downloads', 
    #                   help='保存文件的文件夹路径')
    parser.add_argument('--benchmark', '-b', action='store_true',
                       help='运行基准测试比较不同下载方法')
    parser.add_argument('--method', '-m', type=str, choices=['urllib', 'wget', 'auto'],
                       default='auto', help='指定下载方法')
    
    args = parser.parse_args()
    
    # 读取npy文件中的tns_name列表
    try:
        data = np.load(args.npy_file, allow_pickle=True)
        
        # 检查数据结构并提取tns_name
        tns_names = []
        if isinstance(data, np.ndarray):
            # 如果是结构化数组或元组数组
            if data.dtype.names is not None:
                # 结构化数组 - 尝试获取tns_name字段
                if 'tns_name' in data.dtype.names:
                    tns_names = [item['tns_name'] for item in data]
                elif 'TNS_name' in data.dtype.names:
                    tns_names = [item['TNS_name'] for item in data]
                else:
                    # 使用第一个字段
                    tns_names = [item[0] for item in data]
            elif len(data.shape) > 0:
                # 简单数组
                for item in data:
                    if isinstance(item, (tuple, list, np.ndarray)) and len(item) > 0:
                        # 元组或列表 - 取第一个元素
                        tns_names.append(str(item[0]))
                    elif isinstance(item, str):
                        # 直接是字符串
                        tns_names.append(item)
                    else:
                        # 其他类型转换为字符串
                        tns_names.append(str(item))
        else:
            # 如果不是数组，直接作为列表处理
            tns_names = [str(data)] if data else []
        
        # 过滤掉空值
        tns_names = [name.strip() for name in tns_names if name and str(name).strip()]
        
        print(f"从 {args.npy_file} 读取到 {len(tns_names)} 个有效的tns_name")
        if tns_names:
            print(f"前5个tns_name: {tns_names[:5]}")
        
    except Exception as e:
        print(f"读取npy文件失败: {e}")
        sys.exit(1)
    
    if not tns_names:
        print("没有找到有效的tns_name，退出")
        sys.exit(1)
    
    # 下载成功的计数器
    success_count = 0
    total_count = len(tns_names)
    
    # 遍历所有tns_name并下载对应的文件
    for i, tns_name in enumerate(tns_names, 1):
        # 确保tns_name是字符串
        tns_name_str = str(tns_name).strip()
        if not tns_name_str:
            continue
        
        # 构造URL
        #动态谱pdf图网站
        url = f"https://www.canfar.net/storage/vault/file/AstroDataCitationDOI/CISTI.CANFAR/25.0066/data/dynamic_spectra/plots/data/{tns_name_str}_stokesi_dynamic_spectrum_data.pdf"
        
        #动态谱hdf5数据网站
        #url = f"https://www.canfar.net/storage/vault/file/AstroDataCitationDOI/CISTI.CANFAR/25.0066/data/dynamic_spectra/data/{tns_name_str}_stokesi_dynamic_spectrum.h5"
        
        print(f"正在下载 {tns_name_str} ({i}/{total_count})...")
        
        # 运行下载
        success = download_canfar_file(
            url=url,
            save_folder=args.output,
            benchmark=args.benchmark,
            method=None if args.method == 'auto' else args.method,
            auto_select=True if args.method == 'auto' else False
        )
        
        if success:
            success_count += 1
            print(f"✓ {tns_name_str} 下载成功")
        else:
            print(f"✗ {tns_name_str} 下载失败或文件不存在，跳过")
    
    # 输出统计信息
    print(f"\n下载完成：{success_count}/{total_count} 个文件下载成功")
    
    # 根据是否有成功下载的文件退出
    sys.exit(0 if success_count > 0 else 1)



