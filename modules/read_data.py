#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 20:31:34 2026

@author: ubuntu
"""

import numpy as np
import os
import h5py
import json

def separate_by_repeater(input_filename, output_basename=None):
    """
    根据repeater_name字段分离数据，并保存去重后的唯一repeater_name数据
    
    Parameters:
    -----------
    input_filename : str
        输入数据文件路径
    output_basename : str, optional
        输出文件基本名，如果为None则使用输入文件的基本名
    """
    try:
        # 读取数据
        data = np.load(input_filename, allow_pickle=True)
        print(f"成功加载数据: {input_filename}")
        print(f"总记录数: {len(data)}")
        
        if not data.dtype.names:
            print("错误: 数据不是结构化数组")
            return None, None
        
        # 检查repeater_name字段
        if 'repeater_name' not in data.dtype.names:
            print("错误: 数据中没有'repeater_name'字段")
            print(f"可用字段: {data.dtype.names}")
            return None, None
        
        # 获取repeater_name字段
        repeater_names = data['repeater_name']
        frb_name = data['tns_name']
        
        # 判断哪些记录有repeater_name
        if np.issubdtype(repeater_names.dtype, np.str_):
            # 字符串类型：空字符串视为无repeater_name
            has_repeater_mask = repeater_names.astype(str) != ''
        else:
            # 其他类型：非NaN视为有repeater_name
            has_repeater_mask = ~np.isnan(repeater_names.astype(float))
        
        # 分离数据
        repeater_data = data[has_repeater_mask]
        non_repeater_data = data[~has_repeater_mask]
        
        # 统计信息
        print(f"\nrepeater_name统计:")
        print(f"- 有repeater_name的记录数: {len(repeater_data)}")
        print(f"- 无repeater_name的记录数: {len(non_repeater_data)}")
        
        # 确定输出文件基本名
        if output_basename is None:
            output_basename = os.path.splitext(input_filename)[0]
        
        # 保存repeater和non-repeater数据
        repeater_filename = f"{output_basename}_repeater.npy"
        non_repeater_filename = f"{output_basename}_non_repeater.npy"
        
        if len(repeater_data) > 0:
            np.save(repeater_filename, repeater_data)
            print(f"\n已保存所有repeater数据: '{repeater_filename}'")
            
            # 获取唯一的repeater_name和对应的第一条记录索引
            unique_repeater_names, first_indices = np.unique(repeater_data['repeater_name'],\
                                                                       return_index=True)
            
            print(f"- 唯一repeater_name数量: {len(unique_repeater_names)}")
            
            # 提取每个唯一repeater_name的第一条记录（去重后的repeater数据）
            unique_repeater_data = repeater_data[first_indices]
            
            # 保存唯一repeater_name对应的数据（去重后的）
            unique_repeater_filename = f"{output_basename}_first_repeaters.npy"
            np.save(unique_repeater_filename, unique_repeater_data)
            print(f"- 已保存唯一repeater_name数据: '{unique_repeater_filename}'")
            
            # 显示前10个唯一repeater_name
            print(f"\n前10个唯一repeater_name:")
            for i, (name, idx) in enumerate(zip(unique_repeater_names, first_indices)):
                if i < 10:
                    # 获取该repeater_name的总记录数
                    count = np.sum(repeater_data['repeater_name'] == name)
                    print(f" '{name}' - 共{count}条记录")
                else:
                    print(f"  ... 还有{len(unique_repeater_names)-10}个唯一repeater_name")
                    break
        
        if len(non_repeater_data) > 0:
            np.save(non_repeater_filename, non_repeater_data)
            print(f"\n已保存non-repeater数据: '{non_repeater_filename}'")
        
        # 验证文件保存成功
        print(f"\n验证文件保存:")
        try:
            saved_repeater = np.load(repeater_filename, allow_pickle=True)
            print(f"- repeater文件: {len(saved_repeater)}条记录")
        except:
            print(f"- repeater文件: 加载失败")
        
        try:
            saved_unique = np.load(unique_repeater_filename, allow_pickle=True)
            print(f"- 唯一repeater文件: {len(saved_unique)}条记录")
        except:
            print(f"- 唯一repeater文件: 加载失败")
        
        try:
            saved_non = np.load(non_repeater_filename, allow_pickle=True)
            print(f"- non-repeater文件: {len(saved_non)}条记录")
        except:
            print(f"- non-repeater文件: 加载失败")
        
        # 显示数据结构
        if 'saved_repeater' in locals():
            print(f"\nrepeater数据结构:")
            if saved_repeater.dtype.names:
                print(f"  字段数: {len(saved_repeater.dtype.names)}")
                print(f"  前5个字段: {saved_repeater.dtype.names[:5]}")
        
        return repeater_data, non_repeater_data
        
    except Exception as e:
        print(f"处理数据时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def process_frb_data(input_filename, output_basename=None):
    """
    处理FRB数据，分离重复和唯一的tns_name数据
    
    Parameters:
    -----------
    input_filename : str
        输入数据文件路径
    output_basename : str, optional
        输出文件基本名，如果为None则使用输入文件的基本名
    """
    try:
        # 读取数据
        data = np.load(input_filename, allow_pickle=True)
        print(f"成功加载数据: {input_filename}")
        print(f"总记录数: {len(data)}")
        
        if not data.dtype.names:
            print("错误: 数据不是结构化数组")
            return
        
        # 检查必要字段
        if 'tns_name' not in data.dtype.names:
            print(f"错误: 数据中没有'tns_name'字段")
            print(f"可用字段: {data.dtype.names}")
            return
        
        # 获取tns_name数组
        tns_names = data['tns_name']
        
        # 找出唯一和重复的tns_name
        unique_names, name_indices, name_counts = np.unique(tns_names, 
                                                           return_index=True, 
                                                           return_counts=True)
        
        # 分离数据
        unique_mask = np.zeros(len(data), dtype=bool)
        duplicate_mask = np.zeros(len(data), dtype=bool)
        first_duplicate_mask = np.zeros(len(data), dtype=bool)
        fin_duplicate_mask = np.zeros(len(data), dtype=bool)
        
        for i, (name, idx, count) in enumerate(zip(unique_names, name_indices, name_counts)):
            if count == 1:
                unique_mask[idx] = True
            else:
                # 找到该name的所有记录
                name_mask = tns_names == name
                # 标记所有重复记录
                duplicate_mask[name_mask] = True
                # 标记第一条和最后一条重复记录
                first_idx = np.where(name_mask)[0][0]
                first_duplicate_mask[first_idx] = True
                fin_idx = np.where(name_mask)[0][-1]
                fin_duplicate_mask[fin_idx] = True
        
        # 创建第一个数据集：唯一记录 + 每条重复记录的第一条
        data_set1 = data[unique_mask | first_duplicate_mask]
        
        # 创建第二个数据集：所有重复记录
        data_set2 = data[duplicate_mask]
        
        # 创建第三个数据集：每条重复记录的第一条
        data_set3 = data[fin_duplicate_mask]
        
        # 统计信息
        print(f"\n数据统计:")
        print(f"- 唯一tns_name数量: {np.sum(name_counts == 1)}")
        print(f"- 重复tns_name组数: {np.sum(name_counts > 1)}")
        print(f"- 重复tns_name总记录数: {np.sum(duplicate_mask)}")
        print(f"- 数据集1记录数 (唯一+第一条重复): {len(data_set1)}")
        print(f"- 数据集2记录数 (所有重复): {len(data_set2)}")
        print(f"- 数据集3记录数 (第一条重复): {len(data_set3)}")
        
        # 显示重复次数与sub_num的关系
        if 'sub_num' in data.dtype.names:
            print(f"\n重复数据统计 (每个tns_name的重复次数与对应的sub_num):")
            print("-" * 80)
            
            # 分析重复数据
            repeat_names = unique_names[name_counts > 1]
            repeat_counts = name_counts[name_counts > 1]
            
            print(f"\n总共 {len(repeat_names)} 组重复tns_name:")
            print("-" * 80)
            
            # 显示每组重复数据的详细信息
            for name, count in zip(repeat_names, repeat_counts):
                # 找到该tns_name的所有记录
                name_mask = tns_names == name
                indices = np.where(name_mask)[0]
                
                # 获取对应的sub_num
                sub_nums = data['sub_num'][indices]
                
                print(f"tns_name: {name:15} 重复次数: {count:3} sub_num: {sub_nums}")
        
        # 确定输出文件基本名
        if output_basename is None:
            output_basename = os.path.splitext(input_filename)[0]
        
        # 保存文件
        unique_filename = f"{output_basename}_unique.npy"
        duplicates_filename = f"{output_basename}_duplicates.npy"
        duplicates_first_filename = f"{output_basename}_first_duplicates.npy"
        
        np.save(unique_filename, data_set1)
        np.save(duplicates_filename, data_set2)
        np.save(duplicates_first_filename, data_set3)
        
        print(f"\n已保存文件:")
        print(f"- 唯一数据: '{unique_filename}'")
        print(f"- 重复数据: '{duplicates_filename}'")
        print(f"- 重复数据: '{duplicates_first_filename}'")
        
        return data_set1, data_set2, data_set3
        
    except Exception as e:
        print(f"处理数据时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


def quick_summary(filename):
    """
    快速检查数据文件
    """
    try:
        data = np.load(filename, allow_pickle=True)
        print(f"文件: {filename}")
        print(f"总记录数: {len(data)}")
        
        if data.dtype.names:
            print(f"字段数量: {len(data.dtype.names)}")
            print(f"前10个字段: {data.dtype.names[:10]}")
            
            if 'tns_name' in data.dtype.names:
                tns_names = data['tns_name']
                unique_names, counts = np.unique(tns_names, return_counts=True)
                print(f"tns_name唯一值数量: {len(unique_names)}")
                print(f"重复tns_name组数: {np.sum(counts > 1)}")
            
            if 'repeater_name' in data.dtype.names:
                repeater_names = data['repeater_name']
                if np.issubdtype(repeater_names.dtype, np.str_):
                    non_empty = np.sum(repeater_names.astype(str) != '')
                else:
                    non_empty = np.sum(~np.isnan(repeater_names.astype(float)))
                print(f"repeater_name非空数量: {non_empty}")
                
    except Exception as e:
        print(f"检查数据时出错: {e}")


def list_saved_files(directory, pattern=None):
    """
    列出保存的文件
    """
    if os.path.isdir(directory):
        files = os.listdir(directory)
    else:
        # 如果输入是文件，则获取其所在目录
        directory = os.path.dirname(directory) if os.path.isfile(directory) else '.'
        files = os.listdir(directory)
    
    print(f"\n目录 '{directory}' 中的相关文件:")
    for file in sorted(files):
        if pattern is None or pattern in file:
            filepath = os.path.join(directory, file)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                size_kb = size / 1024
                print(f"  {file} ({size_kb:.1f} KB)")


def load_and_display(filename):
    """
    加载并显示数据文件的基本信息
    
    Parameters:
    -----------
    filename : str
        数据文件路径
    
    Returns:
    --------
    data : numpy array
        加载的数据，如果加载失败则返回None
    """
    try:
        # 读取数据
        data = np.load(filename, allow_pickle=True)
        
        print(f"=== 文件信息: {filename} ===")
        print(f"文件大小: {os.path.getsize(filename) / 1024:.2f} KB")
        print(f"总记录数: {len(data)}")
        
        if data.dtype.names:
            # 结构化数组
            print(f"字段数量: {len(data.dtype.names)}")
            print("\n字段详情:")
            for i, name in enumerate(data.dtype.names[:10]):  # 最多显示前10个字段
                field = data[name]
                unique_count = len(np.unique(field))
                print(f"  {i+1:2d}. {name:20s} | 类型: {str(field.dtype):15s} | "
                      f"唯一值: {unique_count:6d}")
            
            if len(data.dtype.names) > 10:
                print(f"  ... 还有{len(data.dtype.names)-10}个字段未显示")
            
            # 检查是否有特定字段
            for field in ['tns_name', 'repeater_name']:
                if field in data.dtype.names:
                    field_data = data[field]
                    if np.issubdtype(field_data.dtype, np.str_):
                        non_empty = np.sum(field_data.astype(str) != '')
                    else:
                        non_empty = np.sum(~np.isnan(field_data.astype(float)))
                    print(f"\n{field}字段:")
                    print(f"  - 非空值数: {non_empty}")
                    print(f"  - 唯一值数: {len(np.unique(field_data))}")
                    
                    # 显示前几个非空值
                    if non_empty > 0:
                        non_empty_values = field_data[field_data != ''] if np.issubdtype(field_data.dtype, np.str_) else field_data[~np.isnan(field_data)]
                        if len(non_empty_values) > 0:
                            print(f"  - 前10个值: {non_empty_values[:10]}")
        else:
            # 非结构化数组
            print(f"数组形状: {data.shape}")
            print(f"数组维度: {data.ndim}")
            print(f"数据类型: {data.dtype}")
            if len(data) > 0:
                print(f"前10个元素: {data[:10]}")
        
        print("-" * 50)
        return data
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{filename}'")
        return None
    except Exception as e:
        print(f"加载数据时发生错误: {str(e)}")
        return None


def explore_h5_structure(file_path):
    """
    探索HDF5文件的结构和内容
    
    参数:
        file_path (str): HDF5文件路径
    """
    print(f"探索文件: {file_path}")
    print("=" * 50)
    
    with h5py.File(file_path, 'r') as f:
        # 打印文件整体信息
        print(f"文件属性: {dict(f.attrs)}")
        print("\n数据结构:")
        
        # 递归遍历所有组和数据集
        def print_structure(name, obj):
            indent_level = name.count('/')
            indent = '  ' * indent_level
            
            if isinstance(obj, h5py.Group):
                print(f"{indent}📁 {name.split('/')[-1] or '/'} (Group)")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}📊 {name.split('/')[-1]} (Dataset)")
                print(f"{indent}    形状: {obj.shape}, 类型: {obj.dtype}")
                if obj.shape and len(obj.shape) <= 2:
                    print(f"{indent}    值范围: [{obj[:].min():.4g}, {obj[:].max():.4g}]")
            
            # 打印属性
            if obj.attrs:
                for attr_name, attr_value in obj.attrs.items():
                    if isinstance(attr_value, (str, int, float, np.number)):
                        print(f"{indent}    @{attr_name}: {attr_value}")
                    elif isinstance(attr_value, np.ndarray) and attr_value.size <= 10:
                        print(f"{indent}    @{attr_name}: {attr_value}")
        
        f.visititems(print_structure)
        

def read_frb_dynamic_spectrum(file_path, spectrum_key='spectrum'):
    """
    读取FRB动态谱HDF5文件
    针对CHIME/FRB Catalog 2格式优化
    
    参数:
        file_path (str): HDF5文件路径
        spectrum_key (str): 动态谱数据集的键名，默认为'spectrum'
        
    返回:
        dict: 包含动态谱数据及完整元数据的字典
    """
    data = {}
    
    with h5py.File(file_path, 'r') as f:
        # 1. 读取文件属性
        data['attrs'] = dict(f.attrs)
        attrs = data['attrs']
        
        print(f"读取文件: {file_path}")
        print(f"FRB名称: {attrs.get('tns_name', '未知')}")
        print(f"仪器: {attrs.get('instrument', '未知')}")
        
        # 2. 解析JSON格式的参数
        if 'burst_parameters_json' in attrs:
            try:
                data['burst_params'] = json.loads(attrs['burst_parameters_json'])
                print("成功解析爆发参数JSON")
            except:
                print("警告: 无法解析burst_parameters_json")
                data['burst_params'] = {}
        
        if 'pipeline_parameters_json' in attrs:
            try:
                data['pipeline_params'] = json.loads(attrs['pipeline_parameters_json'])
                print("成功解析处理管道参数JSON")
            except:
                print("警告: 无法解析pipeline_parameters_json")
                data['pipeline_params'] = {}
        
        # 3. 读取动态谱数据
        if spectrum_key in f:
            data['dynamic_spectrum'] = np.array(f[spectrum_key])
            print(f"读取动态谱: {spectrum_key}, 形状: {data['dynamic_spectrum'].shape}")
        else:
            # 尝试自动查找动态谱数据集
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    if f[key].ndim == 2 and f[key].shape[0] == attrs.get('num_freq', 0):
                        data['dynamic_spectrum'] = np.array(f[key])
                        spectrum_key = key
                        print(f"自动找到动态谱: {key}, 形状: {data['dynamic_spectrum'].shape}")
                        break
        
        # 4. 构建频率轴 (根据CHIME/FRB属性)
        num_freq = attrs.get('num_freq', 0)
        freqs_bin0 = attrs.get('freqs_bin0', 0.0)
        res_freq = attrs.get('res_freq', 0.0)
        
        if num_freq > 0:
            data['frequencies'] = freqs_bin0 + res_freq * np.arange(num_freq)
            print(f"构建频率轴: {len(data['frequencies'])} 个通道, "
                  f"范围: {data['frequencies'][0]:.3f} - {data['frequencies'][-1]:.3f} MHz")
        else:
            print("警告: 无法构建频率轴，缺少必要属性")
        
        # 5. 构建时间轴 (根据CHIME/FRB属性)
        num_time = attrs.get('num_time', 0)
        times_bin0 = attrs.get('times_bin0', 0.0)
        res_time = attrs.get('res_time', 0.0)
        
        if num_time > 0:
            data['times'] = times_bin0 + res_time * np.arange(num_time)
            # 转换为相对时间 (以中心时间为参考)
            center_time = attrs.get('center_time', 0.0)
            data['times_relative'] = data['times'] - center_time
            print(f"构建时间轴: {len(data['times'])} 个时间点, "
                  f"窗口: {data['times_relative'][0]:.3f} - {data['times_relative'][-1]:.3f} s")
        else:
            print("警告: 无法构建时间轴，缺少必要属性")
        
        # 6. 提取关键参数
        key_params = ['dm_incoherent', 'dm_index', 'ref_freq', 'center_time', 
                     'telescope', 'event_id', 'beam_number']
        
        for param in key_params:
            if param in attrs:
                data[param] = attrs[param]
        
        # 7. 直接从burst参数获取DM值
        if 'burst_params' in data and 'dm' in data['burst_params']:
            data['dm'] = data['burst_params']['dm'][0]  # 取第一个值
        elif 'dm_incoherent' in data:
            data['dm'] = data['dm_incoherent']
        
        # 8. 记录脉冲发射区域
        if 'pulse_emission_region' in attrs:
            data['pulse_emission_region'] = attrs['pulse_emission_region']
            print(f"脉冲发射区域: {data['pulse_emission_region']}")
        
        # 9. 收集其他所有数据集
        for key in f.keys():
            if key != spectrum_key and isinstance(f[key], h5py.Dataset):
                data[key] = np.array(f[key])
    
    return data

def print_data_summary(data):
    """
    打印数据摘要信息
    """
    print("\n" + "="*60)
    print("数据摘要:")
    print("="*60)
    
    if 'dynamic_spectrum' in data:
        ds = data['dynamic_spectrum']
        print(f"动态谱形状: {ds.shape} (频率×时间)")
        print(f"动态谱范围: [{ds.min():.3f}, {ds.max():.3f}]")
    
    if 'frequencies' in data:
        freqs = data['frequencies']
        print(f"频率范围: {freqs[0]:.3f} - {freqs[-1]:.3f} MHz "
              f"({len(freqs)} 通道, 分辨率: {freqs[1]-freqs[0]:.5f} MHz)")
    
    if 'times_relative' in data:
        times = data['times_relative']
        print(f"时间窗口: {times[0]:.5f} - {times[-1]:.5f} s "
              f"({len(times)} 点, 分辨率: {times[1]-times[0]:.6f} s)")
    
    if 'dm' in data:
        print(f"色散量 (DM): {data['dm']} pc/cm³")
    
    if 'center_time' in data:
        print(f"中心时间: {data['center_time']}")
    
    print(f"数据包含的键: {list(data.keys())}")
    print("="*60)













