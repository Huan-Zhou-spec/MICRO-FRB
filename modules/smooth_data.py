#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 19:49:18 2026

@author: ubuntu
"""

#绘制任意平滑之后的FRB的动态谱图
import numpy as np
import scipy.signal


#注意的是width是CHIM时间分辨率的倍数，即width=1时，平滑长度为0.98ms。
def boxcar_kernel_1d(width):
    """一维矩形窗口"""
    width = int(round(width, 0))
    return np.ones(width, dtype="float32") / np.sqrt(width)


def gaussian_kernel_1d(sigma, truncate=4.0):
    """一维高斯窗口"""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / np.sum(kernel)


def smooth_time_dimension(data, time_width=5, kernel_type='boxcar', boundary='symm'):
    """
    对FRB动态谱数据的时间维度进行一维平滑
    
    参数:
    data: 二维数组，形状为(频率通道数, 时间样本数)
    time_width: 时间维度的平滑窗口宽度
    kernel_type: 核类型，'boxcar'或'gaussian'
    boundary: 边界处理方式，'symm'(对称)、'edge'(边缘)或'wrap'(循环)
    
    返回:
    平滑后的二维数据，形状与输入相同
    """
    # 创建合适的一维核
    if kernel_type == 'boxcar':
        kernel = boxcar_kernel_1d(time_width)
    elif kernel_type == 'gaussian':
        # sigma设置为时间宽度的一半
        kernel = gaussian_kernel_1d(time_width/2)
    else:
        raise ValueError("kernel_type必须是'boxcar'或'gaussian'")
    
    # 对每个频率通道（每行）沿时间轴进行卷积
    smoothed = np.apply_along_axis(
        lambda x: scipy.signal.convolve(x, kernel, mode='same', method='auto'),
        axis=1,  # 沿时间轴（每行）
        arr=data
    )
    
    return smoothed

def find_optimal_time_smooth(dynamic_spectrum, min_width=1, max_width=128):
    """
    寻找最优的时间维度平滑参数（基于最大SNR）
    
    参数:
    dynamic_spectrum: 二维动态谱数据
    min_width: 最小时间宽度
    max_width: 最大时间宽度
    
    返回:
    (最佳时间宽度, 最佳SNR, 平滑后的时间序列)
    """
    # 首先将动态谱沿频率轴积分，得到时间序列
    ts_original = dynamic_spectrum.sum(axis=0)
    
    min_width = int(min_width)
    max_width = int(max_width)
    
    # 限制宽度不超过时间序列长度
    widths = list(range(min_width, min(max_width + 1, len(ts_original) - 2)))
    
    # 记录每个宽度的SNR
    snrs = np.empty_like(widths, dtype=float)
    smoothed_ts_list = []
    
    for i, width in enumerate(widths):
        # 创建核
        kernel = boxcar_kernel_1d(width)
        
        # 平滑时间序列
        smoothed_ts = scipy.signal.convolve(ts_original, kernel, mode='same')
        smoothed_ts_list.append(smoothed_ts)
        
        # 计算SNR（峰值与背景噪声之比）
        peak_value = np.max(smoothed_ts)
        
        # 计算背景噪声（排除峰值附近区域）
        peak_idx = np.argmax(smoothed_ts)
        exclude_size = width * 2  # 排除峰值附近区域
        
        # 创建背景掩码
        mask = np.ones_like(smoothed_ts, dtype=bool)
        start_idx = max(0, peak_idx - exclude_size)
        end_idx = min(len(smoothed_ts), peak_idx + exclude_size + 1)
        mask[start_idx:end_idx] = False
        
        # 如果有足够的背景点，计算SNR
        background = smoothed_ts[mask]
        if len(background) > 0:
            noise_std = np.std(background)
            snrs[i] = peak_value / noise_std if noise_std > 0 else 0
        else:
            snrs[i] = 0
    
    # 找到最佳SNR对应的宽度
    best_idx = np.argmax(snrs)
    best_width = widths[best_idx]
    best_snr = snrs[best_idx]
    best_smoothed_ts = smoothed_ts_list[best_idx]
    
    return best_width, best_snr, best_smoothed_ts


def smooth_frb_data(data_dict, time_width=5, kernel_type='boxcar'):
    """
    对完整的FRB数据进行时间维度平滑
    
    参数:
    data_dict: FRB数据字典（包含dynamic_spectrum等）
    time_width: 时间维度的平滑窗口宽度
    kernel_type: 核类型
    
    返回:
    更新后的数据字典（包含平滑后的动态谱和时间序列）
    """
    # 获取原始数据
    dyn_spec = data_dict['dynamic_spectrum']
    
    # 对时间维度进行平滑
    smoothed_dyn_spec = smooth_time_dimension(dyn_spec, time_width, kernel_type)
    
    # 计算平滑后的时间序列（沿频率轴积分）
    smoothed_ts = smoothed_dyn_spec.sum(axis=0)
    
    # 更新数据字典
    data_dict_smoothed = data_dict.copy()
    data_dict_smoothed['dynamic_spectrum'] = smoothed_dyn_spec
    data_dict_smoothed['ts'] = smoothed_ts
    data_dict_smoothed['smooth_time_width'] = time_width
    data_dict_smoothed['smooth_kernel_type'] = kernel_type
    
    return data_dict_smoothed