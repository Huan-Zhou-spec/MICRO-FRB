#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:08:41 2026

@author: ubuntu
"""

'''
import numpy as np
from modules import (
    read_frb_dynamic_spectrum
)

#以FWHM和时间延迟作为标准来确定信号宽度
def compute_fwhm(time_series, peak_index, interp=True):
    """返回左半高位置、右半高位置（浮点数）及 FWHM"""
    peak_flux = time_series[peak_index]
    half_flux = peak_flux / 2.0

    # 向左搜索
    left_idx = peak_index
    while left_idx > 0 and time_series[left_idx] > half_flux:
        left_idx -= 1
    if interp and left_idx < peak_index:
        x1, y1 = left_idx, time_series[left_idx]
        x2, y2 = left_idx + 1, time_series[left_idx + 1]
        left_pos = x1 + (half_flux - y1) / (y2 - y1) if y2 != y1 else x1
    else:
        left_pos = left_idx

    # 向右搜索
    right_idx = peak_index
    while right_idx < len(time_series) - 1 and time_series[right_idx] > half_flux:
        right_idx += 1
    if interp and right_idx > peak_index:
        x1, y1 = right_idx - 1, time_series[right_idx - 1]
        x2, y2 = right_idx, time_series[right_idx]
        right_pos = x1 + (half_flux - y1) / (y2 - y1) if y2 != y1 else x1
    else:
        right_pos = right_idx

    return left_pos, right_pos, right_pos - left_pos


def extract_with_fwhm(frb_name, peak1_index, peak2_index,
                      window_factor=1.0, signal_factor=2.5,
                      chime_file='FRB_data/CHIME_cat2_frb/chimefrbcat2.npy',
                      data_dir='FRB_data/canfar_downloads/'):
    """
    基于总光变曲线中 peak1 的半高宽确定整数半径，构建对称奇数长度窗口。
    若两峰相距足够远，各自以峰为中心构建不重叠的对称窗口；
    若靠得太近，则以 peak_cut 为 peak1 的右边界，确定窗口长度后平移至 peak2。
    """
    # 1. 读取动态谱
    file_path = f"{data_dir}{frb_name}_stokesi_dynamic_spectrum.h5"
    raw = read_frb_dynamic_spectrum(file_path)
    freq_axis = raw['frequencies']
    dynamic = raw['dynamic_spectrum']

    # 2. 获取观测频带
    chime_data = np.load(chime_file, allow_pickle=True)
    entry = next((e for e in chime_data if e['tns_name'] == frb_name), None)
    if entry is None:
        raise ValueError(f"未找到 {frb_name} 对应的 CHIME 数据")
    low_freq, high_freq = entry['low_freq'], entry['high_freq']
    print(low_freq, high_freq)

    # 3. 频带切割并合成 L/M/H 及总光变曲线
    idx_low = np.argmin(np.abs(freq_axis - low_freq))
    idx_high = np.argmin(np.abs(freq_axis - high_freq))
    if idx_low > idx_high:
        idx_low, idx_high = idx_high, idx_low
    sub_band = dynamic[idx_low:idx_high+1, :]

    groups = np.array_split(sub_band, 3, axis=0)
    L = np.sum(groups[0], axis=0)
    M = np.sum(groups[1], axis=0)
    H = np.sum(groups[2], axis=0)
    total = L + M + H
    T_len = total.shape[0]

    # 4. 计算 peak1 的 FWHM 和理论半径（像素数的一半）
    left1_raw, right1_raw, fwhm1 = compute_fwhm(total, peak1_index, interp=True)
    _, _, fwhm2 = compute_fwhm(total, peak2_index, interp=True)
    raw_radius_float = (right1_raw - left1_raw) / 2.0 * window_factor

    # 两峰中点
    peak_cut_float = (peak1_index + peak2_index) / 2.0
    peak_cut = int(round(peak_cut_float-0.1))

    # 判断是否足够分离：原始半径是否会使两个窗口重叠（或接触）
    half_sep = (peak2_index - peak1_index) / 2.0
    use_shared_boundary = (raw_radius_float >= half_sep)

    if use_shared_boundary:
        # ---------- 共用边界模式（保持不变） ----------
        # 1) 根据 peak1 和 peak_cut 确定半长度（整数）
        half_len = peak_cut - peak1_index
        if half_len < 0:
            raise ValueError("peak_cut 小于 peak1_index，无法构建窗口")
        # 窗口长度 = 2*half_len + 1（奇数）
        # peak1 窗口边界
        left1 = peak1_index - half_len
        right1_exclusive = peak_cut + 1   # 包含 peak_cut
        # 检查 peak1 窗口不越界
        if left1 < 0 or right1_exclusive > T_len:
            raise ValueError(f"peak1 窗口超出数据范围: left1={left1}, right1={right1_exclusive}, T_len={T_len}")
        
        # 2) 相同半长度应用到 peak2，对称于 peak2_index
        left2 = peak2_index - half_len
        right2_exclusive = peak2_index + half_len + 1
        # 检查 peak2 窗口越界，若越界则尝试调整（保持对称，但可能改变长度？尽量保持）
        if left2 < 0:
            shift = -left2
            left2 = 0
            right2_exclusive = min(T_len, right2_exclusive + shift)
            actual_half_len = (right2_exclusive - left2 - 1) // 2
            if actual_half_len < 0:
                raise ValueError("peak2 窗口无法调整至有效范围")
            if right2_exclusive - left2 != 2*half_len + 1:
                print(f"警告: peak2 窗口因边界截断，长度从 {2*half_len+1} 变为 {right2_exclusive-left2}")
        if right2_exclusive > T_len:
            shift = right2_exclusive - T_len
            right2_exclusive = T_len
            left2 = max(0, left2 - shift)
            actual_half_len = (right2_exclusive - left2 - 1) // 2
            if actual_half_len < 0:
                raise ValueError("peak2 窗口无法调整至有效范围")
            if right2_exclusive - left2 != 2*half_len + 1:
                print(f"警告: peak2 窗口因边界截断，长度从 {2*half_len+1} 变为 {right2_exclusive-left2}")
        
        # 最终检查窗口有效且包含峰值
        if left1 >= right1_exclusive or left2 >= right2_exclusive:
            raise ValueError("共用边界模式下窗口无效")
        if not (left1 <= peak1_index < right1_exclusive):
            raise ValueError("peak1 不在其窗口内")
        if not (left2 <= peak2_index < right2_exclusive):
            raise ValueError("peak2 不在其窗口内")
        
        radius = half_len  # 实际使用的半长度（peak1 严格，peak2 可能因越界而变小）
        
    else:
        # ---------- 正常模式：先确定 peak1 窗口，再令 peak2 窗口与之等长且以 peak2 为中心 ----------
        # 最大允许的半窗口长度
        max_half_len1 = peak_cut - peak1_index - 1  # 确保右边界 < peak_cut
        max_half_len2 = peak2_index - peak_cut - 1  # 确保左边界 > peak_cut
        if max_half_len1 < 0 or max_half_len2 < 0:
            # 这种情况应进入共用边界模式，但由于浮点比较可能未进入，强制转换
            print("警告：正常模式下最大半长度负数，转为共用边界模式")
            return extract_with_fwhm(frb_name, peak1_index, peak2_index,
                                     window_factor, signal_factor,
                                     chime_file, data_dir)
        
        half_len_float = raw_radius_float
        half_len = int(round(half_len_float))
        half_len1 = min(half_len, max_half_len1)
        
        # ----- 确定 peak1 窗口（沿用原有边界调整逻辑）-----
        left1 = peak1_index - half_len1
        right1_exclusive = peak1_index + half_len1 + 1
        
        # 边界调整（保留原代码的完整性）
        if left1 < 0:
            left1 = 0
            right1_exclusive = peak1_index + (peak1_index - left1) + 1
        if right1_exclusive > T_len:
            right1_exclusive = T_len
            left1 = peak1_index - (right1_exclusive - peak1_index - 1)
        if right1_exclusive > peak_cut:
            right1_exclusive = peak_cut
            left1 = peak1_index - (right1_exclusive - peak1_index - 1)
            if left1 < 0:
                left1 = 0
                right1_exclusive = peak1_index + (peak1_index - left1) + 1
        
        # 确保窗口有效且包含 peak1
        if left1 >= right1_exclusive:
            raise ValueError("正常模式下 peak1 窗口无效")
        if not (left1 <= peak1_index < right1_exclusive):
            raise ValueError("peak1 不在其窗口内")
        
        # 窗口长度（点数）
        len1 = right1_exclusive - left1
        # 要求长度为奇数（原逻辑保证，但边界调整后仍应为奇数）
        if len1 % 2 == 0:
            print(f"警告：peak1 窗口长度为偶数 {len1}，可能无法严格对称，将按原长度处理")
        
        # ----- 为 peak2 构造等长、以 peak2 为中心的窗口 -----
        half2 = (len1 - 1) // 2          # 半窗口长度（向下取整，保证总长度为奇数）
        left2 = peak2_index - half2
        right2_exclusive = peak2_index + half2 + 1
        
        # 检查是否满足不越过 peak_cut 及数据边界
        valid = True
        if left2 <= peak_cut:
            print(f"警告：以 peak2 为中心的窗口左边界 {left2} 不满足 > peak_cut ({peak_cut})，转为共用边界模式")
            valid = False
        if left2 < 0 or right2_exclusive > T_len:
            print(f"警告：以 peak2 为中心的窗口超出数据边界 (left2={left2}, right2={right2_exclusive})，转为共用边界模式")
            valid = False
        
        if not valid:
            # 回退到共用边界模式
            return extract_with_fwhm(frb_name, peak1_index, peak2_index,
                                     window_factor, signal_factor,
                                     chime_file, data_dir)
        
        # 验证窗口包含 peak2
        if not (left2 <= peak2_index < right2_exclusive):
            raise ValueError("peak2 不在构造的窗口内")
        
        # 记录实际使用的半长度（基于 peak1）
        radius = (len1 - 1) // 2
    
    # 提取窗口内的数据
    L_left = L[left1:right1_exclusive]
    M_left = M[left1:right1_exclusive]
    H_left = H[left1:right1_exclusive]
    L_right = L[left2:right2_exclusive]
    M_right = M[left2:right2_exclusive]
    H_right = H[left2:right2_exclusive]

    # 噪声区域（基于信号扩展）
    signal_pad1 = int(round(signal_factor * fwhm1))
    signal_pad2 = int(round(signal_factor * fwhm2))
    signal_start = max(0, peak1_index - signal_pad1)
    signal_end = min(T_len, peak2_index + signal_pad2)
    print([signal_start, signal_end])

    def slice_noise(data):
        left = data[:signal_start] if signal_start > 0 else np.array([])
        right = data[signal_end:] if signal_end < T_len else np.array([])
        return left, right

    L_left_noise, L_right_noise = slice_noise(L)
    M_left_noise, M_right_noise = slice_noise(M)
    H_left_noise, H_right_noise = slice_noise(H)

    def concat_safe(a, b):
        if a.size and b.size:
            return np.concatenate([a, b])
        return a if a.size else b

    L_noise_all = concat_safe(L_left_noise, L_right_noise)
    M_noise_all = concat_safe(M_left_noise, M_right_noise)
    H_noise_all = concat_safe(H_left_noise, H_right_noise)

    return {
        'L_left': L_left, 'L_right': L_right,
        'M_left': M_left, 'M_right': M_right,
        'H_left': H_left, 'H_right': H_right,
        'L_noise_all': L_noise_all, 'M_noise_all': M_noise_all, 'H_noise_all': H_noise_all,
        'L_noise_left': L_left_noise, 'L_noise_right': L_right_noise,
        'M_noise_left': M_left_noise, 'M_noise_right': M_right_noise,
        'H_noise_left': H_left_noise, 'H_noise_right': H_right_noise,
        'n_left': len(L_left), 'n_right': len(L_right),
        'peak_cut_index': peak_cut,
        'left_boundary1': left1, 'right_boundary1': right1_exclusive,
        'left_boundary2': left2, 'right_boundary2': right2_exclusive,
        'radius': radius,
        'fwhm1': fwhm1, 'fwhm2': fwhm2,
        'shared_boundary': use_shared_boundary
    }

#硬度比计算（高斯误差模型）
def net_intensity_and_error(I_sum, noise_rms, n_bins, bg_mean, N_noise):
    """
    计算净强度 I_sum - B_sum 及其高斯误差。
    参数:
        I_sum: 窗口总强度（未减背景）
        noise_rms: 噪声区域每个时间点的标准差（背景涨落）
        n_bins: 窗口内点数
        bg_mean: 噪声区域每个时间点的均值
        N_noise: 噪声区域总点数
    返回: (净强度, 净强度误差)
    """
    # 总强度误差（背景噪声涨落）
    I_err = np.sqrt(n_bins) * noise_rms
    # 背景总计数误差
    if N_noise > 0:
        B_err = noise_rms * n_bins / np.sqrt(N_noise)
    else:
        B_err = 0.0
    net = I_sum - bg_mean * n_bins
    net_err = np.sqrt(I_err**2 + B_err**2)
    return net, net_err


def hardness_ratio_ml_hm(M_sum, L_sum, H_sum,
                         noise_rms_M, noise_rms_L, noise_rms_H,
                         n_bins,
                         bg_mean_M, bg_mean_L, bg_mean_H,
                         N_noise_M, N_noise_L, N_noise_H):
    """
    计算硬度比 ML = (M - B_M)/(L - B_L) 和 HM = (H - B_H)/(M - B_M)，高斯误差。
    返回 (ML, err_ML, HM, err_HM)
    """
    M_net, err_M = net_intensity_and_error(M_sum, noise_rms_M, n_bins, bg_mean_M, N_noise_M)
    L_net, err_L = net_intensity_and_error(L_sum, noise_rms_L, n_bins, bg_mean_L, N_noise_L)
    H_net, err_H = net_intensity_and_error(H_sum, noise_rms_H, n_bins, bg_mean_H, N_noise_H)

    # ML = M_net / L_net
    if L_net <= 0 or M_net <= 0:
        ML, err_ML = np.nan, np.nan
    else:
        ML = M_net / L_net
        err_ML = ML * np.sqrt((err_M/M_net)**2 + (err_L/L_net)**2)

    # HM = H_net / M_net
    if M_net <= 0:
        HM, err_HM = np.nan, np.nan
    else:
        HM = H_net / M_net
        err_HM = HM * np.sqrt((err_H/H_net)**2 + (err_M/M_net)**2)

    return ML, err_ML, HM, err_HM


# ------------------------------------------------------------
# 4. 透镜质量计算函数
# ------------------------------------------------------------
def lens_mass(dt_ms, f):
    """
    计算透镜质量 M = M_L (1+z_L) [太阳质量]。
    公式: M = (c^3 * Δt) / [2G * ((f-1)*f^{-1/2} + ln f)]
    参数:
        dt_ms : float, 时间延迟 (毫秒)
        f     : float, 放大比 (净流量比)
    返回:
        M : float, 透镜质量 (太阳质量)
    """
    c = 3.0e8          # m/s
    G = 6.67430e-11    # m^3 kg^{-1} s^{-2}
    M_sun = 1.989e30   # kg
    dt_sec = dt_ms * 1e-3
    # 分母项 g(f) = (f-1)*f^{-1/2} + ln f
    g = (f - 1) * f**(-0.5) + np.log(f)
    M_kg = (c**3 * dt_sec) / (2.0 * G * g)
    M = M_kg / M_sun
    return M
'''

import numpy as np
from modules import (
    extract_with_fwhm, net_intensity_and_error, hardness_ratio_ml_hm, lens_mass
)
# ------------------------------------------------------------
# 5. 主程序
# ------------------------------------------------------------
if __name__ == "__main__":
    # ---- 1. 提取数据 ----
    dt_ms = 3.92   # 时间延迟（毫秒）
    data = extract_with_fwhm(
        frb_name="FRB20230402B",
        peak1_index=52,
        peak2_index=55,
        window_factor=3,
        signal_factor=10,
    )
    
    # ---- 2. 准备窗口数据 ----
    L_left, L_right = data['L_left'], data['L_right']
    M_left, M_right = data['M_left'], data['M_right']
    H_left, H_right = data['H_left'], data['H_right']
    n_left = data['n_left']
    n_right = data['n_right']
    
    # 全局噪声统计（左右噪声合并）
    L_noise = data['L_noise_all']
    M_noise = data['M_noise_all']
    H_noise = data['H_noise_all']
    
    rms_L = np.std(L_noise) if len(L_noise) > 0 else 0.0
    rms_M = np.std(M_noise) if len(M_noise) > 0 else 0.0
    rms_H = np.std(H_noise) if len(H_noise) > 0 else 0.0
    
    bg_mean_L = np.mean(L_noise) if len(L_noise) > 0 else 0.0
    bg_mean_M = np.mean(M_noise) if len(M_noise) > 0 else 0.0
    bg_mean_H = np.mean(H_noise) if len(H_noise) > 0 else 0.0
    
    N_noise_L = len(L_noise)
    N_noise_M = len(M_noise)
    N_noise_H = len(H_noise)
    
    # 窗口总强度（未减背景）
    L_left_sum = np.sum(L_left)
    M_left_sum = np.sum(M_left)
    H_left_sum = np.sum(H_left)
    L_right_sum = np.sum(L_right)
    M_right_sum = np.sum(M_right)
    H_right_sum = np.sum(H_right)
    
    # ---- 3. 计算左窗口硬度比 ----
    ML_left, err_ML_left, HM_left, err_HM_left = hardness_ratio_ml_hm(
        M_left_sum, L_left_sum, H_left_sum,
        rms_M, rms_L, rms_H,
        n_left,
        bg_mean_M, bg_mean_L, bg_mean_H,
        N_noise_M, N_noise_L, N_noise_H
    )
    
    # ---- 4. 计算右窗口硬度比 ----
    ML_right, err_ML_right, HM_right, err_HM_right = hardness_ratio_ml_hm(
        M_right_sum, L_right_sum, H_right_sum,
        rms_M, rms_L, rms_H,
        n_right,
        bg_mean_M, bg_mean_L, bg_mean_H,
        N_noise_M, N_noise_L, N_noise_H
    )
    
    # ---- 5. 计算透镜体的红移质量（使用净强度）----
    # 左窗口净总强度
    L_left_net, _ = net_intensity_and_error(L_left_sum, rms_L, n_left, bg_mean_L, N_noise_L)
    M_left_net, _ = net_intensity_and_error(M_left_sum, rms_M, n_left, bg_mean_M, N_noise_M)
    H_left_net, _ = net_intensity_and_error(H_left_sum, rms_H, n_left, bg_mean_H, N_noise_H)
    I_L_net = L_left_net + M_left_net + H_left_net
    
    # 右窗口净总强度
    L_right_net, _ = net_intensity_and_error(L_right_sum, rms_L, n_right, bg_mean_L, N_noise_L)
    M_right_net, _ = net_intensity_and_error(M_right_sum, rms_M, n_right, bg_mean_M, N_noise_M)
    H_right_net, _ = net_intensity_and_error(H_right_sum, rms_H, n_right, bg_mean_H, N_noise_H)
    I_R_net = L_right_net + M_right_net + H_right_net
    
    if I_R_net > 0:
        R = I_L_net / I_R_net
        Mz = lens_mass(dt_ms, R)
        print(f"透镜红移质量 M_L(1+z_L) = {Mz:.4e} M_sun, 放大比 R = {R:.4f}")
    else:
        print("右窗口净强度非正，无法计算放大比")
        R = np.nan
    
    # ---- 6. 输出硬度比结果 ----
    print("\n===== 左窗口（峰值前） =====")
    if not np.isnan(ML_left):
        print(f"HR_ML = {ML_left:.2f} ± {err_ML_left:.2f}")
    else:
        print("HR_ML 无效（净强度≤0）")
    if not np.isnan(HM_left):
        print(f"HR_HM = {HM_left:.2f} ± {err_HM_left:.2f}")
    else:
        print("HR_HM 无效（净强度≤0）")
    
    print("\n===== 右窗口（峰值后） =====")
    if not np.isnan(ML_right):
        print(f"HR_ML = {ML_right:.2f} ± {err_ML_right:.2f}")
    else:
        print("HR_ML 无效（净强度≤0）")
    if not np.isnan(HM_right):
        print(f"HR_HM = {HM_right:.2f} ± {err_HM_right:.2f}")
    else:
        print("HR_HM 无效（净强度≤0）")
    
    # ---- 7. 调试信息 ----
    print(f"\n噪声统计 (每时间点):")
    print(f"L: RMS={rms_L:.3f}, 均值={bg_mean_L:.3f}, 点数={N_noise_L}")
    print(f"M: RMS={rms_M:.3f}, 均值={bg_mean_M:.3f}, 点数={N_noise_M}")
    print(f"H: RMS={rms_H:.3f}, 均值={bg_mean_H:.3f}, 点数={N_noise_H}")
    print("Peak1 窗口范围: [%d, %d)" % (data['left_boundary1'], data['right_boundary1']))
    print("Peak2 窗口范围: [%d, %d)" % (data['left_boundary2'], data['right_boundary2']))
    print(data['fwhm1'], data['fwhm2'])
    print(f"左窗口点数={n_left}, 右窗口点数={n_right}")