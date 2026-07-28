#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:08:41 2026

@author: ubuntu
"""

import numpy as np
from modules import (extract_with_fwhm, net_intensity_and_error, 
hardness_ratio, compare_hardness_ratios, lens_mass)


if __name__ == "__main__":
    dt_ms = 8.82
    k = 3
    n_sigma = 1.0  # 置信度参数: 1.0=68%, 2.0=95%, 3.0=99.7%

    data = extract_with_fwhm(
        frb_name="FRB20190131D",
        peak1_index=76,
        peak2_index=85,
        k=k,
        window_factor=3,
        signal_factor=10,
    )

    n_bands = data['n_bands']
    n_left = data['n_left']
    n_right = data['n_right']

    band_left = [data[f'band{i}_left'] for i in range(n_bands)]
    band_right = [data[f'band{i}_right'] for i in range(n_bands)]
    band_noise = [data[f'band{i}_noise_all'] for i in range(n_bands)]

    rms_list = [np.std(bn) if len(bn) > 0 else 0.0 for bn in band_noise]
    bg_mean_list = [np.mean(bn) if len(bn) > 0 else 0.0 for bn in band_noise]
    N_noise_list = [len(bn) for bn in band_noise]

    left_sums = [np.sum(bl) for bl in band_left]
    right_sums = [np.sum(br) for br in band_right]

    hr_left, hr_err_left = hardness_ratio(
        left_sums, rms_list, [n_left] * n_bands,
        bg_mean_list, N_noise_list
    )

    hr_right, hr_err_right = hardness_ratio(
        right_sums, rms_list, [n_right] * n_bands,
        bg_mean_list, N_noise_list
    )

    I_L_net = 0.0
    I_R_net = 0.0
    for i in range(n_bands):
        net_l, _ = net_intensity_and_error(left_sums[i], rms_list[i], n_left, bg_mean_list[i], N_noise_list[i])
        net_r, _ = net_intensity_and_error(right_sums[i], rms_list[i], n_right, bg_mean_list[i], N_noise_list[i])
        I_L_net += net_l
        I_R_net += net_r

    if I_R_net > 0:
        R = I_L_net / I_R_net
        Mz = lens_mass(dt_ms, R)
        print(f"透镜红移质量 M_L(1+z_L) = {Mz:.4e} M_sun, 放大比 R = {R:.4f}")
    else:
        print("右窗口净强度非正，无法计算放大比")
        R = np.nan

    print("\n===== 左窗口（峰值前） =====")
    for i in range(n_bands - 1):
        if not np.isnan(hr_left[i]):
            print(f"HR_{i+1}_{i} = {hr_left[i]:.2f} ± {hr_err_left[i]:.2f}")
        else:
            print(f"HR_{i+1}_{i} 无效（净强度≤0）")

    print("\n===== 右窗口（峰值后） =====")
    for i in range(n_bands - 1):
        if not np.isnan(hr_right[i]):
            print(f"HR_{i+1}_{i} = {hr_right[i]:.2f} ± {hr_err_right[i]:.2f}")
        else:
            print(f"HR_{i+1}_{i} 无效（净强度≤0）")

    # 比较左右窗口硬度比
    all_match, match_results = compare_hardness_ratios(
        hr_left, hr_err_left, hr_right, hr_err_right, n_sigma
    )
    
    print(f"\n===== 硬度比一致性检验 (n_sigma={n_sigma}) =====")
    for i in range(n_bands - 1):
        status = "✓ PASS" if match_results[i] else "✗ FAIL"
        if not np.isnan(hr_left[i]) and not np.isnan(hr_right[i]):
            diff = abs(hr_left[i] - hr_right[i])
            combined_err = np.sqrt(hr_err_left[i]**2 + hr_err_right[i]**2)
            print(f"HR_{i+1}_{i}: {status} (|Δ|={diff:.3f}, {n_sigma}σ={n_sigma*combined_err:.3f})")
        else:
            print(f"HR_{i+1}_{i}: ✗ FAIL (无效数据)")
    
    if all_match:
        print(f"\n>>> 结果: PASS (k={k}, 所有硬度比在 {n_sigma}σ 置信区间内吻合)")
    else:
        print(f"\n>>> 结果: FAIL (k={k}, 存在不吻合的硬度比)")

    print(f"\n噪声统计 (每时间点):")
    for i in range(n_bands):
        print(f"band{i}: RMS={rms_list[i]:.3f}, 均值={bg_mean_list[i]:.3f}, 点数={N_noise_list[i]}")
    print("Peak1 窗口范围: [%d, %d)" % (data['left_boundary1'], data['right_boundary1']))
    print("Peak2 窗口范围: [%d, %d)" % (data['left_boundary2'], data['right_boundary2']))
    print(data['fwhm1'], data['fwhm2'])
    print(f"左窗口点数={n_left}, 右窗口点数={n_right}")
