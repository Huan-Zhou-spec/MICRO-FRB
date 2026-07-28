#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:54:23 2026

@author: ubuntu
"""

import numpy as np
from modules import load_and_clean_data, fpbh
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from modules import (extract_with_fwhm, net_intensity_and_error, 
hardness_ratio, compare_hardness_ratios, lens_mass)
from modules import (
    read_frb_dynamic_spectrum, process_data_ts, compute_autocorr_with_spikes
)

'''
#计算FRB20190131D的红移范围和限制PBH的范围
def zm(dm_wo_mw, c2):
    """
    根据总 DM 减去银河系贡献后估算红移 z。
    公式来源：z = [ (a-b2) + sqrt((b2-a)^2 - 4*b2*(c2-a)) ] / (2*b2)
    其中 a = dm_wo_mw, b2=855, c2=200
    """
    a = dm_wo_mw
    b2 = 855.0
    delta = (b2 - a)**2 - 4 * b2 * (c2 - a)
    if delta < 0:
        return np.nan
    z = ((a - b2) + np.sqrt(delta)) / (2 * b2)
    if z <= 0:
        return 1e-3
    return z

DM_YMW16 = 568.8
zs1 = zm(DM_YMW16, 0)
zs2 = zm(DM_YMW16, 200)
print(zs1, zs2)

Mz = 466.50
M1= Mz/(1+0)
M2= Mz/(1+zs1)
print(M1,M2)

# ==================== 使用示例 ====================
if __name__ == "__main__":
    repeater_file = 'FRB_data/CHIME_cat2_frb/chimefrbcat2_unique_first_repeaters.npy'
    non_repeater_file = 'FRB_data/CHIME_cat2_frb/chimefrbcat2_unique_non_repeater.npy'

    data = load_and_clean_data(repeater_file, non_repeater_file)
    if len(data['DM_wo_mw']) == 0:
        print("没有有效数据，退出")
        exit()

    # 质量取值：从 1 到 1000，对数均匀分布（30 个点）
    ML_values = np.linspace(M1, M2, 2)  # 10^0 = 1 到 10^3 = 1000
    results = []

    print("\n开始计算不同透镜质量对应的 f_up...")
    for ML in ML_values:
        f_up = fpbh(ML, data['DM_wo_mw'], data['wi'], data['snr'])
        results.append([ML, f_up])

    results = np.array(results)
'''    


'''
#重新画动态谱图
def plot_dynamic_spectrum(data_dict, frb_name, peak_indices=None, time_unit='ms',
                          green_peak_indices=None,
                          scale_region_indices=None, scaling_factor=1.0,
                          post_shift=0, label_text="2'"):
    """
    绘制FRB动态谱和时间序列，支持峰值标记、区域强度缩放及平移。

    参数:
    data_dict          : 数据字典，必须包含以下键：
                         'dynamic_spectrum' : 2D数组 (频率×时间)
                         'dm'               : 色散量 (float)
                         'times_relative'   : 相对时间数组 (1D)
                         'frequencies'      : 频率数组 (1D)
                         'ts'               : 时间序列强度 (1D)
    frb_name           : FRB名称 (字符串)
    peak_indices       : 需要标记的峰值索引列表，如 [78,81,85]
    time_unit          : 时间单位，'ms' 或 's' (默认 'ms')
    green_peak_indices : 标记为绿色的峰值索引列表，如 [78,85] (其余红色)
    scale_region_indices : 需要强度缩放并平移的区域索引列表，如 [80,81,82,83]
    scaling_factor       : 强度缩放因子 (>1 缩小强度，<1 放大强度)
    post_shift           : 缩放完成后整体向右平移的采样点步数 (正值向右)
    label_text           : 变换后波形的标注文本，如 "2'"
    """
    # 获取数据
    d_w = data_dict['dynamic_spectrum']
    dm = data_dict['dm']
    t = data_dict['times_relative'].copy()
    freq = data_dict['frequencies']
    ts = data_dict['ts'].copy()

    # 转换时间单位
    if time_unit == 'ms':
        t = t * 1e3

    # 创建图形
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, hspace=0.0, height_ratios=[1, 3])
    ax_ts = plt.subplot(gs[0])   # 时间序列子图
    ax_im = plt.subplot(gs[1])   # 动态谱子图

    # ----- 绘制动态谱 -----
    vmin = np.percentile(d_w, 1)
    vmax = np.percentile(d_w, 99)
    T, F = np.meshgrid(t, freq)
    ax_im.pcolormesh(T, F, d_w, vmin=vmin, vmax=vmax, cmap='viridis')
    ax_im.set_xlabel(f'Time ({time_unit})', fontsize=14)
    ax_im.set_ylabel('Frequency (MHz)', fontsize=14)

    # ----- 绘制原始时间序列 -----
    ax_ts.plot(t, ts, 'k-', lw=1)
    ax_ts.set_xticklabels([])
    ax_ts.set_yticks([])

    # ----- 显示FRB名称和DM值 -----
    xpos = 0.95 * t.max()
    ypos = ts.max() * 0.95
    if dm >= 100:
        dm_str = f"{dm:.0f}"
    elif dm >= 10:
        dm_str = f"{dm:.1f}"
    else:
        dm_str = f"{dm:.2f}"
    ax_ts.text(xpos, ypos, frb_name, ha='right', va='top', fontsize=16)
    ax_ts.text(xpos, 0.7 * ts.max(), f"DM= {dm_str} pc cm⁻³",
               ha='right', va='top', fontsize=13)

    # ----- 标记原始峰值（红/绿点及序号）-----
    if peak_indices is not None:
        green_set = set(green_peak_indices) if green_peak_indices else set()
        for i, idx in enumerate(peak_indices, 1):
            if idx >= len(t):
                continue
            peak_time = t[idx]
            peak_height = ts[idx]
            color = 'green' if idx in green_set else 'red'
            ax_ts.scatter(peak_time, peak_height, color=color, s=3, zorder=5)
            label_y = peak_height + 0.05 * (ts.max() - ts.min())
            ax_ts.text(peak_time, label_y, str(i), ha='center', va='bottom',
                       fontsize=12, fontweight='bold', color=color)

    # ----- 区域强度缩放 + 平移，并直接标注 label_text -----
    if scale_region_indices is not None and len(scale_region_indices) >= 2:
        indices = np.array(scale_region_indices)
        indices = indices[(indices >= 0) & (indices < len(t))]
        if len(indices) < 2:
            print("警告：缩放区域有效点少于2，无法绘制曲线")
        else:
            orig_t = t[indices]
            orig_h = ts[indices]

            # 强度缩放（除以 scaling_factor）
            scaled_h = orig_h / scaling_factor

            # 时间平移（索引偏移 -> 时间偏移）
            dt_sample = t[1] - t[0] if len(t) > 1 else 1.0
            shift_time = post_shift * dt_sample
            shifted_t = orig_t + shift_time

            # 绘制红色虚线（变换后的波形）
            ax_ts.plot(shifted_t, scaled_h, 'r--', lw=1.5, alpha=0.8)

            # 找到变换后波形的最高点，在其上方标注 label_text（如 "2'"）
            max_idx = np.argmax(scaled_h)
            peak_time_label = shifted_t[max_idx]
            peak_height_label = scaled_h[max_idx]
            label_offset = 0.005 * (ts.max() - ts.min())
            ax_ts.text(peak_time_label, peak_height_label + label_offset, label_text,
                       ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')

    plt.tight_layout()
    plt.show()
    return fig

# 您已有的参数
frb_name = 'FRB20190131D'
peak_indices = [76, 85]           # 1,2 号峰
green_peak_indices = [76, 85]          # peak1 和 peak3 用绿色

# 缩放平移参数
Rf = 1.19                              # 缩放因子（压缩时间）
dp_index = round(8.82 / 0.98)

file_path = "FRB_data/canfar_downloads/"+frb_name+"_stokesi_dynamic_spectrum.h5"
# 读取数据
raw_data = read_frb_dynamic_spectrum(file_path)
# 处理数据
proc_data = process_data_ts(raw_data, f_down=32, t_down=1, rfi_factor=3)

# 调用增强后的函数
plot_dynamic_spectrum(
    data_dict=proc_data,
    frb_name=frb_name,
    peak_indices=peak_indices,
    green_peak_indices=green_peak_indices,
    #scale_region_indices=scale_region,
    #scaling_factor=Rf,                 # 时间轴压缩为原来的 1/1.86
    #post_shift=dp_index,               # 向右平移 7 个时间采样点
    #label_text="2'",
    time_unit='ms'
)
'''



#多段Hardness test检测
if __name__ == "__main__":
    dt_ms = 8.82
    k_HR = 7
    n_sigma = 2.0  # 置信度参数: 1.0=68%, 2.0=95%, 3.0=99.7%

    data = extract_with_fwhm(
        frb_name="FRB20190131D",
        peak1_index=76,
        peak2_index=85,
        k=k_HR,
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
        print(f"\n>>> 结果: PASS (k={k_HR}, 所有硬度比在 {n_sigma}σ 置信区间内吻合)")
    else:
        print(f"\n>>> 结果: FAIL (k={k_HR}, 存在不吻合的硬度比)")

    print(f"\n噪声统计 (每时间点):")
    for i in range(n_bands):
        print(f"band{i}: RMS={rms_list[i]:.3f}, 均值={bg_mean_list[i]:.3f}, 点数={N_noise_list[i]}")
    print("Peak1 窗口范围: [%d, %d)" % (data['left_boundary1'], data['right_boundary1']))
    print("Peak2 窗口范围: [%d, %d)" % (data['left_boundary2'], data['right_boundary2']))
    print(data['fwhm1'], data['fwhm2'])
    print(f"左窗口点数={n_left}, 右窗口点数={n_right}")



#画出k条ACF对比图
def compute_band_acf(sub_band, k, smooth_sigma=3, threshold=3,
                     min_lag=1, positive_lags_only=True):
    """
    将频带分为 k 段，计算每段的 ACF 及 total 的 ACF。
    返回: results 列表 (total + k个子带), labels, colors
    """
    groups = np.array_split(sub_band, k, axis=0)
    band_ts = [np.sum(g, axis=0) for g in groups]
    total = np.sum(band_ts, axis=0)

    def _acf(ts):
        return compute_autocorr_with_spikes(
            ts,
            smooth_sigma=smooth_sigma,
            threshold=threshold,
            min_lag=min_lag,
            positive_lags_only=positive_lags_only,
            detect_spikes=True,
            return_details=True,
        )

    results = [_acf(total)] + [_acf(b) for b in band_ts]
    labels = ['Total'] + [f'Band {i+1}' for i in range(k)]

    cmap = plt.cm.jet
    colors = ['black'] + [cmap(i / max(k - 1, 1)) for i in range(k)]

    return results, labels, colors


def plot_multi_autocorr_with_shadow(results, labels, colors, time_step_ms=0.98,
                                    total_index=0, spike_window_ms=2.0,
                                    show_smoothed=True,
                                    figsize=(10, 6), label_fontsize=16,
                                    legend_fontsize=21):
    """
    绘制多条自相关曲线，并为 total 的 spike 添加灰色阴影（±spike_window_ms）。
    其他曲线仅显示落在阴影内的 spike。
    """
    fig, ax = plt.subplots(figsize=figsize)

    total_res = results[total_index]
    total_spike_times = []
    if 'spike_result' in total_res and total_res['spike_result'] is not None:
        total_spike_lags = total_res['spike_result']['spike_lags']
        total_spike_times = [lag * time_step_ms for lag in total_spike_lags]

    for t_spike in total_spike_times:
        ax.axvspan(t_spike - spike_window_ms, t_spike + spike_window_ms,
                   alpha=0.3, color='gray', zorder=0)

    line_handles = []
    has_spike_in_shadow = [False] * len(results)

    for idx, (res, label, color) in enumerate(zip(results, labels, colors)):
        autocorr = res['autocorr']
        lags = res['lags']
        time_lags = lags * time_step_ms

        ln, = ax.plot(time_lags, autocorr, color=color, lw=2, label=label)
        line_handles.append(ln)

        if show_smoothed and 'spike_result' in res and res['spike_result'] is not None:
            sr = res['spike_result']
            x_smooth = sr['lags_analyzed'] * time_step_ms
            y_smooth = sr['autocorr_smoothed']
            ax.plot(x_smooth, y_smooth, '--', color=color, alpha=1, lw=2)

        if 'spike_result' in res and res['spike_result'] is not None:
            spike_lags = res['spike_result']['spike_lags']
            if idx == total_index:
                for lag in spike_lags:
                    lag_time = lag * time_step_ms
                    pos = np.where(lags == lag)[0]
                    if len(pos) == 0:
                        continue
                    yval = autocorr[pos[0]]
                    ax.axvline(lag_time, color=color, linestyle=':', alpha=1, lw=2)
                    ax.plot(lag_time, yval, 'o', color=color, markersize=10,
                            markeredgecolor='white', markeredgewidth=1)
            else:
                for lag in spike_lags:
                    lag_time = lag * time_step_ms
                    in_window = any(abs(lag_time - t_total) <= spike_window_ms
                                    for t_total in total_spike_times)
                    if in_window:
                        has_spike_in_shadow[idx] = True
                        pos = np.where(lags == lag)[0]
                        if len(pos) == 0:
                            continue
                        yval = autocorr[pos[0]]
                        ax.plot(lag_time, yval, 'o', color=color, markersize=10,
                                markeredgecolor='white', markeredgewidth=0.5)

    handles = []
    leg_labels = []
    total_has_spike = len(total_spike_times) > 0

    for idx, (ln, label, color) in enumerate(zip(line_handles, labels, colors)):
        handles.append(ln)
        leg_labels.append(label)

        if idx == total_index and total_has_spike:
            circle_handle = plt.Line2D([0], [0], color=color, marker='o',
                                       linestyle='None', markersize=10,
                                       markeredgecolor='white', markeredgewidth=1)
            handles.append(circle_handle)
            leg_labels.append(f"{label} spike")
        elif idx != total_index and has_spike_in_shadow[idx]:
            circle_handle = plt.Line2D([0], [0], color=color, marker='o',
                                       linestyle='None', markersize=10,
                                       markeredgecolor='white', markeredgewidth=0.5)
            handles.append(circle_handle)
            leg_labels.append(f"{label} spike")

    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Time Delay (ms)', fontsize=label_fontsize)
    ax.set_ylabel('Autocorrelation', fontsize=label_fontsize)
    ax.set_xlim(-0.1, 100)
    ax.set_ylim(-0.25, 1.05)
    ax.tick_params(labelsize=label_fontsize - 2)
    ax.legend(handles, leg_labels, fontsize=legend_fontsize, loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def main():
    frb_name = 'FRB20190131D'
    k_ACF = k_HR+2
    file_path = f"FRB_data/canfar_downloads/{frb_name}_stokesi_dynamic_spectrum.h5"
    raw_data = read_frb_dynamic_spectrum(file_path)
    proc_data = process_data_ts(raw_data, f_down=32, t_down=1, rfi_factor=3)
    freq_axis = proc_data['frequencies']
    dynamic = proc_data['dynamic_spectrum']

    chime_file = 'FRB_data/CHIME_cat2_frb/chimefrbcat2.npy'
    chime_data = np.load(chime_file, allow_pickle=True)
    entry = next((e for e in chime_data if e['tns_name'] == frb_name), None)
    if entry is None:
        raise ValueError(f"未找到 {frb_name} 对应的 CHIME 数据")
    low_freq, high_freq = entry['low_freq'], entry['high_freq']

    idx_low = np.argmin(np.abs(freq_axis - low_freq))
    idx_high = np.argmin(np.abs(freq_axis - high_freq))
    if idx_low > idx_high:
        idx_low, idx_high = idx_high, idx_low
    sub_band = dynamic[idx_low:idx_high+1, :]

    results, labels, colors = compute_band_acf(
        sub_band, k=k_ACF,
        smooth_sigma=3, threshold=3,
        min_lag=1, positive_lags_only=True
    )

    fig = plot_multi_autocorr_with_shadow(
        results, labels, colors,
        time_step_ms=0.98,
        spike_window_ms=2.0,
        show_smoothed=True,
        figsize=(10, 8)
    )


if __name__ == "__main__":
    main()