#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 13:12:31 2026

@author: ubuntu
"""

import numpy as np
from scipy.signal import find_peaks, argrelextrema
from scipy.stats import ks_2samp, gaussian_kde
from scipy.stats import gaussian_kde
import warnings
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


#对数据的基本处理分为：1.合并数据的部分频率函数downsample；
#2.去除RFI掩码函数mask_rfi，受干扰区域用0替代；
#3.得到噪声区域以及对应的噪声谱函数extract_noise_spectra，用于后续寻找峰值、计算峰值SNR、K-S检验、自相关检验等；
#4.寻找FRB的信号峰值函数detect_peaks_robust_raw，用于后续分析。
def downsample(data, f_down, t_down):
    """对动态谱数据进行下采样"""
    f_step = f_down
    t_step = t_down
    
    # 频率方向下采样
    d_new_freq = np.zeros((data.shape[0]//f_step, data.shape[1]))
    for f in range(data.shape[0]//f_step):
        d_new_freq[f, :] = data[f*f_step:(f+1)*f_step, :].sum(axis=0)
    
    # 时间方向下采样
    d_new = np.zeros((data.shape[0]//f_step, data.shape[1]//t_step))
    for t in range(data.shape[1]//t_step):
        d_new[:, t] = d_new_freq[:, t*t_step:(t+1)*t_step].sum(axis=1)
    
    return d_new

def mask_rfi(wfall, rfi_factor=3):
    """Removing the Radio Frequency Interference，应用RFI掩码，受干扰区域用0替代"""
    # 计算频谱
    spec = np.sum(wfall, axis=1)
    
    # 计算四分位距
    q1 = np.quantile(spec, 0.25)
    q3 = np.quantile(spec, 0.75)
    iqr = q3 - q1
    
    # 计算通道方差
    channel_var = np.var(wfall, axis=1)
    mean_var = np.mean(channel_var)
    
    # 创建RFI掩码
    rfi_mask = (channel_var > rfi_factor * mean_var) | \
               (spec < q1 - 1.5 * iqr) | \
               (spec > q3 + 1.5 * iqr)
    
    # 用0替换受干扰区域
    wfall_masked = wfall.copy()
    wfall_masked[rfi_mask, :] = 0
    
    # 重新计算频谱和时间序列
    spec_masked = np.sum(wfall_masked, axis=1)
    ts_masked = np.sum(wfall_masked, axis=0)
    
    return wfall_masked, spec_masked, ts_masked, rfi_mask


def process_data_ts(data, f_down=32, t_down=1, rfi_factor=3, apply_downsampling=True):
    """用于处理FRB数据时域上的数据
    参数：
        data: 包含 'dynamic_spectrum', 'times_relative', 'frequencies', 'dm' 的字典
        f_down: 频率下采样因子，仅在 apply_downsampling=True 时有效
        t_down: 时间下采样因子，仅在 apply_downsampling=True 时有效
        rfi_factor: RFI 掩码阈值因子
        apply_downsampling: 是否应用下采样，若为 False 则直接使用原始分辨率数据
    """
    # 获取原始数据
    d_w = data['dynamic_spectrum']
    t = data['times_relative']
    freq = data['frequencies']
    dm = data['dm']
    
    # 处理NaN值
    d_w = np.nan_to_num(d_w)
    
    # 应用RFI掩码（必须应用）
    d_w_masked, spec, ts, rfi_mask = mask_rfi(d_w, rfi_factor)
    
    # 打印RFI掩码信息
    print(f"RFI掩码: 移除了 {np.sum(rfi_mask)} 个通道")
    
    if apply_downsampling:
        # 应用下采样
        d_w_down = downsample(d_w_masked, f_down, t_down)
        # 重新生成坐标轴（假设原始坐标是均匀的，否则会丢失原始刻度信息）
        freq_down = np.linspace(freq[0], freq[-1], d_w_down.shape[0])
        t_down_axis = np.linspace(t[0], t[-1], d_w_down.shape[1])
        print(f"原始形状: {d_w.shape} -> 下采样后: {d_w_down.shape}")
    else:
        # 不下采样，直接使用原始数据
        d_w_down = d_w_masked
        freq_down = freq
        t_down_axis = t
        print(f"使用原始分辨率数据，形状: {d_w.shape}")

    return {
        'dynamic_spectrum': d_w_down,
        'times_relative': t_down_axis,
        'frequencies': freq_down,
        'spec': spec,
        'ts': ts,
        'rfi_mask': rfi_mask,
        'dm': dm
    }


def extract_noise_spectra(proc_data,
                          n_noise=30,
                          noise_method='quantile',
                          quantile_threshold=0.25,
                          peak_prominence=None,
                          peak_window_width=5,
                          n_sigma=3.0,          # 新增：MAD 法的阈值倍数
                          allow_oversample=True):
    """
    从处理后的FRB数据中提取指定数量的噪声频谱。

    参数:
        proc_data : dict
            process_data_ts 返回的数据字典，必须包含：
            - 'ts' : 一维时间序列 (n_time)
            - 'dynamic_spectrum' : 二维数组 (n_freq, n_time)
            - 'frequencies' : 一维频率轴 (n_freq)
        n_noise : int
            需要提取的噪声频谱数量。
        noise_method : str
            'quantile' : 基于时间序列强度分位数（取强度低于 quantile_threshold 的bin）。
            'multi_peak_window' : 自动检测所有显著峰值，排除每个峰周围窗口。
            'mad' : 基于中位数绝对偏差（MAD）估计噪声标准差，选择强度低于 median + n_sigma * sigma 的bin。
        quantile_threshold : float
            仅当 noise_method='quantile' 时有效，强度分位数阈值（0~1）。
        peak_prominence : float or None
            仅当 noise_method='multi_peak_window' 时有效，find_peaks 的 prominence 参数。
            若为None，则自动设为时间序列标准差的3倍。
        peak_window_width : int
            每个峰周围排除的窗口宽度（单边），即排除 [peak_idx-width, peak_idx+width]。
        n_sigma : float
            仅当 noise_method='mad' 时有效，阈值倍数。噪声候选为 ts < median + n_sigma * sigma。
        random_seed : int
            随机种子，保证随机选择可重复。
        allow_oversample : bool
            如果 True，当检测到的噪声bin少于 n_noise 时，通过有放回抽样补足；
            如果 False，则当检测到的噪声bin少于 n_noise 时引发 ValueError。

    返回:
        dict 包含：
            'noise_spectra' : list of 1D numpy arrays，长度为 n_noise（除非 allow_oversample=False 且实际噪声bin数少于 n_noise，此时返回全部检测到的噪声）。
            'noise_indices' : list of int，对应的时间索引。
            'noise_mask' : bool 数组，所有时间bin的噪声掩码（基于原始检测）。
            'method' : 使用的方法。
            'sigma_noise' : 仅当 method='mad' 时返回估计的噪声标准差。
    """

    # 提取数据
    ts = proc_data['ts']
    dynamic = proc_data['dynamic_spectrum']
    n_time = len(ts)

    # 初始化噪声掩码
    noise_mask = np.ones(n_time, dtype=bool)

    if noise_method == 'quantile':
        threshold = np.quantile(ts, quantile_threshold)
        noise_mask = ts < threshold

    elif noise_method == 'multi_peak_window':
        if peak_prominence is None:
            peak_prominence = 3 * np.std(ts)
        peaks, _ = find_peaks(ts, prominence=peak_prominence)
        if len(peaks) == 0:
            print("警告: 未检测到显著峰值，将使用全时间序列作为噪声（可能包含信号）。")
        else:
            for p in peaks:
                start = max(0, p - peak_window_width)
                end = min(n_time, p + peak_window_width + 1)
                noise_mask[start:end] = False

    elif noise_method == 'mad':
        # 计算中位数和 MAD
        med = np.median(ts)
        mad = np.median(np.abs(ts - med))
        sigma_noise = 1.4826 * mad
        # 设置阈值：只保留低于中位数 + n_sigma * sigma 的点（单侧，因为 FRB 信号是正尖峰）
        threshold = med + n_sigma * sigma_noise
        noise_mask = ts < threshold
        #noise_mask = abs(ts - med) < n_sigma * sigma_noise
        # 可选：若需要更严格的对称剔除，可以用 abs(ts - med) < n_sigma * sigma_noise
        # 但通常 FRB 信号为正，单侧即可
        # 记录估计的 sigma 供返回
        estimated_sigma = sigma_noise

    else:
        raise ValueError("noise_method 必须是 'quantile', 'multi_peak_window' 或 'mad'")

    noise_indices = np.where(noise_mask)[0]
    n_found = len(noise_indices)

    # 处理数量不足的情况
    if n_found < n_noise:
        if allow_oversample:
            # 有放回抽样补足
            selected_indices = np.random.choice(noise_indices, size=n_noise, replace=True)
            print(f"警告: 检测到 {n_found} 个噪声bin，少于所需 {n_noise}，使用有放回抽样补足。")
        else:
            # 返回全部，并给出警告
            print(f"警告: 检测到 {n_found} 个噪声bin，少于所需 {n_noise}，返回全部（可能无法满足精确数量要求）。")
            selected_indices = noise_indices
    else:
        # 从噪声索引中随机选择 n_noise 个（无放回）
        selected_indices = np.random.choice(noise_indices, size=n_noise, replace=False)

    # 提取对应的频谱
    selected_spectra = [dynamic[:, idx] for idx in selected_indices]

    # 构建返回结果
    result = {
        'noise_spectra': selected_spectra,
        'noise_indices': selected_indices.tolist() if isinstance(selected_indices, np.ndarray) else selected_indices,
        'noise_mask': noise_mask,
        'method': noise_method
    }
    if noise_method == 'mad':
        result['sigma_noise'] = estimated_sigma

    return result


def detect_peaks_robust_raw(ts,
                            main_prominence_factor=3.0,
                            main_height_percentile=75,
                            sub_height_ratio=0.2,
                            sub_snr_threshold=2.0,
                            neighbor_radius_factor=2.0,
                            min_prominence_global=1.0,
                            valley_depth_ratio=0.2,          # 相邻峰合并的谷深阈值
                            merge_distance=1):
    """
    在原始数据上鲁棒地检测主峰及其次峰，最后通过相邻峰谷深比例合并。

    新增参数:
        valley_depth_ratio: 相邻峰合并的谷深比例阈值。如果两峰之间的谷深度相对于较低峰的比例小于此值，则合并两峰（保留较高者）。
    """
    ts = np.asarray(ts).flatten()
    n = len(ts)
   
    # ---------- 1. 噪声估计（百分位数方法） ----------
    threshold = np.percentile(ts, 25)          # 低于25%分位数的点视为噪声
    noise_data = ts[ts < threshold]
    noise_std = np.std(noise_data) if len(noise_data) > 1 else 1e-6
    
    # ---------- 2. 找到所有局部极小值（谷）用于后续合并 ----------
    minima = argrelextrema(ts, np.less)[0]
    # 方便起见，将端点也作为潜在谷（用于边界情况）
    minima = np.unique(np.concatenate(([0], minima, [n-1])))

    # ---------- 3. 检测所有峰（并计算突出度） ----------
    all_peaks, properties = find_peaks(ts, prominence=0, height=None, distance=1)
    if len(all_peaks) == 0:
        return []
    all_heights = ts[all_peaks]
    all_prominences = properties['prominences']

    # ---------- 4. 筛选主峰 ----------
    height_threshold = np.percentile(all_heights, main_height_percentile)
    prom_threshold = main_prominence_factor * noise_std

    main_candidates = []
    for i, p in enumerate(all_peaks):
        if all_heights[i] >= height_threshold and all_prominences[i] >= prom_threshold:
            main_candidates.append(p)

    # 如果主峰候选为空，则退而求其次：保留突出度最高的几个峰
    if len(main_candidates) == 0:
        sorted_idx = np.argsort(all_prominences)[::-1]
        num_main = max(1, min(5, len(all_peaks)//10))
        main_candidates = [all_peaks[sorted_idx[i]] for i in range(num_main)]

    main_peaks = sorted(main_candidates)

    # ---------- 5. 为每个主峰估计半高宽并确定邻域半径 ----------
    def estimate_fwhm(peak_idx):
        half_height = ts[peak_idx] / 2
        left = peak_idx
        while left > 0 and ts[left] > half_height:
            left -= 1
        right = peak_idx
        while right < n-1 and ts[right] > half_height:
            right += 1
        return right - left

    # ---------- 6. 收集所有候选峰（主峰+次峰） ----------
    detected_set = set(main_peaks)

    for main_peak in main_peaks:
        fwhm = estimate_fwhm(main_peak)
        radius = int(fwhm * neighbor_radius_factor)
        radius = max(radius, 5)

        left = max(0, main_peak - radius)
        right = min(n, main_peak + radius + 1)

        # 在邻域内检测所有局部极大值
        segment = ts[left:right]
        seg_indices = np.arange(left, right)
        local_peaks, _ = find_peaks(segment, prominence=0, height=None, distance=1)
        local_peaks = seg_indices[local_peaks]

        # 排除主峰自身
        local_peaks = [p for p in local_peaks if p != main_peak]

        for p in local_peaks:
            # 高度条件
            if ts[p] < ts[main_peak] * sub_height_ratio:
                continue

            # 局部信噪比计算
            bg_left = max(left, p - radius//2)
            bg_right = min(right, p + radius//2 + 1)
            bg_indices = np.arange(bg_left, bg_right)
            bg_indices = bg_indices[abs(bg_indices - p) > 2]
            if len(bg_indices) < 5:
                bg_indices = np.arange(bg_left, bg_right)
            bg_data = ts[bg_indices]
            bg_mean = np.mean(bg_data)
            bg_std = np.std(bg_data)
            if bg_std == 0:
                bg_std = 1e-6
            snr = (ts[p] - bg_mean) / bg_std

            if snr >= sub_snr_threshold:
                # 全局最小突出度检查
                local_min = np.min(ts[bg_left:bg_right])
                prominence = ts[p] - local_min
                if prominence >= min_prominence_global * noise_std:
                    detected_set.add(p)

    # ---------- 7. 相邻峰谷深比例合并 ----------
    # 将候选峰排序
    peaks_list = sorted(detected_set)
    
    # 辅助函数：找到两个峰之间的最低点（谷）
    def valley_between(p1, p2):
        # 取 p1 和 p2 之间的局部极小值中最低的点，若没有则取区间内最低点
        between_minima = minima[(minima > p1) & (minima < p2)]
        if len(between_minima) > 0:
            # 选择其中值最小的极小值点
            valley_idx = between_minima[np.argmin(ts[between_minima])]
        else:
            # 区间内直接找最低点（可能不是严格极小值，但作为谷的近似）
            interval = slice(p1, p2+1)
            valley_idx = p1 + np.argmin(ts[interval])
        return valley_idx

    # 迭代合并
    merged = True
    while merged and len(peaks_list) > 1:
        merged = False
        new_list = []
        i = 0
        while i < len(peaks_list):
            if i == len(peaks_list) - 1:
                new_list.append(peaks_list[i])
                break
            p1 = peaks_list[i]
            p2 = peaks_list[i+1]
            # 计算两峰之间的谷
            v_idx = valley_between(p1, p2)
            v_val = ts[v_idx]
            h1 = ts[p1]
            h2 = ts[p2]
            lower_h = min(h1, h2)
            # 谷深比例 = (较低峰高 - 谷值) / 较低峰高
            if lower_h > 0:
                depth = (lower_h - v_val) / lower_h
            else:
                depth = 0
            if depth < valley_depth_ratio:
                # 合并，保留较高的峰
                if h1 >= h2:
                    new_list.append(p1)
                else:
                    new_list.append(p2)
                i += 2  # 跳过 p2
                merged = True
            else:
                new_list.append(p1)
                i += 1
        peaks_list = new_list

    # ---------- 8. 最终合并距离过近的峰（可选，但可能已被谷深合并覆盖） ----------
    # 这里仍然保留 merge_distance 作为最后的去重，避免非常接近的峰（例如由于噪声导致的重复）
    #暂时没用，因为选择了谷深合并覆盖足够了
    final_peaks = []
    i = 0
    while i < len(peaks_list):
        j = i + 1
        group = [peaks_list[i]]
        while j < len(peaks_list) and peaks_list[j] - peaks_list[i] <= merge_distance:
            group.append(peaks_list[j])
            j += 1
        best = max(group, key=lambda idx: ts[idx])
        final_peaks.append(best)
        i = j

    return final_peaks



#进行进一步分析数据，包括：
#1.计算峰值信噪比函数calculate_snr_peaks；
#2.提取峰值位置的频率谱线函数extract_peak_spectra，用于后续的K-S检验函数compare_spectra_ks，
#最终确立严重漂移峰对；
#3.用于计算得到ts信号中强自相关函数detect_autocorr_spikes和compute_autocorr_with_spikes。
def calculate_snr_peaks(ts, n=None, adaptive=False, **kwargs):
    """
    使用鲁棒峰检测算法检测峰，并计算每个峰的 SNR。

    参数:
        ts : array_like
            输入时间序列（一维数组）
        noise_percentile : float, 默认 20
            用于估计噪声的百分位数，数据中小于该分位数的点视为噪声
        n : int or None, 默认 None
            若 adaptive=False:
                若指定正整数，则只返回 SNR 最高的前 n 个峰（按时间顺序输出）；
                若为 None 或大于等于总峰数，则返回所有峰；
                若 n <= 0，返回空列表。
            若 adaptive=True:
                n 表示期望的峰数，函数将根据自适应规则动态选择峰的数量，
                规则如下：
                - 若总峰数 M ≤ n，则返回所有峰。
                - 否则，取前 n+1 个峰（按 SNR 降序），并检查第 n+1 个峰的 SNR：
                  - 若其 ≥ 0.3 × 最高 SNR，则返回前 n+1 个峰；
                  - 否则，检查第 n 个峰的 SNR：
                    - 若其 ≥ 0.1 × 最高 SNR，则返回前 n 个峰；
                    - 否则，返回前 n-1 个峰（至少保留 2 个，即 max(2, n-1)）。
        adaptive : bool, default False
            是否使用自适应峰数选择规则。当 adaptive=True 且 n>0 时生效。
        **kwargs : 
            传递给 detect_peaks_robust_raw 的其他参数，例如：
            main_prominence_factor, valley_depth_ratio, sub_height_ratio 等

    返回:
        dict : 包含以下字段的字典
            snr_values : list
                每个峰的 SNR 值（按时间顺序）
            peak_indices : list
                峰在原始序列中的索引（按时间顺序）
            peak_values : list
                峰的高度值（按时间顺序）
            noise_mean : float
                噪声均值
            noise_std : float
                噪声标准差
            noise_percentile : float
                使用的噪声百分位数
            num_noise_points : int
                用于估计噪声的数据点数
            num_peaks_found : int
                实际返回的峰的数量
            max_snr : float
                返回的峰中最大的 SNR 值
            avg_snr : float
                返回的峰的平均 SNR 值
    """
    import numpy as np
    # 确保输入为一维数组
    ts = np.asarray(ts).flatten()
    
    # 1. 调用鲁棒峰检测函数获取所有峰索引
    peaks = detect_peaks_robust_raw(ts, **kwargs)
    
    # 2. 基于相同百分位数估计噪声
    threshold = np.percentile(ts, 25)          # 低于25%分位数的点视为噪声
    noise_data = ts[ts < threshold]
    noise_mean = np.mean(noise_data)
    noise_std = np.std(noise_data) if len(noise_data) > 1 else 1e-6
    
    
    # 3. 按时间顺序整理峰
    peaks_sorted = sorted(peaks)
    peak_values = ts[peaks_sorted]
    
    # 4. 计算每个峰的 SNR
    snr_values = [(val - noise_mean) / noise_std for val in peak_values]
    
    total_peaks = len(peaks_sorted)
    
    # 5. 处理峰选择逻辑
    if adaptive and n is not None and n > 0:
        # 自适应选择逻辑
        if total_peaks == 0:
            selected_snr = []
            selected_indices = []
            selected_peak_values = []
            num_returned = 0
        else:
            # 按 SNR 降序排序
            candidates = list(zip(snr_values, peaks_sorted))
            candidates_sorted = sorted(candidates, key=lambda x: x[0], reverse=True)
            max_snr = candidates_sorted[0][0]
            M = total_peaks
            
            if M <= n:
                k = M
            else:
                # 检查第 n+1 个峰（索引 n）的 SNR
                if candidates_sorted[n][0] >= 0.3 * max_snr:
                    k = min(n + 1, M)
                else:
                    # 检查第 n 个峰（索引 n-1）的 SNR
                    if candidates_sorted[n-1][0] >= 0.1 * max_snr:
                        k = n
                    else:
                        k = max(2, n - 1)
                        k = min(k, M)
            
            # 取前 k 个峰，按时间顺序排序
            top_candidates = candidates_sorted[:k]
            top_candidates_sorted_by_time = sorted(top_candidates, key=lambda x: x[1])
            selected_snr = [item[0] for item in top_candidates_sorted_by_time]
            selected_indices = [item[1] for item in top_candidates_sorted_by_time]
            selected_peak_values = [ts[i] for i in selected_indices]
            num_returned = k
    else:
        # 原有的简单选择逻辑
        if n is not None and n > 0 and n < total_peaks:
            sorted_indices = np.argsort(snr_values)[::-1]
            top_snr_indices = sorted_indices[:n]
            top_peak_indices_unsorted = [peaks_sorted[i] for i in top_snr_indices]
            top_peak_indices = sorted(top_peak_indices_unsorted)
            final_positions = [peaks_sorted.index(idx) for idx in top_peak_indices]
            selected_snr = [snr_values[i] for i in final_positions]
            selected_indices = top_peak_indices
            selected_peak_values = [ts[i] for i in top_peak_indices]
            num_returned = n
        else:
            if n is not None and n <= 0:
                selected_snr = []
                selected_indices = []
                selected_peak_values = []
                num_returned = 0
            else:
                selected_snr = snr_values
                selected_indices = peaks_sorted
                selected_peak_values = peak_values.tolist()
                num_returned = total_peaks
    
    # 6. 构造返回字典
    result = {
        'snr_values': selected_snr,
        'peak_indices': selected_indices,
        'peak_values': selected_peak_values,
        'noise_mean': noise_mean,
        'noise_std': noise_std,
        'num_noise_points': len(noise_data),
        'num_peaks_found': num_returned,
        'max_snr': max(selected_snr) if selected_snr else 0,
        'avg_snr': np.mean(selected_snr) if selected_snr else 0
    }
    return result


def extract_peak_spectra(processed_data, peak_idxs, time_window=2, aggregation='mean'):
    """
    提取峰值附近时间窗口内的信号强度，按频率函数输出。

    参数:
    processed_data : dict
        process_data_ts 返回的字典，包含以下键：
        - 'dynamic_spectrum': 2D 数组 (频率 × 时间)
        - 'frequencies': 1D 频率轴
        - 'times_relative': 1D 时间轴
    peaks_result : dict
        calculate_snr_peaks 返回的字典，包含键 'peak_indices' (时间索引列表)
    time_window : int or None, 默认 2
        以峰为中心的单侧时间窗口大小（样本数）。总窗口宽度 = 2*time_window + 1。
        若设为 None，则表示只取峰值点本身（即窗口大小为1）。
    aggregation : str 或 None, 默认 'mean'
        沿时间轴的聚合方式：
        - 'mean' : 返回每个频率上时间窗口内的平均强度
        - 'sum'  : 返回每个频率上时间窗口内的总强度
        - None   : 返回完整的二维窗口数据（频率 × 时间窗口长度）

    返回:
    dict
        包含以下键：
        - 'peak_spectra' : 列表，每个元素对应一个峰的频谱数据。
          若 aggregation 不为 None，则每个元素为 1D 数组（长度 = 频率数）；
          若 aggregation 为 None，则每个元素为 2D 数组（频率 × 窗口长度）。
        - 'frequencies'  : 1D 频率轴（与输入 processed_data 中的相同）
        - 'peak_indices' : 各峰在原始时间轴上的索引（整数列表）
        - 'peak_times'   : 各峰对应的时间值（浮点数列表）
        - 'window_indices' : 若 aggregation 为 None，返回每个峰窗口的时间索引范围（列表，每个元素为 [start, end)）
        - 'window_times'   : 若 aggregation 为 None，返回每个峰窗口的时间值（列表，每个元素为 1D 数组）
    """
    # 提取数据
    dyn_spec = processed_data['dynamic_spectrum']
    freqs = processed_data['frequencies']
    times = processed_data['times_relative']
    #peak_idxs = peaks_result['peak_indices']

    n_freq, n_time = dyn_spec.shape
    peak_times = [times[i] for i in peak_idxs if 0 <= i < n_time]

    peak_spectra = []
    window_indices = []
    window_times_list = []

    for idx in peak_idxs:
        if idx < 0 or idx >= n_time:
            continue
        if time_window is None:
            # 只取峰值点本身
            start = idx
            end = idx + 1
        else:
            start = max(0, idx - time_window)
            end = min(n_time, idx + time_window + 1)
        window_len = end - start
        window_data = dyn_spec[:, start:end]

        if aggregation is None:
            peak_spectra.append(window_data)
            window_indices.append((start, end))
            window_times_list.append(times[start:end])
        else:
            if aggregation == 'mean':
                agg_data = np.mean(window_data, axis=1)
            elif aggregation == 'sum':
                agg_data = np.sum(window_data, axis=1)
            else:
                raise ValueError("aggregation 参数必须是 'mean', 'sum' 或 None")
            peak_spectra.append(agg_data)

    result = {
        'peak_spectra': peak_spectra,
        'frequencies': freqs,
        'peak_indices': peak_idxs,
        'peak_times': peak_times,
    }
    if aggregation is None:
        result['window_indices'] = window_indices
        result['window_times'] = window_times_list

    return result


def compare_spectra_ks(spectra_list, freq_axis=None, n_samples=10000, random_seed=42,
                       p_adjust=None, use_kde=True, bw_method='scott',
                       bootstrap=False, n_bootstrap=1000, bootstrap_method='add_noise',
                       noise_samples=None, return_extra=False):
    """
    对多个频谱进行两两K-S检验，返回p值矩阵和D统计量矩阵。
    支持Bootstrap估计D统计量的误差（标准差、置信区间）。

    参数:
        spectra_list : list of 1D numpy arrays
            每个元素是一个频谱的强度数组（长度需相同）。
        freq_axis : 1D numpy array, optional
            频率轴（与频谱长度一致）。若为None，则假设频率为等间隔的整数索引。
        n_samples : int
            从每个频谱中抽样的样本数，用于K-S检验。
        random_seed : int or None
            随机种子，保证结果可重复。
        p_adjust : str or None
            多重比较校正方法，可选 'bonferroni', 'fdr_bh'。
        use_kde : bool
            是否使用核密度估计（KDE）平滑抽样。若False，则直接按强度归一化离散抽样。
        bw_method : str, scalar or callable, optional
            KDE的带宽选择方法，仅在use_kde=True时有效。
        bootstrap : bool or int, default False
            若为 True 或整数，则执行 Bootstrap 误差估计。整数指定 Bootstrap 次数（默认 n_bootstrap）。
        n_bootstrap : int, default 1000
            Bootstrap 重复次数（当 bootstrap=True 时有效）。
        bootstrap_method : {'add_noise', 'sample'}, default 'add_noise'
            'add_noise'：使用噪声样本加扰生成 Bootstrap 样本（需要 noise_samples）。
            'sample'：对抽样样本进行有放回重采样（无需噪声样本）。
        noise_samples : list of 1D numpy arrays, optional
            噪声样本列表（长度需与频谱相同）。当 bootstrap_method='add_noise' 时必须提供。
        return_extra : bool, default False
            若为 True，则在返回字典中包含 'samples_list'（抽样样本）和
            'bootstrap_D_samples'（Bootstrap D统计量样本列表，当 bootstrap=True 时）。

    返回:
        dict 包含以下键：
            'p_matrix' : (N,N) p值矩阵（对角为1）
            'D_matrix' : (N,N) K-S统计量矩阵（对角为0）
            若 bootstrap=True 则额外包含：
                'D_error' : (N,N) D统计量的标准差矩阵
                'D_ci_lower' : (N,N) 95%置信区间下限矩阵
                'D_ci_upper' : (N,N) 95%置信区间上限矩阵
            若 return_extra=True 则额外包含：
                'samples_list' : list of arrays, 每个频谱的抽样样本
                'bootstrap_D_samples' : list of lists of lists, 每对频谱的Bootstrap D样本列表
    """
    n_spectra = len(spectra_list)
    if n_spectra == 0:
        result = {'p_matrix': np.array([]), 'D_matrix': np.array([])}
        if return_extra:
            result['samples_list'] = []
        if bootstrap:
            result['D_error'] = np.array([])
            result['D_ci_lower'] = np.array([])
            result['D_ci_upper'] = np.array([])
            if return_extra:
                result['bootstrap_D_samples'] = []
        return result

    # 检查频谱长度一致性
    lengths = [len(s) for s in spectra_list]
    if len(set(lengths)) != 1:
        raise ValueError("所有频谱的长度必须相同。若不同，请先插值到公共频率轴。")
    L = lengths[0]

    # 频率轴处理
    if freq_axis is None:
        freq_axis = np.arange(L)
    else:
        if len(freq_axis) != L:
            raise ValueError("freq_axis 的长度必须与每个频谱长度一致。")

    # 随机种子
    if random_seed is not None:
        np.random.seed(random_seed)

    # 处理 bootstrap 参数
    if bootstrap is True:
        bootstrap = n_bootstrap
    elif isinstance(bootstrap, int) and bootstrap > 0:
        n_bootstrap = bootstrap
        bootstrap = True
    else:
        bootstrap = False

    # Bootstrap 方法验证
    if bootstrap and bootstrap_method == 'add_noise':
        if noise_samples is None:
            raise ValueError("bootstrap_method='add_noise' 需要提供 noise_samples")
        # 检查噪声样本长度
        for ns in noise_samples:
            if len(ns) != L:
                raise ValueError("所有噪声样本长度必须与频谱相同")
    elif bootstrap and bootstrap_method not in ['add_noise', 'sample']:
        raise ValueError("bootstrap_method 必须是 'add_noise' 或 'sample'")

    # 定义从频谱抽样函数
    def sample_from_spectrum(intensity, method='discrete'):
        intensity = np.maximum(intensity, 0)
        total = np.sum(intensity)
        if total == 0:
            warnings.warn("频谱总强度为零，使用均匀抽样。")
            prob = np.ones_like(intensity) / len(intensity)
        else:
            prob = intensity / total

        if method == 'discrete':
            samples = np.random.choice(freq_axis, size=n_samples, p=prob)
        elif method == 'kde':
            kde = gaussian_kde(freq_axis, weights=prob, bw_method=bw_method)
            samples = kde.resample(n_samples).flatten()
        else:
            raise ValueError("method 必须是 'discrete' 或 'kde'")
        return samples

    # 对每个原始频谱抽样
    method = 'kde' if use_kde else 'discrete'
    samples_list = [sample_from_spectrum(intensity, method=method) for intensity in spectra_list]

    # 初始化矩阵
    p_matrix = np.ones((n_spectra, n_spectra))
    D_matrix = np.zeros((n_spectra, n_spectra))

    # Bootstrap 相关变量
    D_error = None
    D_ci_lower = None
    D_ci_upper = None
    bootstrap_D_samples = None
    if bootstrap:
        D_error = np.zeros((n_spectra, n_spectra))
        D_ci_lower = np.zeros((n_spectra, n_spectra))
        D_ci_upper = np.zeros((n_spectra, n_spectra))
        if return_extra:
            bootstrap_D_samples = [[[] for _ in range(n_spectra)] for __ in range(n_spectra)]

    # 主循环：计算每对频谱
    for i in range(n_spectra):
        for j in range(i+1, n_spectra):
            # 原始观测统计量
            stat_obs, p_val = ks_2samp(samples_list[i], samples_list[j])
            D_matrix[i, j] = stat_obs
            D_matrix[j, i] = stat_obs
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val

            # Bootstrap 误差估计
            if bootstrap:
                if bootstrap_method == 'add_noise':
                    boot_stats = []
                    for _ in range(n_bootstrap):
                        # 随机选取两个噪声样本（可重复）
                        idx1 = np.random.randint(0, len(noise_samples))
                        idx2 = np.random.randint(0, len(noise_samples))
                        n1 = noise_samples[idx1]
                        n2 = noise_samples[idx2]
                        # 加噪
                        a_raw = n1
                        b_raw = n2
                        # 归一化（零均值、单位标准差）
                        a_norm = (a_raw - np.mean(a_raw)) / (np.std(a_raw) + 1e-12)
                        b_norm = (b_raw - np.mean(b_raw)) / (np.std(b_raw) + 1e-12)
                        # 从归一化加噪频谱中抽样
                        a_samples = sample_from_spectrum(a_norm, method=method)
                        b_samples = sample_from_spectrum(b_norm, method=method)
                        stat_boot, _ = ks_2samp(a_samples, b_samples)
                        boot_stats.append(stat_boot)
                else:  # 'sample'
                    boot_stats = []
                    for _ in range(n_bootstrap):
                        # 对原始抽样样本进行有放回重采样
                        boot_i = np.random.choice(samples_list[i], size=n_samples, replace=True)
                        boot_j = np.random.choice(samples_list[j], size=n_samples, replace=True)
                        stat_boot, _ = ks_2samp(boot_i, boot_j)
                        boot_stats.append(stat_boot)

                boot_stats = np.array(boot_stats)
                D_error[i, j] = np.std(boot_stats)
                D_error[j, i] = D_error[i, j]
                D_ci_lower[i, j] = np.percentile(boot_stats, 0.15)
                D_ci_upper[i, j] = np.percentile(boot_stats, 99.85)
                D_ci_lower[j, i] = D_ci_lower[i, j]
                D_ci_upper[j, i] = D_ci_upper[i, j]

                if return_extra and bootstrap_D_samples is not None:
                    bootstrap_D_samples[i][j] = boot_stats.tolist()
                    bootstrap_D_samples[j][i] = boot_stats.tolist()  # 对称存储

    # 多重比较校正（p值）
    if p_adjust is not None:
        triu_indices = np.triu_indices(n_spectra, k=1)
        p_vals_flat = p_matrix[triu_indices]
        n_tests = len(p_vals_flat)

        if p_adjust.lower() == 'bonferroni':
            p_adj_flat = np.minimum(p_vals_flat * n_tests, 1.0)
        elif p_adjust.lower() in ['fdr_bh', 'bh']:
            try:
                from statsmodels.stats.multitest import multipletests
                _, p_adj_flat, _, _ = multipletests(p_vals_flat, method='fdr_bh')
            except ImportError:
                raise ImportError("fdr_bh校正需要statsmodels库，请安装或使用Bonferroni。")
        else:
            raise ValueError(f"不支持的校正方法: {p_adjust}")

        p_matrix[triu_indices] = p_adj_flat
        p_matrix[(triu_indices[1], triu_indices[0])] = p_adj_flat

    # 构建结果字典
    result = {
        'p_matrix': p_matrix,
        'D_matrix': D_matrix,
    }
    if bootstrap:
        result['D_error'] = D_error
        result['D_ci_lower'] = D_ci_lower
        result['D_ci_upper'] = D_ci_upper
    if return_extra:
        result['samples_list'] = samples_list
        if bootstrap:
            result['bootstrap_D_samples'] = bootstrap_D_samples

    return result


def detect_autocorr_spikes(autocorr, lags, smooth_sigma=3.0, threshold=3.0,
                           min_lag=0, positive_lags_only=True, 
                           return_details=False):
    """
    基于公式 Ji et al. 2018年文章公式(11) 的自相关尖峰检测算法。

    参数:
        autocorr : 1D array
            自相关函数值（通常已归一化，零滞后=1）。
        lags : 1D array
            对应的滞后坐标（样本数或实际时间）。
        smooth_sigma : float
            高斯平滑的标准差（以滞后间隔为单位）。
        threshold : float
            检测阈值倍数（kσ），例如 3.0。
        min_lag : int
            忽略小于此值的滞后（避免零滞后附近）。
        positive_lags_only : bool
            是否只考虑正滞后部分（通常透镜延迟为正）。
        return_details : bool
            是否返回残差、平滑曲线、阈值等详细信息。

    返回:
        dict
            - 'spike_lags' : 检测到的尖峰滞后（数组）。
            - 'spike_values' : 尖峰处的自相关值。
            - 'spike_residuals' : 尖峰处的残差。
            - (可选) 'autocorr_smoothed' : 高斯平滑后的自相关。
            - (可选) 'residuals' : 残差数组。
            - (可选) 'sigma' : 计算出的 σ 值。
    """
    # 提取要分析的部分
    if positive_lags_only:
        idx = np.where(lags > 0)[0]
    else:
        idx = np.arange(len(lags))
    # 进一步过滤 min_lag
    idx = idx[lags[idx] >= min_lag]
    if len(idx) == 0:
        raise ValueError("没有满足滞后范围的点。")

    x = lags[idx]
    y = autocorr[idx]

    # 高斯平滑
    y_smooth = gaussian_filter1d(y, sigma=smooth_sigma, mode='reflect')
    
    # Savitzky-Golay光滑
    #y_smooth = savgol_filter(y, window_length=100, polyorder=3, deriv=0, mode='interp')

    # 残差
    residuals = y - y_smooth

    # 计算 sigma 参数（残差的标准差）
    sigma = np.sqrt(np.mean(residuals**2))

    # 检测尖峰：残差 > threshold * sigma 且为正尖峰（即自相关大于平滑值）
    spike_mask = (residuals > threshold * sigma) & (y > y_smooth)
    
    # 可选：使用 scipy.signal.find_peaks 提取局部极大值，避免邻域重复
    candidate_idx = idx[spike_mask]
    if len(candidate_idx) > 0:
        # 在原始数组上找局部极大值
        y_full = autocorr
        final_spikes = []
        for i in candidate_idx:
            # 简单邻域检查（左右各1点）
            if (i == 0 or y_full[i] > y_full[i-1]) and (i == len(y_full)-1 or y_full[i] > y_full[i+1]):
                final_spikes.append(i)
        final_spikes = np.array(final_spikes, dtype=int)   # 确保整数类型
    else:
        final_spikes = np.array([], dtype=int)              # 空数组也指定整数类型

    # 构建结果
    result = {
        'spike_lags': lags[final_spikes],
        'spike_values': autocorr[final_spikes],
        'spike_residuals': residuals[np.where(idx == final_spikes[:, None])[1]] if len(final_spikes) else np.array([]),
        'sigma': sigma
    }
    if return_details:
        result['autocorr_smoothed'] = y_smooth
        result['residuals'] = residuals
        result['lags_analyzed'] = x

    return result

def compute_autocorr_with_spikes(ts, smooth_sigma=3.0, threshold=3.0,
                                 min_lag=1, positive_lags_only=True,
                                 detect_spikes=True, return_details=False,
                                 demean=True):
    """
    计算时间序列的自相关，并可选择检测尖峰。

    参数:
        ts : array_like
            一维时间序列。
        smooth_sigma : float
            高斯平滑的标准差，用于尖峰检测（仅当 detect_spikes=True 时有效）。
        threshold : float
            检测阈值倍数（kσ），用于尖峰检测。
        min_lag : int
            忽略小于此值的滞后（避免零滞后附近）。
        positive_lags_only : bool
            是否只考虑正滞后部分（通常透镜延迟为正）。
        detect_spikes : bool
            是否进行尖峰检测。
        return_details : bool
            是否返回平滑曲线、残差等细节（仅在 detect_spikes=True 时有效）。
        demean : bool
            是否在计算自相关前减去均值。

    返回:
        dict
            - 'autocorr' : 自相关函数（正滞后部分，包括零滞后）。
            - 'lags' : 滞后坐标（样本数，从0到len(ts)-1）。
            - 'spike_result' : 尖峰检测结果字典（如果 detect_spikes=True），否则 None。
    """
    ts = np.asarray(ts)
    if demean:
        ts = ts - np.mean(ts)
    n = len(ts)

    # 计算自相关（有偏归一化）
    ac_full = np.correlate(ts, ts, mode='full')
    ac_full = ac_full / ac_full[n-1]          # 归一化，零滞后=1
    lags = np.arange(0, n)
    autocorr = ac_full[n-1:]                  # 正滞后部分（包括零滞后）

    spike_result = None
    if detect_spikes:
        # 调用尖峰检测函数
        spike_result = detect_autocorr_spikes(
            autocorr, lags,
            smooth_sigma=smooth_sigma,
            threshold=threshold,
            min_lag=min_lag,
            positive_lags_only=positive_lags_only,
            return_details=return_details
        )

    return {
        'autocorr': autocorr,
        'lags': lags,
        'spike_result': spike_result
    }















