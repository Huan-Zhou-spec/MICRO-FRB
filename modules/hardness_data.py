import numpy as np
from .read_data import read_frb_dynamic_spectrum
from .analysis_data import process_data_ts


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
                      k=1, window_factor=1.0, signal_factor=2.5,
                      chime_file='FRB_data/CHIME_cat2_frb/chimefrbcat2.npy',
                      data_dir='FRB_data/canfar_downloads/'):
    """
    基于总光变曲线中 peak1 的半高宽确定整数半径，构建对称奇数长度窗口。
    若两峰相距足够远，各自以峰为中心构建不重叠的对称窗口；
    若靠得太近，则以 peak_cut 为 peak1 的右边界，确定窗口长度后平移至 peak2。

    参数:
        k: 将频带分为 k+2 段（k=1 对应原来的 L, M, H 三段）
    """
    n_bands = k + 2  # 总子带数

    # 1. 读取动态谱并处理数据中的RFI
    file_path = f"{data_dir}{frb_name}_stokesi_dynamic_spectrum.h5"
    raw_data = read_frb_dynamic_spectrum(file_path)
    proc_data = process_data_ts(raw_data, f_down=32, t_down=1, rfi_factor=3)
    freq_axis = proc_data['frequencies']
    dynamic = proc_data['dynamic_spectrum']

    # 2. 获取观测频带
    chime_data = np.load(chime_file, allow_pickle=True)
    entry = next((e for e in chime_data if e['tns_name'] == frb_name), None)
    if entry is None:
        raise ValueError(f"未找到 {frb_name} 对应的 CHIME 数据")
    low_freq, high_freq = entry['low_freq'], entry['high_freq']
    print(low_freq, high_freq)

    # 3. 频带切割并合成 n_bands 段及总光变曲线
    idx_low = np.argmin(np.abs(freq_axis - low_freq))
    idx_high = np.argmin(np.abs(freq_axis - high_freq))
    if idx_low > idx_high:
        idx_low, idx_high = idx_high, idx_low
    sub_band = dynamic[idx_low:idx_high+1, :]

    groups = np.array_split(sub_band, n_bands, axis=0)
    band_ts = [np.sum(g, axis=0) for g in groups]  # 各子带光变曲线列表
    total = np.sum(band_ts, axis=0)
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
        # ---------- 共用边界模式 ----------
        # 1) 根据 peak1 和 peak_cut 确定半长度（整数）
        half_len = peak_cut - peak1_index
        if half_len < 0:
            raise ValueError("peak_cut 小于 peak1_index，无法构建窗口")
        # peak1 窗口边界
        left1 = peak1_index - half_len
        right1_exclusive = peak_cut + 1   # 包含 peak_cut
        # 检查 peak1 窗口不越界
        if left1 < 0 or right1_exclusive > T_len:
            raise ValueError(f"peak1 窗口超出数据范围: left1={left1}, right1={right1_exclusive}, T_len={T_len}")

        # 2) 相同半长度应用到 peak2，对称于 peak2_index
        left2 = peak2_index - half_len
        right2_exclusive = peak2_index + half_len + 1
        # 检查 peak2 窗口越界
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

        radius = half_len

    else:
        # ---------- 正常模式 ----------
        max_half_len1 = peak_cut - peak1_index - 1
        max_half_len2 = peak2_index - peak_cut - 1
        if max_half_len1 < 0 or max_half_len2 < 0:
            print("警告：正常模式下最大半长度负数，转为共用边界模式")
            return extract_with_fwhm(frb_name, peak1_index, peak2_index,
                                     k, window_factor, signal_factor,
                                     chime_file, data_dir)

        half_len_float = raw_radius_float
        half_len = int(round(half_len_float))
        half_len1 = min(half_len, max_half_len1)

        # ----- 确定 peak1 窗口 -----
        left1 = peak1_index - half_len1
        right1_exclusive = peak1_index + half_len1 + 1

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

        if left1 >= right1_exclusive:
            raise ValueError("正常模式下 peak1 窗口无效")
        if not (left1 <= peak1_index < right1_exclusive):
            raise ValueError("peak1 不在其窗口内")

        len1 = right1_exclusive - left1
        if len1 % 2 == 0:
            print(f"警告：peak1 窗口长度为偶数 {len1}，可能无法严格对称，将按原长度处理")

        # ----- 为 peak2 构造等长、以 peak2 为中心的窗口 -----
        half2 = (len1 - 1) // 2
        left2 = peak2_index - half2
        right2_exclusive = peak2_index + half2 + 1

        valid = True
        if left2 <= peak_cut:
            print(f"警告：以 peak2 为中心的窗口左边界 {left2} 不满足 > peak_cut ({peak_cut})，转为共用边界模式")
            valid = False
        if left2 < 0 or right2_exclusive > T_len:
            print(f"警告：以 peak2 为中心的窗口超出数据边界 (left2={left2}, right2={right2_exclusive})，转为共用边界模式")
            valid = False

        if not valid:
            return extract_with_fwhm(frb_name, peak1_index, peak2_index,
                                     k, window_factor, signal_factor,
                                     chime_file, data_dir)

        if not (left2 <= peak2_index < right2_exclusive):
            raise ValueError("peak2 不在构造的窗口内")

        radius = (len1 - 1) // 2

    # 提取各子带窗口内的数据
    band_left = [b[left1:right1_exclusive] for b in band_ts]
    band_right = [b[left2:right2_exclusive] for b in band_ts]

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

    band_noise = []
    band_noise_left = []
    band_noise_right = []
    for b in band_ts:
        nl, nr = slice_noise(b)
        band_noise_left.append(nl)
        band_noise_right.append(nr)
        if nl.size and nr.size:
            band_noise.append(np.concatenate([nl, nr]))
        else:
            band_noise.append(nl if nl.size else nr)

    # 构建返回字典
    result = {
        'n_bands': n_bands,
        'peak_cut_index': peak_cut,
        'left_boundary1': left1, 'right_boundary1': right1_exclusive,
        'left_boundary2': left2, 'right_boundary2': right2_exclusive,
        'radius': radius,
        'fwhm1': fwhm1, 'fwhm2': fwhm2,
        'shared_boundary': use_shared_boundary
    }

    for i in range(n_bands):
        result[f'band{i}_left'] = band_left[i]
        result[f'band{i}_right'] = band_right[i]
        result[f'band{i}_noise_all'] = band_noise[i]
        result[f'band{i}_noise_left'] = band_noise_left[i]
        result[f'band{i}_noise_right'] = band_noise_right[i]

    result['n_left'] = len(band_left[0])
    result['n_right'] = len(band_right[0])

    return result


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
    I_err = np.sqrt(n_bins) * noise_rms
    if N_noise > 0:
        B_err = noise_rms * n_bins / np.sqrt(N_noise)
    else:
        B_err = 0.0
    net = I_sum - bg_mean * n_bins
    net_err = np.sqrt(I_err**2 + B_err**2)
    return net, net_err


def hardness_ratio(band_sums, noise_rms_list, n_bins_list,
                   bg_mean_list, N_noise_list):
    """
    计算相邻子带间的硬度比及高斯误差。
    HR_i = band_{i+1}_net / band_i_net, i = 0, 1, ..., n_bands-2

    参数:
        band_sums: 各子带窗口总强度列表 [I_0, I_1, ..., I_{n-1}]
        noise_rms_list: 各子带噪声RMS列表
        n_bins_list: 各子带窗口点数列表
        bg_mean_list: 各子带噪声均值列表
        N_noise_list: 各子带噪声点数列表
    返回:
        hr: 硬度比列表 (长度 n_bands-1)，无效值为 np.nan
        hr_err: 硬度比误差列表
    """
    n_bands = len(band_sums)
    net_list = []
    err_list = []
    for i in range(n_bands):
        net_i, err_i = net_intensity_and_error(
            band_sums[i], noise_rms_list[i], n_bins_list[i],
            bg_mean_list[i], N_noise_list[i]
        )
        net_list.append(net_i)
        err_list.append(err_i)

    hr = []
    hr_err = []
    for i in range(n_bands - 1):
        num_net = net_list[i + 1]
        den_net = net_list[i]
        num_err = err_list[i + 1]
        den_err = err_list[i]
        if den_net <= 0 or num_net <= 0:
            hr.append(np.nan)
            hr_err.append(np.nan)
        else:
            ratio = num_net / den_net
            err = ratio * np.sqrt((num_err / num_net)**2 + (den_err / den_net)**2)
            hr.append(ratio)
            hr_err.append(err)

    return hr, hr_err


def compare_hardness_ratios(hr_left, hr_err_left, hr_right, hr_err_right, n_sigma=1.0):
    """
    比较左右窗口的硬度比是否在置信区间内吻合。
    判断标准: |HR_left - HR_right| <= n_sigma * sqrt(err_left^2 + err_right^2)
    
    参数:
        hr_left, hr_err_left: 左窗口硬度比及误差列表
        hr_right, hr_err_right: 右窗口硬度比及误差列表
        n_sigma: 置信度参数 (1.0=68%, 2.0=95%, 3.0=99.7%)
    
    返回:
        all_match: bool, 是否全部吻合
        match_results: list, 各硬度比的吻合结果
    """
    match_results = []
    all_match = True
    
    for i in range(len(hr_left)):
        if np.isnan(hr_left[i]) or np.isnan(hr_right[i]):
            match_results.append(False)
            all_match = False
        else:
            diff = abs(hr_left[i] - hr_right[i])
            combined_err = np.sqrt(hr_err_left[i]**2 + hr_err_right[i]**2)
            match = diff <= n_sigma * combined_err
            match_results.append(match)
            if not match:
                all_match = False
    
    return all_match, match_results


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
    c = 3.0e8
    G = 6.67430e-11
    M_sun = 1.989e30
    dt_sec = dt_ms * 1e-3
    g = (f - 1) * f**(-0.5) + np.log(f)
    M_kg = (c**3 * dt_sec) / (2.0 * G * g)
    M = M_kg / M_sun
    return M
