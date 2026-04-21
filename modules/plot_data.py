#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 17:22:52 2026

@author: ubuntu
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import norm


#所有的画图函数，包含：
#1.plot_dm_width_snr是画出FRB样本中DM/z-信号宽度-信噪比的函数；
#2.plot_dynamic_spectrum是画出每个FRB动态谱图的函数；
#3.plot_ks_heatmap和plot_qq_matrix是画出峰与峰之间的K-S检验结果表征频率漂移的函数；
#4.plot_autocorr_with_spikes是画出ts自相关以及存在明显自相关信号的图。

def zm(dm_wo_mw):
    """
    根据总 DM 减去银河系贡献后估算红移 z。
    公式来源：z = [ (a-b2) + sqrt((b2-a)^2 - 4*b2*(c2-a)) ] / (2*b2)
    其中 a = dm_wo_mw, b2=855, c2=200
    """
    a = dm_wo_mw
    b2 = 855.0
    c2 = 200.0
    delta = (b2 - a)**2 - 4 * b2 * (c2 - a)
    if delta < 0:
        return np.nan
    z = ((a - b2) + np.sqrt(delta)) / (2 * b2)
    if z <= 0:
        return 1e-3
    return z


def clean_data_for_scatter(data, name, x_axis='dm'):
    """
    根据 x_axis 模式清理数据，返回清理后的数据数组及无效计数。
    - x_axis='dm' : 需要 dm_fitb, width_fitb, snr_fitb 均有效 (>0 且有限)
    - x_axis='redshift' : 需要 dm_exc_ymw16 (>=0 且有限) 且能计算出有效红移，
                         同时 width_fitb, snr_fitb 有效。
    """
    print(f"\n=== 检查 {name} 的无效数据 (模式: {x_axis}) ===")
    
    # 检查必需字段是否存在
    if x_axis == 'dm':
        required = ['dm_fitb', 'width_fitb', 'snr_fitb']
    else:  # redshift
        required = ['dm_exc_ymw16', 'width_fitb', 'snr_fitb']
    for field in required:
        if field not in data.dtype.names:
            raise KeyError(f"数据中缺少必需字段: {field}")
    
    valid_mask = np.ones(len(data), dtype=bool)
    invalid_indices = []
    
    # 1. 检查 width_fitb 和 snr_fitb（两种模式都需要）
    for field in ['width_fitb', 'snr_fitb']:
        vals = data[field]
        invalid = ~np.isfinite(vals) | (vals <= 0)
        if np.any(invalid):
            print(f"  {field}: 发现 {np.sum(invalid)} 个无效值 (NaN/Inf/<=0)")
            for idx in np.where(invalid)[0]:
                if idx not in invalid_indices:
                    invalid_indices.append(idx)
                val = vals[idx]
                info = f"    索引 {idx} | {field}: {val}"
                if 'tns_name' in data.dtype.names:
                    info += f" | TNS: {data['tns_name'][idx]}"
                print(info)
            valid_mask &= ~invalid
    
    # 2. 根据模式检查横坐标字段
    if x_axis == 'dm':
        dm_vals = data['dm_fitb']
        dm_invalid = ~np.isfinite(dm_vals) | (dm_vals <= 0)
        if np.any(dm_invalid):
            print(f"  dm_fitb: 发现 {np.sum(dm_invalid)} 个无效值 (NaN/Inf/<=0)")
            for idx in np.where(dm_invalid)[0]:
                if idx not in invalid_indices:
                    invalid_indices.append(idx)
                val = dm_vals[idx]
                info = f"    索引 {idx} | dm_fitb: {val}"
                if 'tns_name' in data.dtype.names:
                    info += f" | TNS: {data['tns_name'][idx]}"
                print(info)
            valid_mask &= ~dm_invalid
    else:  # redshift
        dmm_vals = data['dm_exc_ymw16']
        dmm_invalid = ~np.isfinite(dmm_vals) | (dmm_vals < 0)
        if np.any(dmm_invalid):
            print(f"  dm_exc_ymw16: 发现 {np.sum(dmm_invalid)} 个无效值 (NaN/Inf/<0)")
            for idx in np.where(dmm_invalid)[0]:
                if idx not in invalid_indices:
                    invalid_indices.append(idx)
                val = dmm_vals[idx]
                info = f"    索引 {idx} | dm_exc_ymw16: {val}"
                if 'tns_name' in data.dtype.names:
                    info += f" | TNS: {data['tns_name'][idx]}"
                print(info)
            valid_mask &= ~dmm_invalid
        
        # 额外检查红移计算的有效性（对通过前两步的数据）
        temp_valid = data[valid_mask]
        z_list = []
        redshift_invalid_mask = np.zeros(len(temp_valid), dtype=bool)
        for i, row in enumerate(temp_valid):
            z_val = zm(row['dm_exc_ymw16'])
            if not np.isfinite(z_val) or z_val <= 0:
                redshift_invalid_mask[i] = True
            else:
                z_list.append(z_val)
        if np.any(redshift_invalid_mask):
            # 需要将无效红移对应的原始索引标记为无效
            orig_indices = np.where(valid_mask)[0][redshift_invalid_mask]
            for idx in orig_indices:
                if idx not in invalid_indices:
                    invalid_indices.append(idx)
                print(f"    索引 {idx} | 计算红移无效 (dm_exc_ymw16={data['dm_exc_ymw16'][idx]})")
            # 更新 valid_mask
            valid_mask[np.where(valid_mask)[0][redshift_invalid_mask]] = False
    
    cleaned = data[valid_mask]
    print(f"  清理后剩余 {len(cleaned)} 个有效数据点 (共移除 {len(invalid_indices)} 个无效点)")
    return cleaned, len(invalid_indices)


def plot_scatter_with_choice(repeater_file, non_repeater_file, x_axis='dm'):
    """
    绘制重复暴和非重复暴的散点图（合并到一张图），颜色表示 snr_fitb。
    
    参数:
    repeater_file: 重复暴数据文件路径 (.npy)
    non_repeater_file: 非重复暴数据文件路径 (.npy)
    x_axis: 横坐标类型，可选 'dm' 或 'redshift'
            - 'dm': 横坐标为 dm_fitb (pc cm^-3)
            - 'redshift': 横坐标为根据 dm_exc_ymw16 估算的红移 z
    """
    # 加载数据
    repeater_data = np.load(repeater_file, allow_pickle=True)
    non_repeater_data = np.load(non_repeater_file, allow_pickle=True)
    
    print("=" * 60)
    print(f"开始检查数据质量 (横坐标模式: {x_axis})...")
    repeater_clean, rep_invalid = clean_data_for_scatter(repeater_data, "重复暴", x_axis)
    non_repeater_clean, non_invalid = clean_data_for_scatter(non_repeater_data, "非重复暴", x_axis)
    print("=" * 60)
    
    # 如果没有有效数据则退出
    if len(repeater_clean) == 0 and len(non_repeater_clean) == 0:
        print("错误: 没有有效数据可以绘图!")
        return
    
    # 准备横坐标数据
    if x_axis == 'dm':
        x_rep = repeater_clean['dm_fitb'] if len(repeater_clean) > 0 else np.array([])
        x_non = non_repeater_clean['dm_fitb'] if len(non_repeater_clean) > 0 else np.array([])
        x_label = r'DM (pc cm$^{-3}$)'
        out_filename = './Figures/frb_dm_width_snr.pdf'
    else:  # redshift
        def get_redshifts(data):
            return np.array([zm(row['dm_exc_ymw16']) for row in data])
        x_rep = get_redshifts(repeater_clean) if len(repeater_clean) > 0 else np.array([])
        x_non = get_redshifts(non_repeater_clean) if len(non_repeater_clean) > 0 else np.array([])
        x_label = r'$z$'
        out_filename = './Figures/frb_redshift_width_snr.pdf'
    
    # 纵坐标 (width in ms)
    y_rep = repeater_clean['width_fitb'] * 1e3 if len(repeater_clean) > 0 else np.array([])
    y_non = non_repeater_clean['width_fitb'] * 1e3 if len(non_repeater_clean) > 0 else np.array([])
    
    # SNR 值（用于颜色映射）
    snr_rep = repeater_clean['snr_fitb'] if len(repeater_clean) > 0 else np.array([])
    snr_non = non_repeater_clean['snr_fitb'] if len(non_repeater_clean) > 0 else np.array([])
    
    # 确定统一的坐标范围
    all_x = np.concatenate([x_rep, x_non]) if len(x_rep)+len(x_non)>0 else np.array([])
    all_y = np.concatenate([y_rep, y_non]) if len(y_rep)+len(y_non)>0 else np.array([])
    if len(all_x) > 0:
        x_min, x_max = all_x.min() * 0.9, all_x.max() * 1.1
        y_min, y_max = all_y.min() * 0.9, all_y.max() * 1.1
    else:
        x_min, x_max = 0, 50
        y_min, y_max = 0, 50
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.plasma
    
    # 绘制重复暴 (圆圈)
    if len(x_rep) > 0:
        sc1 = ax.scatter(x_rep, y_rep, c=snr_rep, cmap=cmap,
                         marker='o', s=40, alpha=0.5,
                         edgecolors='black', linewidth=0.5,
                         label=f'Repeaters (N={len(x_rep)})')
    
    # 绘制非重复暴 (三角形)
    if len(x_non) > 0:
        sc2 = ax.scatter(x_non, y_non, c=snr_non, cmap=cmap,
                         marker='^', s=40, alpha=0.5,
                         edgecolors='black', linewidth=0.5,
                         label=f'Non-repeaters (N={len(x_non)})')
    
    # 坐标轴设置
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel('Width (ms)', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc=1, fontsize=16)
    
    # 颜色条
    if len(x_rep) > 0:
        cbar = plt.colorbar(sc1, ax=ax)
    elif len(x_non) > 0:
        cbar = plt.colorbar(sc2, ax=ax)
    else:
        cbar = None
    if cbar:
        cbar.ax.tick_params(labelsize=13)
        cbar.set_label(label='SNR', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印最终统计
    print("\n" + "=" * 60)
    print("最终统计结果:")
    print("=" * 60)
    print(f"重复暴:")
    print(f"  原始数据: {len(repeater_data)} 个")
    print(f"  无效数据: {rep_invalid} 个")
    print(f"  有效数据: {len(repeater_clean)} 个")
    if len(repeater_clean) > 0:
        if x_axis == 'dm':
            print(f"  DM范围: [{x_rep.min():.1f}, {x_rep.max():.1f}]")
        else:
            print(f"  Redshift范围: [{x_rep.min():.3f}, {x_rep.max():.3f}]")
        print(f"  Width范围: [{y_rep.min():.2f}, {y_rep.max():.2f}] ms")
        print(f"  SNR范围: [{snr_rep.min():.1f}, {snr_rep.max():.1f}]")
    
    print(f"\n非重复暴:")
    print(f"  原始数据: {len(non_repeater_data)} 个")
    print(f"  无效数据: {non_invalid} 个")
    print(f"  有效数据: {len(non_repeater_clean)} 个")
    if len(non_repeater_clean) > 0:
        if x_axis == 'dm':
            print(f"  DM范围: [{x_non.min():.1f}, {x_non.max():.1f}]")
        else:
            print(f"  Redshift范围: [{x_non.min():.3f}, {x_non.max():.3f}]")
        print(f"  Width范围: [{y_non.min():.2f}, {y_non.max():.2f}] ms")
        print(f"  SNR范围: [{snr_non.min():.1f}, {snr_non.max():.1f}]")
# 使用示例（取消注释并替换实际文件路径）
# plot_redshift_distribution('./data/repeaters.npy', './data/non_repeaters.npy')


def plot_dynamic_spectrum(data_dict, frb_name, peak_indices=None, time_unit='ms'):
    """
    绘制FRB动态谱和时间序列，并标记峰值位置
    
    参数:
    data_dict: 包含FRB数据的字典
    frb_name: FRB名称
    peak_indices: 峰值索引列表，如 [16, 79]，None表示不标记峰值
    time_unit: 时间单位
    """
    # 获取数据
    d_w = data_dict['dynamic_spectrum']
    dm = data_dict['dm']
    t = data_dict['times_relative']
    freq = data_dict['frequencies']
    ts = data_dict['ts']
    
    # 转换时间单位
    if time_unit == 'ms':
        t = t * 1e3
    
    # 创建图形
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, hspace=0.0, height_ratios=[1, 3])
    
    # 创建子图
    ax_ts = plt.subplot(gs[0])  # 时间序列
    ax_im = plt.subplot(gs[1])  # 动态谱
    
    # 计算颜色范围
    vmin = np.percentile(d_w, 1)
    vmax = np.percentile(d_w, 99)
    
    # 创建网格并绘制动态谱
    T, F = np.meshgrid(t, freq)
    ax_im.pcolormesh(T, F, d_w, vmin=vmin, vmax=vmax, cmap='viridis')
    
    # 设置坐标轴
    ax_im.set_xlabel(f'Time ({time_unit})', fontsize=14)
    ax_im.set_ylabel('Frequency (MHz)', fontsize=14)
    
    # 绘制时间序列
    ax_ts.plot(t, ts, 'k-', lw=1)
    ax_ts.set_xticklabels([])
    ax_ts.set_yticks([])
    
    # 添加FRB名称和DM值（换行显示）
    xpos = 0.95 * t.max()
    ypos = ts.max() * 0.95
    
    # 格式化DM值
    if dm >= 100:
        dm_str = f"{dm:.0f}"
    elif dm >= 10:
        dm_str = f"{dm:.1f}"
    else:
        dm_str = f"{dm:.2f}"
    
    # 第一行：FRB名称
    ax_ts.text(xpos, ypos, frb_name, ha='right', va='top', fontsize=16)
    
    # 第二行：DM值（放在FRB名称下方）
    ypos_dm = 0.7*ts.max()  # 再往下一些
    ax_ts.text(xpos, ypos_dm, f"DM= {dm_str} pc cm⁻³", 
               ha='right', va='top', fontsize=13)
    
    # 标记峰值位置（如果提供了peak_indices）
    if peak_indices is not None and len(peak_indices) > 0:
        # 获取峰值对应的时间和强度值
        peak_times = t[peak_indices]
        peak_heights = ts[peak_indices]
        
        # 绘制红点标记峰值位置
        ax_ts.scatter(peak_times, peak_heights, color='red', s=3, zorder=5)
        
        # 标记峰值序号（1, 2, 3...）
        for i, (peak_time, peak_height) in enumerate(zip(peak_times, peak_heights), 1):
            # 计算标记位置（在红点上方）
            label_y_offset = 0.05 * (ts.max() - ts.min())
            label_y = peak_height + label_y_offset
            
            # 添加序号标记
            ax_ts.text(peak_time, label_y, str(i), 
                      ha='center', va='bottom', 
                      fontsize=12, fontweight='bold',
                      color='red')
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_ks_heatmap(D_matrix, error_matrix=None, ci_upper=None, ci_lower=None,
                    signal_names=None, frb_name=None, tick_fontsize=16,
                    p_threshold=0.05, title="Pairwise K-S Statistics",
                    correction='fdr_bh', min_diff_threshold=0.1,
                    peak_indices=None, time_step_ms=0.98, color_label=None):
    """
    绘制热图，颜色基于KS统计量，格子内显示时间差（若提供peak_indices），星号标记显著性。

    参数:
        D_matrix : (N,N) K-S统计量矩阵（用于颜色和显著性）
        error_matrix : (N,N) 标准差矩阵（当需要计算p值时使用）
        ci_lower, ci_upper : (N,N) 置信区间下限和上限矩阵（优先使用）
        signal_names : list of str, 信号名称（默认使用索引）
        frb_name : str, optional FRB名称，若提供则添加到图标题前
        tick_fontsize : int, 坐标轴刻度标签字体大小
        p_threshold : float, 显著性阈值（仅当使用p值校正且未提供ci_upper时有效）
        title : str, 图标题基础部分
        correction : str, 多重比较校正方法 'fdr_bh', 'bonferroni', 'none'
        min_diff_threshold : float, 实际显著性阈值（D_stat > min_diff_threshold才标记）
        peak_indices : list or array, 峰值索引（样本点）。如果提供，格子内显示时间差矩阵
        time_step_ms : float, 每个样本对应的时间（毫秒），默认0.98
        color_label : str, 颜色条标签（默认 'K-S statistic'）
    """
    N = D_matrix.shape[0]
    if signal_names is None:
        signal_names = [f"P{i+1}" for i in range(N)]

    # ----- 显著性标记（基于原始KS统计量）-----
    if ci_upper is not None and ci_lower is not None:
        if ci_lower.shape != (N, N) or ci_upper.shape != (N, N):
            raise ValueError("ci_lower and ci_upper must have same shape as D_matrix")
        stat_sig = D_matrix > ci_upper
        practical_sig = D_matrix > min_diff_threshold
        significant_mask = stat_sig & practical_sig
        np.fill_diagonal(significant_mask, False)
        sig_mark = np.full((N, N), "", dtype=object)
        sig_mark[significant_mask] = "*"
        corr_note = f"99.75% CI upper bound & D > {min_diff_threshold}"
    elif error_matrix is not None:
        pvals = []
        pairs = []
        for i in range(N):
            for j in range(i+1, N):
                z = D_matrix[i, j] / (error_matrix[i, j] + 1e-12)
                p = 2 * (1 - norm.cdf(z))
                pvals.append(p)
                pairs.append((i, j))
        if correction == 'fdr_bh':
            try:
                from statsmodels.stats.multitest import multipletests
                reject, p_corr, _, _ = multipletests(pvals, alpha=p_threshold, method='fdr_bh')
                corr_name = "FDR"
            except ImportError:
                print("Warning: statsmodels not installed, falling back to Bonferroni correction.")
                m = len(pvals)
                p_corr = np.minimum(np.array(pvals) * m, 1.0)
                reject = p_corr < p_threshold
                corr_name = "Bonferroni"
        elif correction == 'bonferroni':
            m = len(pvals)
            p_corr = np.minimum(np.array(pvals) * m, 1.0)
            reject = p_corr < p_threshold
            corr_name = "Bonferroni"
        elif correction == 'none':
            p_corr = np.array(pvals)
            reject = p_corr < p_threshold
            corr_name = "uncorrected"
        else:
            raise ValueError("correction must be 'fdr_bh', 'bonferroni' or 'none'")
        sig_mark = np.full((N, N), "", dtype=object)
        for (i, j), rej in zip(pairs, reject):
            if rej and D_matrix[i, j] > min_diff_threshold:
                sig_mark[i, j] = "*"
                sig_mark[j, i] = "*"
        corr_note = f"p<{p_threshold} ({corr_name}) & D > {min_diff_threshold}"
    else:
        raise ValueError("Must provide either ci_upper/ci_lower or error_matrix to determine significance")

    # ----- 确定显示文本的矩阵（时间差或D_matrix）-----
    if peak_indices is not None:
        peaks = np.asarray(peak_indices)
        if len(peaks) != N:
            raise ValueError("Length of peak_indices must match D_matrix dimension")
        diff_samples = np.abs(peaks[:, None] - peaks[None, :])
        text_matrix = diff_samples * time_step_ms
        np.fill_diagonal(text_matrix, 0.0)
        # 颜色条标签固定为KS统计量
        default_color_label = r'$D_{\rm max}$'
    else:
        text_matrix = D_matrix
        default_color_label = r'$D_{\rm max}$'
    color_label = color_label if color_label is not None else default_color_label

    # 颜色数据始终为 D_matrix
    color_data = D_matrix

    # 标题
    if frb_name:
        full_title = f"{frb_name} - {title}\nSignificance: * {corr_note}"
    else:
        full_title = f"{title}\nSignificance: * {corr_note}"

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(color_data, cmap='viridis', aspect='auto', origin='upper')
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(color_label, fontsize=16)

    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(signal_names, fontsize=tick_fontsize)
    ax.set_yticklabels(signal_names, fontsize=tick_fontsize)

    # 添加文本：显示 text_matrix 的值 + 星号
    for i in range(N):
        for j in range(N):
            if i != j:
                val = text_matrix[i, j]
                text = f"{val:.2f}{sig_mark[i, j]}"
                # 文本颜色：如果时间差（或D值）超过 min_diff_threshold 则红色，否则黑色
                color = 'red' if val > min_diff_threshold else 'black'
                ax.text(j, i, text, ha='center', va='center',
                        color=color, fontsize=15)

    plt.tight_layout()
    plt.show()
    return fig


def plot_qq_matrix(samples_list, labels=None, colors=None, linestyles=None,
                   highlight_colors=None, highlight_linewidth=3,
                   figsize=(10, 8), alpha=0.7, linewidth=1.5,
                   xlabel='First spectrum',
                   ylabel='second spectrum',
                   title='Q-Q Plot: All Pairwise Comparisons',
                   show_legend=True, show_dull_legend=False,
                   legend_fontsize=16, label_fontsize=16, tick_fontsize=16,
                   highlight_pairs=None, pairs_subset=None,
                   dull_alpha=0.5, dull_linewidth=2, dull_color='gray',
                   same_scale=True, normalize=True):
    
    """
    绘制 Q-Q 图，每条线代表一对频谱的比较。
    高亮对（如统计显著且实际差异大的对）使用不同颜色和粗线，
    非高亮对使用细线（可自定义颜色和线型）。
    y=x 参考线保留但不显示图例标签。

    参数:
        samples_list : list of 1D arrays
            从频谱中抽样得到的频率值列表（原始值）。
        normalize : bool, default True
            是否对每个样本进行 min-max 归一化到 [0,1] 区间，以便比较分布形状。
        ... 其他参数含义不变 ...
    """
    n = len(samples_list)
    if n < 2:
        raise ValueError("至少需要两个样本才能绘制比较图。")

    if labels is None:
        labels = [f'Peak {i+1}' for i in range(n)]

    # 可选归一化
    if normalize:
        norm_samples = []
        for s in samples_list:
            min_s = np.min(s)
            max_s = np.max(s)
            if max_s - min_s == 0:
                norm_s = np.full_like(s, 0.5)
            else:
                norm_s = (s - min_s) / (max_s - min_s)
            norm_samples.append(norm_s)
        samples_use = norm_samples
    else:
        samples_use = samples_list

    # 处理高亮集合
    if highlight_pairs is not None:
        if isinstance(highlight_pairs, np.ndarray) and highlight_pairs.shape == (n, n):
            highlight_set = set()
            for i in range(n):
                for j in range(i+1, n):
                    if highlight_pairs[i, j]:
                        highlight_set.add((i, j))
        elif isinstance(highlight_pairs, list):
            highlight_set = set(highlight_pairs)
        else:
            raise ValueError("highlight_pairs must be a list of (i,j) tuples or a boolean matrix")
    else:
        highlight_set = None

    # 确定要绘制的对
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    if pairs_subset is not None:
        pairs_to_plot = [p for p in all_pairs if p in pairs_subset]
    else:
        pairs_to_plot = all_pairs

    n_pairs = len(pairs_to_plot)

    # 自动分配非高亮对的颜色和线型
    if colors is None:
        cmap = plt.cm.tab10
        colors = [cmap(i % 10) for i in range(n_pairs)]
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':']
        linestyles = [linestyles[i % len(linestyles)] for i in range(n_pairs)]

    # 为高亮对分配颜色
    n_highlights = len(highlight_set) if highlight_set else 0
    if highlight_colors is None:
        if colors:
            highlight_colors = colors[:n_highlights]
        else:
            cmap = plt.cm.tab10
            highlight_colors = [cmap(i % 10) for i in range(n_highlights)]
    highlight_color_map = {}
    if highlight_set:
        for idx, pair in enumerate(highlight_set):
            highlight_color_map[pair] = highlight_colors[idx % len(highlight_colors)]

    fig, ax = plt.subplots(figsize=figsize)

    # 坐标轴范围
    all_data = np.concatenate(samples_use)
    min_val, max_val = all_data.min(), all_data.max()
    if same_scale:
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    # 绘制 y=x 参考线（不添加图例标签）
    ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=2, label=None)

    # 绘制
    highlight_handles = []
    pair_idx = 0
    for i, j in pairs_to_plot:
        s_i = np.sort(samples_use[i])
        s_j = np.sort(samples_use[j])
        is_highlight = (highlight_set is not None) and ((i, j) in highlight_set)
        if is_highlight:
            color = highlight_color_map[(i, j)]
            line, = ax.plot(s_i, s_j,
                            color=color,
                            alpha=alpha,
                            linewidth=highlight_linewidth,
                            label=f'{labels[i]} vs {labels[j]}')
            highlight_handles.append(line)
        else:
            color = colors[pair_idx % len(colors)]
            ls = linestyles[pair_idx % len(linestyles)]
            line, = ax.plot(s_i, s_j,
                            color=color,
                            linestyle=ls,
                            alpha=dull_alpha,
                            linewidth=dull_linewidth,
                            label=f'{labels[i]} vs {labels[j]}' if show_dull_legend else None)
        pair_idx += 1

    # 图例
    if show_legend:
        if highlight_handles:
            ax.legend(handles=highlight_handles, loc='best', fontsize=legend_fontsize)
        elif show_dull_legend:
            ax.legend(loc='best', fontsize=legend_fontsize)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    #ax.set_title(title, fontsize=label_fontsize+2)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    return fig


def plot_autocorr_with_spikes(autocorr, lags, spike_result=None,
                              sigma_range=3, figsize=(10,8),
                              show_peaks=True,
                              label_fontsize=16, legend_fontsize=18,
                              tick_fontsize=16,
                              time_step_ms=0.98,
                              show=True
                             ):   # 新增参数，控制是否显示图形
    """
    绘制自相关曲线，横坐标为时间延迟（毫秒），尖峰处标注具体时间。

    参数:
        autocorr : 1D array
            自相关函数值（归一化，零滞后=1）。
        lags : 1D array
            滞后样本数（从0到len(ts)-1）。
        spike_result : dict or None
            尖峰检测结果，必须包含：
                - 'lags_analyzed' : 分析范围内的滞后（样本数）。
                - 'autocorr_smoothed' : 平滑曲线。
                - 'sigma' : 残差标准差。
                - 'spike_lags' : 尖峰滞后（样本数）。
        sigma_range : float
            置信区间倍数（如3）。
        figsize : tuple
            图形尺寸。
        show_peaks : bool
            是否标记尖峰。
        label_fontsize : int
            轴标签字体大小。
        legend_fontsize : int
            图注字体大小。
        tick_fontsize : int
            刻度字体大小。
        time_step_ms : float
            相邻样本的时间间隔（毫秒），默认0.98 ms。
        show : bool
            是否显示图形，默认 True。设为 False 时可用于批量保存，避免弹窗。
    """

    # 转换为时间坐标（毫秒）
    time_lags = lags * time_step_ms

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_lags, autocorr, label='Autocorrelation', color='blue')

    if spike_result is not None:
        # 平滑曲线的时间坐标
        x_smooth = spike_result['lags_analyzed'] * time_step_ms
        y_smooth = spike_result['autocorr_smoothed']
        sigma = spike_result['sigma']
        ax.plot(x_smooth, y_smooth, '--', label='Smoothed', color='orange')

        # 置信区间
        upper = y_smooth + sigma_range * sigma
        lower = y_smooth - sigma_range * sigma
        ax.fill_between(x_smooth, lower, upper, alpha=0.3, color='gray',
                        label=f'±{sigma_range}σ range')

        # 标记尖峰
        if show_peaks and len(spike_result['spike_lags']) > 0:
            for lag_sample in spike_result['spike_lags']:
                # 尖峰对应的时间（毫秒）
                lag_time = lag_sample * time_step_ms
                # 找到尖峰的自相关值
                idx = np.where(lags == lag_sample)[0]
                if len(idx) == 0:
                    continue
                peak_val = autocorr[idx[0]]
                # 绘制竖线
                ax.axvline(lag_time, color='red', linestyle=':', alpha=1,
                           label='Spike' if lag_sample == spike_result['spike_lags'][0] else "")
                # 在尖峰上方添加时间标签
                ax.text(lag_time, peak_val + 0.02, f'{lag_time:.2f} ms',
                        ha='center', va='bottom', fontsize=16, color='red')

    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Time Delay (ms)', fontsize=label_fontsize)
    ax.set_ylabel('Autocorrelation', fontsize=label_fontsize)
    ax.set_ylim(top=1)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(alpha=0.3)
    plt.show()
    return fig





