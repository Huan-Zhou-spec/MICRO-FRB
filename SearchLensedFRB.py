#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:29:34 2026

@author: ubuntu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from modules import (
    read_frb_dynamic_spectrum, process_data_ts,
    calculate_snr_peaks, plot_dynamic_spectrum,
    extract_peak_spectra, extract_noise_spectra, compare_spectra_ks,
    plot_ks_heatmap, plot_qq_matrix,
    compute_autocorr_with_spikes, plot_autocorr_with_spikes
)
import pandas as pd


# ============================================================
# 核心分析函数
# ============================================================

def analyze_lensing_candidate(frb_name, data_dir="FRB_data/canfar_downloads/",
                              output_dir="FRB_lensing_results/",
                              f_down=32, t_down=1, rfi_factor=3,
                              time_step_ms=0.98, min_diff_threshold=0.1,
                              smooth_sigma=3, threshold=3, n_peaks_list=[5,7],
                              n_noise=30, n_bootstrap=1000, random_seed=42,
                              save_figure=True, show_plots=False):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_seed)

    file_path = os.path.join(data_dir, f"{frb_name}_stokesi_dynamic_spectrum.h5")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    raw_data = read_frb_dynamic_spectrum(file_path)
    proc_data = process_data_ts(raw_data, f_down=f_down, t_down=t_down, rfi_factor=rfi_factor)
    ts = proc_data['ts']

    # 1. 自相关与尖峰检测
    autocorr_result = compute_autocorr_with_spikes(
        ts, smooth_sigma=smooth_sigma, threshold=threshold,
        min_lag=1, positive_lags_only=True,
        detect_spikes=True, return_details=True, demean=True
    )
    spike_lags = autocorr_result['spike_result']['spike_lags']
    spike_times = [lag * time_step_ms for lag in spike_lags]

    exclude = False
    lens_candidate = False
    has_drift = False
    matched_peak_indices = []
    matched_pairs = []
    peak_indices = []

    if not spike_times:
        print(f"{frb_name}: 无自相关尖峰，排除透镜候选")
        exclude = True
    else:
        print(f"{frb_name}: 检测到自相关尖峰，尖峰时间 (ms): {spike_times}")

        # 2. 尝试匹配峰对
        for n_peaks in n_peaks_list:
            peaks = calculate_snr_peaks(ts, n=n_peaks, adaptive=True)
            tmp_indices = peaks['peak_indices']
            snr_peaks = peaks['snr_values']
            pairs = []
            for i in range(len(tmp_indices)):
                for j in range(i+1, len(tmp_indices)):
                    diff_ms = abs(tmp_indices[j] - tmp_indices[i]) * time_step_ms
                    pairs.append((i, j, tmp_indices[i], tmp_indices[j], diff_ms))
            matched = []
            for spike in spike_times:
                for (i, j, idx_i, idx_j, diff_ms) in pairs:
                    if abs(diff_ms - spike) <= 2:
                        matched.append((idx_i, idx_j))
            if matched:
                matched_peak_indices = sorted(set([idx for pair in matched for idx in pair]))
                matched_pairs = matched
                peak_indices = tmp_indices
                break
        else:
            default_peaks = calculate_snr_peaks(ts, n=n_peaks_list[1], adaptive=True)
            peak_indices = default_peaks['peak_indices']
            print(f"  未匹配到峰对，使用检测到的 {len(peak_indices)} 个峰值")

        # 3. 漂移判断（仅针对匹配到的峰对）
        if matched_pairs:
            # ----- 新增：SNR 顺序检查（前峰 SNR 必须 >= 后峰 SNR，否则排除）-----
            idx_to_snr = {idx: snr for idx, snr in zip(peak_indices, snr_peaks)}
            snr_order_ok = True
            for (idx_i, idx_j) in matched_pairs:
                snr_i = idx_to_snr.get(idx_i)
                snr_j = idx_to_snr.get(idx_j)
                if snr_i is not None and snr_j is not None and snr_i < snr_j:
                    snr_order_ok = False
                    print(f"  峰对 ({idx_i}, {idx_j}) 的 SNR 顺序错误：前峰 SNR={snr_i:.2f} < 后峰 SNR={snr_j:.2f}，排除透镜候选")
                    break
            if not snr_order_ok:
                exclude = True
                lens_candidate = False
                has_drift = False
                # 跳过后续漂移判断
            else:
                # ----- 原有漂移判断逻辑 -----
                all_matched_indices = sorted(set([idx for pair in matched_pairs for idx in pair]))
                if len(all_matched_indices) < 2:
                    print("  匹配峰对数量不足，无法进行漂移判断，保留候选")
                    lens_candidate = True
                else:
                    # 提取频谱
                    spectra_info = extract_peak_spectra(proc_data, all_matched_indices, time_window=None)
                    freq_axis = spectra_info['frequencies']
                    spectra_list = spectra_info['peak_spectra']
                    idx_to_pos = {idx: pos for pos, idx in enumerate(all_matched_indices)}

                    noise_result = extract_noise_spectra(
                        proc_data, n_noise=n_noise, noise_method='quantile',
                        quantile_threshold=0.25, allow_oversample=True
                    )
                    noise_spectra = noise_result['noise_spectra']

                    res_ks = compare_spectra_ks(
                        spectra_list, freq_axis=freq_axis, noise_samples=noise_spectra,
                        bootstrap=True, bootstrap_method='add_noise',
                        n_bootstrap=n_bootstrap, random_seed=random_seed,
                        return_extra=True
                    )
                    shift_mat = res_ks['D_matrix']
                    ci_high = res_ks.get('D_ci_upper', None)

                    if ci_high is not None:
                        all_pairs_drift = True
                        for idx_i, idx_j in matched_pairs:
                            pos_i = idx_to_pos[idx_i]
                            pos_j = idx_to_pos[idx_j]
                            D = shift_mat[pos_i, pos_j]
                            if hasattr(ci_high, 'shape') and len(ci_high.shape) == 2:
                                thresh = ci_high[pos_i, pos_j]
                            else:
                                thresh = ci_high
                            if D > thresh and D > min_diff_threshold:
                                continue
                            else:
                                all_pairs_drift = False
                                break
                        if all_pairs_drift:
                            has_drift = True
                            exclude = True
                            print("  所有匹配峰对均存在严重频率漂移，排除透镜候选")
                        else:
                            lens_candidate = True
                    else:
                        print("  无置信区间信息，无法判断频率漂移，保留候选")
                        lens_candidate = True
        else:
            print("  未匹配到任何峰对，跳过漂移判断，不保留候选")
            exclude = True
            lens_candidate = False

    # 4. 保存图片和报告
    output_paths = {}
    report_file = None

    if not exclude and save_figure:
        if not peak_indices:
            default_peaks = calculate_snr_peaks(ts, n=n_peaks_list[0], adaptive=True)
            peak_indices = default_peaks['peak_indices']

        # 动态谱
        fig_ds = plot_dynamic_spectrum(proc_data, frb_name, peak_indices=peak_indices, time_unit='ms')
        if fig_ds is not None:
            save_path = os.path.join(output_dir, f"{frb_name}_dynamic_spectrum.png")
            fig_ds.savefig(save_path, dpi=150, bbox_inches='tight')
            output_paths['dynamic_spectrum'] = save_path
            plt.close(fig_ds)

        # 自相关图
        fig_acf = plot_autocorr_with_spikes(
            autocorr_result['autocorr'], autocorr_result['lags'],
            spike_result=autocorr_result['spike_result'],
            sigma_range=3, time_step_ms=time_step_ms
        )
        if fig_acf is not None:
            save_path = os.path.join(output_dir, f"{frb_name}_autocorr.png")
            fig_acf.savefig(save_path, dpi=150, bbox_inches='tight')
            output_paths['autocorr'] = save_path
            plt.close(fig_acf)

        # KS热图和QQ矩阵
        if len(peak_indices) >= 2:
            spectra_info = extract_peak_spectra(proc_data, peak_indices, time_window=None)
            freq_axis = spectra_info['frequencies']
            spectra_list = spectra_info['peak_spectra']
            noise_result = extract_noise_spectra(
                proc_data, n_noise=n_noise, noise_method='quantile',
                quantile_threshold=0.25, allow_oversample=False
            )
            noise_spectra = noise_result['noise_spectra']
            res_ks_plot = compare_spectra_ks(
                spectra_list, freq_axis=freq_axis, noise_samples=noise_spectra,
                bootstrap=True, bootstrap_method='add_noise',
                n_bootstrap=n_bootstrap, random_seed=random_seed,
                return_extra=True
            )
            shift_mat_plot = res_ks_plot['D_matrix']
            err_mat_plot = res_ks_plot['D_error']
            ci_low_plot = res_ks_plot.get('D_ci_lower', None)
            ci_high_plot = res_ks_plot.get('D_ci_upper', None)
            samples_list_plot = res_ks_plot['samples_list']

            # KS热图
            fig_heat = plot_ks_heatmap(
                shift_mat_plot, error_matrix=err_mat_plot,
                ci_lower=ci_low_plot, ci_upper=ci_high_plot,
                frb_name=frb_name, tick_fontsize=16,
                min_diff_threshold=min_diff_threshold,
                peak_indices=peak_indices
            )
            if fig_heat is not None:
                save_path = os.path.join(output_dir, f"{frb_name}_ks_heatmap.png")
                fig_heat.savefig(save_path, dpi=150, bbox_inches='tight')
                output_paths['ks_heatmap'] = save_path
                plt.close(fig_heat)

            # QQ矩阵
            if ci_high_plot is not None:
                N_plot = len(samples_list_plot)
                highlight_mask = (shift_mat_plot > ci_high_plot) & (shift_mat_plot > min_diff_threshold)
                np.fill_diagonal(highlight_mask, False)
                highlight_pairs = [(i, j) for i in range(N_plot) for j in range(i+1, N_plot) if highlight_mask[i, j]]
            else:
                highlight_pairs = None
            fig_qq = plot_qq_matrix(samples_list_plot, highlight_pairs=highlight_pairs,
                                    show_legend=True, same_scale=True)
            if fig_qq is not None:
                save_path = os.path.join(output_dir, f"{frb_name}_qq_matrix.png")
                fig_qq.savefig(save_path, dpi=150, bbox_inches='tight')
                output_paths['qq_matrix'] = save_path
                plt.close(fig_qq)

        # 文本报告
        report_file = os.path.join(output_dir, f"{frb_name}_report.txt")
        with open(report_file, 'w') as f:
            f.write(f"FRB透镜候选分析报告: {frb_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"自相关尖峰时间 (ms): {spike_times}\n")
            f.write(f"透镜候选: {lens_candidate}\n")
            f.write(f"匹配到的峰索引: {matched_peak_indices if matched_peak_indices else '无'}\n")
            f.write(f"匹配到的峰对: {matched_pairs if matched_pairs else '无'}\n")
            f.write(f"是否存在严重频率漂移: {has_drift}\n")
            for name, path in output_paths.items():
                f.write(f"{name}: {path}\n")
        print(f"{frb_name}: 分析完成，已保存图片和报告至 {output_dir}")

    else:
        print(f"{frb_name}: 已剔除，不保存任何文件")

    return {
        'frb_name': frb_name,
        'lens_candidate': lens_candidate,
        'spike_times': spike_times,
        'matched_peak_indices': matched_peak_indices,
        'matched_pairs': matched_pairs,
        'has_drift': has_drift,
        'output_paths': output_paths,
        'report_file': report_file
    }


def generate_lens_report(results, output_dir):
    """只保留未被剔除（即有尖峰且无漂移）的 FRB 信息"""
    valid = [r for r in results if r.get('status') == 'success' and r.get('report_file')]

    records = []
    for r in valid:
        records.append({
            'frb_name': r['frb_name'],
            'lens_candidate': r.get('lens_candidate', False),
            'num_spikes': len(r.get('spike_times', [])),
            'matched_peak_indices': str(r.get('matched_peak_indices', [])),
            'matched_pairs': str(r.get('matched_pairs', [])),
            'has_drift': r.get('has_drift', False),
            'report_file': r.get('report_file', '')
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, 'lens_catalog_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"汇总报告（仅保留分析成功的 FRB）已保存至 {csv_path}")

    # 文本摘要
    summary_path = os.path.join(output_dir, 'lens_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("FRB 透镜候选分析汇总报告（仅含通过剔除条件的 FRB）\n")
        f.write("=" * 60 + "\n")
        f.write(f"总处理 FRB 数量: {len(results)}\n")
        f.write(f"有效分析（有尖峰且无严重漂移）: {len(valid)}\n")
        f.write(f"其中透镜候选: {sum(1 for r in valid if r['lens_candidate'])}\n\n")
        if valid:
            f.write("有效 FRB 列表:\n")
            for r in valid:
                f.write(f"  {r['frb_name']}: 尖峰 {r['spike_times']}, 候选={r['lens_candidate']}\n")
        else:
            f.write("无有效 FRB（所有 FRB 均被剔除）。\n")
    print(f"文本摘要已保存至 {summary_path}")


def process_frb_catalog_lens(catalog_file, data_dir="FRB_data/canfar_downloads/",
                             output_dir="FRB_lensing_results/",
                             f_down=32, t_down=1, rfi_factor=3,
                             time_step_ms=0.98, min_diff_threshold=0.1,
                             smooth_sigma=3, threshold=3,
                             n_noise=30,
                             n_bootstrap=1000, random_seed=42,
                             save_figure=True, show_plots=False,
                             N=None):
    """
    批量处理FRB catalog，进行透镜候选分析。

    参数:
        catalog_file : str
            catalog文件路径（npy格式，必须包含'tns_name'字段）。
        data_dir : str
            FRB数据文件目录。
        output_dir : str
            输出图片和报告的目录（每个FRB的图片和报告会保存在此目录下）。
        f_down, t_down, rfi_factor : int
            数据处理参数（传给 process_data_ts）。
        time_step_ms : float
            时间分辨率（毫秒）。
        min_diff_threshold : float
            最小显著漂移阈值。
        smooth_sigma, threshold : 自相关尖峰检测参数。
        n_noise, noise_quantile : 噪声提取参数。
        n_bootstrap, random_seed : Bootstrap参数。
        save_figure : bool
            是否保存图片（传递给 analyze_lensing_candidate）。
        show_plots : bool
            是否显示图片（传递给 analyze_lensing_candidate）。
        N : int or None
            处理前 N 个 FRB，None 表示处理所有。

    返回:
        results : list of dict
            每个 FRB 的处理结果，包含基本信息、透镜候选状态等。
    """
    data_cata = np.load(catalog_file, allow_pickle=True)
    frb_names = data_cata['tns_name']
    sub_nums = data_cata['sub_num']

    if N is not None:
        frb_names = frb_names[:N]
        sub_nums = sub_nums[:N]

    os.makedirs(output_dir, exist_ok=True)
    print(f"开始透镜候选分析，共 {len(frb_names)} 个FRB...")
    results = []

    for i, (frb_name, sub_num) in enumerate(zip(frb_names, sub_nums)):
        n_peaks_expected = int(sub_num) + 1
        print(f"\n[{i+1}/{len(frb_names)}] 处理 {frb_name}")

        try:
            lens_result = analyze_lensing_candidate(
                frb_name=frb_name,
                data_dir=data_dir,
                output_dir=output_dir,
                f_down=f_down, t_down=t_down, rfi_factor=rfi_factor,
                time_step_ms=time_step_ms, min_diff_threshold=min_diff_threshold,
                smooth_sigma=smooth_sigma, threshold=threshold,
                n_peaks_list=[n_peaks_expected, n_peaks_expected+1],
                n_noise=n_noise,
                n_bootstrap=n_bootstrap, random_seed=random_seed,
                save_figure=save_figure, show_plots=show_plots
            )

            result_item = {
                'frb_name': frb_name,
                'lens_candidate': lens_result['lens_candidate'],
                'spike_times': lens_result['spike_times'],
                'matched_peak_indices': lens_result['matched_peak_indices'],
                'matched_pairs': lens_result['matched_pairs'],
                'has_drift': lens_result['has_drift'],
                'output_paths': lens_result['output_paths'],
                'report_file': lens_result['report_file'],
                'status': 'success'
            }
            results.append(result_item)
            print(f"  透镜候选: {lens_result['lens_candidate']}, 尖峰数: {len(lens_result['spike_times'])}, 匹配峰索引: {lens_result['matched_peak_indices']}")

        except FileNotFoundError as e:
            print(f"  文件不存在: {e}")
            results.append({'frb_name': frb_name, 'status': 'missing_file', 'error': str(e)})
        except Exception as e:
            print(f"  处理出错: {str(e)}")
            results.append({'frb_name': frb_name, 'status': 'error', 'error': str(e)})

    generate_lens_report(results, output_dir)
    return results


results = process_frb_catalog_lens(
    catalog_file="FRB_data/CHIME_cat2_frb/chimefrbcat2_first_duplicates.npy",
    data_dir="FRB_data/canfar_downloads/",
    output_dir="Figures/FRB_lensing_results/",
    f_down=32, t_down=1, rfi_factor=3,
    time_step_ms=0.98, min_diff_threshold=0.1,
    smooth_sigma=3, threshold=3,
    n_noise=30,
    n_bootstrap=1000, random_seed=42,
    save_figure=True, show_plots=False,
    N=340  # 仅处理前10个，用于测试
)


