#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心函数的单元测试。
测试范围：纯计算函数（不依赖 HDF5 数据文件或 CHIME catalog）。
"""

import sys
import os
import numpy as np
import pytest

# 确保可以导入 modules 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.analysis_data import (
    downsample, mask_rfi, calculate_snr_peaks,
    compute_autocorr_with_spikes, compare_spectra_ks,
    extract_peak_spectra, detect_autocorr_spikes,
)
from modules.smooth_data import (
    boxcar_kernel_1d, gaussian_kernel_1d, smooth_time_dimension,
)
from modules.hardness_data import (
    compute_fwhm, net_intensity_and_error,
    hardness_ratio, compare_hardness_ratios, lens_mass,
)
from modules.fpbh_data import zm


# ============================================================
# analysis_data 模块
# ============================================================

class TestDownsample:
    """测试 downsampling 函数"""

    def test_basic_downsample(self):
        data = np.ones((64, 100))
        result = downsample(data, f_down=2, t_down=2)
        assert result.shape == (32, 50)

    def test_no_downsample(self):
        data = np.ones((16, 20))
        result = downsample(data, f_down=1, t_down=1)
        assert result.shape == data.shape

    def test_preserves_total_flux(self):
        data = np.random.rand(32, 40)
        result = downsample(data, f_down=4, t_down=4)
        assert result.shape == (8, 10)
        # 下采样是对分组求和，总通量应近似守恒
        assert np.isclose(np.sum(data), np.sum(result), rtol=1e-10)


class TestMaskRFI:
    """测试 RFI 掩码函数"""

    def make_wfall(self, n_freq=100, n_time=200):
        np.random.seed(42)
        wfall = np.random.randn(n_freq, n_time) * 5 + 10
        return wfall

    def test_no_outlier_all_kept(self):
        wfall = self.make_wfall()
        wf_masked, spec, ts, rfi_mask = mask_rfi(wfall, rfi_factor=10)
        assert np.sum(rfi_mask) == 0
        np.testing.assert_array_almost_equal(wf_masked, wfall)

    def test_extreme_rfi_masked(self):
        wfall = self.make_wfall()
        wfall[0, :] = 1e6  # 一个极强的 RFI 通道
        _, _, _, rfi_mask = mask_rfi(wfall, rfi_factor=3)
        assert rfi_mask[0]
        # 其他通道不应被掩码
        assert np.sum(rfi_mask) <= 10

    def test_output_shapes(self):
        wfall = self.make_wfall(n_freq=50, n_time=80)
        wf_masked, spec, ts, rfi_mask = mask_rfi(wfall)
        assert wf_masked.shape == wfall.shape
        assert len(spec) == 50
        assert len(ts) == 80
        assert len(rfi_mask) == 50


class TestCalculateSNRPeaks:
    """测试 SNR 峰值检测"""

    def test_single_gaussian_peak(self):
        t = np.linspace(0, 10, 500)
        ts = np.exp(-0.5 * ((t - 5) / 0.5)**2) * 50 + np.random.randn(500) * 0.5
        result = calculate_snr_peaks(ts, n=5, adaptive=False)
        assert len(result['peak_indices']) > 0
        # 最高峰应在约 t=5 处（索引 250）
        highest_idx = result['peak_indices'][np.argmax(result['snr_values'])]
        assert 230 <= highest_idx <= 270

    def test_multiple_peaks(self):
        t = np.linspace(0, 20, 1000)
        ts = np.zeros(1000)
        for center in [200, 400, 700]:
            ts += np.exp(-0.5 * ((t - t[center]) / 0.3)**2) * 30
        ts += np.random.randn(1000) * 0.5
        result = calculate_snr_peaks(ts, n=3, adaptive=False)
        assert 2 <= len(result['peak_indices']) <= 5

    def test_adaptive_mode(self):
        t = np.linspace(0, 10, 500)
        ts = (np.exp(-0.5 * ((t - 3) / 0.4)**2) * 40
              + np.exp(-0.5 * ((t - 7) / 0.3)**2) * 35
              + np.random.randn(500) * 0.5)
        result = calculate_snr_peaks(ts, n=2, adaptive=True)
        assert 1 <= len(result['peak_indices']) <= 4

    def test_return_fields(self):
        np.random.seed(0)
        ts = np.random.randn(200) + 5 * np.exp(-0.5 * ((np.linspace(0, 10, 200) - 5) / 0.5)**2)
        result = calculate_snr_peaks(ts, n=3)
        for key in ['snr_values', 'peak_indices', 'peak_values',
                     'noise_mean', 'noise_std', 'num_peaks_found']:
            assert key in result

    def test_empty_flat_data(self):
        ts = np.ones(500) + np.random.randn(500) * 0.01
        result = calculate_snr_peaks(ts, n=3, adaptive=False,
                                     main_prominence_factor=3.0)
        assert isinstance(result['peak_indices'], list)


class TestComputeAutocorr:
    """测试自相关计算和尖峰检测"""

    def test_periodic_signal(self):
        t = np.linspace(0, 20, 1000)
        ts = (np.sin(2 * np.pi * t / 2.0) * 20
              + np.random.randn(1000) * 0.5)
        result = compute_autocorr_with_spikes(ts, smooth_sigma=3, threshold=2,
                                               detect_spikes=True)
        assert 'autocorr' in result
        assert 'lags' in result
        assert 'spike_result' in result
        # 零滞后处自相关应为 1
        assert np.isclose(result['autocorr'][0], 1.0, atol=0.01)

    def test_demean_flag(self):
        np.random.seed(1)
        ts = np.random.randn(300) + 100  # 有大偏移量
        r1 = compute_autocorr_with_spikes(ts, demean=True, detect_spikes=False)
        r2 = compute_autocorr_with_spikes(ts, demean=False, detect_spikes=False)
        # demean=True 应给出不同的自相关
        assert not np.allclose(r1['autocorr'], r2['autocorr'])


class TestCompareSpectraKS:
    """测试 KS 频谱比较"""

    def test_identical_spectra(self):
        spec = np.random.rand(100)
        spec = spec / np.sum(spec)
        result = compare_spectra_ks([spec, spec.copy()],
                                     n_samples=500, random_seed=42)
        # 相同频谱应有高 p 值
        assert result['p_matrix'][0, 1] > 0.1
        assert result['D_matrix'][0, 1] < 0.1

    def test_different_spectra(self):
        spec1 = np.ones(100)
        spec1[20:40] = 10
        spec2 = np.ones(100)
        spec2[60:80] = 10
        result = compare_spectra_ks([spec1, spec2],
                                     n_samples=500, random_seed=42)
        # 不同频谱应有低 p 值
        assert result['p_matrix'][0, 1] < 0.05

    def test_with_bootstrap(self):
        spec1 = np.random.rand(80)
        spec2 = np.random.rand(80)
        result = compare_spectra_ks(
            [spec1, spec2], n_samples=300, random_seed=42,
            bootstrap=True, n_bootstrap=50,
            bootstrap_method='sample',
        )
        assert 'D_error' in result
        assert 'D_ci_lower' in result
        assert 'D_ci_upper' in result

    def test_three_spectra(self):
        specs = [np.random.rand(60) for _ in range(3)]
        result = compare_spectra_ks(specs, n_samples=300, random_seed=42)
        assert result['p_matrix'].shape == (3, 3)
        assert result['D_matrix'].shape == (3, 3)
        # 对角应为 p=1, D=0
        assert np.allclose(np.diag(result['p_matrix']), 1.0)
        assert np.allclose(np.diag(result['D_matrix']), 0.0)


class TestExtractPeakSpectra:
    """测试峰值频谱提取"""

    def test_basic_extraction(self):
        n_freq, n_time = 64, 200
        dyn = np.random.randn(n_freq, n_time) * 2 + 10
        dyn[:, 100] = 50  # 在索引 100 处做一个峰
        proc_data = {
            'dynamic_spectrum': dyn,
            'frequencies': np.linspace(400, 800, n_freq),
            'times_relative': np.arange(n_time) * 0.98,
        }
        result = extract_peak_spectra(proc_data, peak_idxs=[100],
                                       time_window=2, aggregation='mean')
        assert len(result['peak_spectra']) == 1
        assert len(result['peak_spectra'][0]) == n_freq


class TestDetectAutocorrSpikes:
    """测试自相关尖峰检测"""

    def test_periodic_signal_spikes(self):
        t = np.arange(0, 500)
        ts = np.sin(2 * np.pi * t / 20) + 5
        ac = np.correlate(ts - np.mean(ts), ts - np.mean(ts), mode='full')
        ac = ac / ac[len(ts) - 1]
        lags = np.arange(len(ts))
        autocorr = ac[len(ts) - 1:]

        result = detect_autocorr_spikes(
            autocorr, lags, smooth_sigma=5, threshold=2,
            positive_lags_only=True, min_lag=1,
        )
        assert 'spike_lags' in result
        assert 'sigma' in result


# ============================================================
# smooth_data 模块
# ============================================================

class TestKernels:
    """测试平滑核函数"""

    def test_boxcar_kernel_sum(self):
        k = boxcar_kernel_1d(5)
        assert len(k) == 5

    def test_boxcar_kernel_values(self):
        k = boxcar_kernel_1d(9)
        assert np.allclose(k, np.ones(9) / 3.0)  # np.ones / sqrt(9)

    def test_gaussian_kernel_sum(self):
        k = gaussian_kernel_1d(sigma=2.0)
        assert np.isclose(np.sum(k), 1.0)

    def test_gaussian_kernel_symmetry(self):
        k = gaussian_kernel_1d(sigma=3.0)
        assert np.allclose(k, k[::-1])


class TestSmoothTimeDimension:
    """测试时间维度平滑"""

    def test_boxcar_smoothing(self):
        data = np.random.randn(16, 100)
        smoothed = smooth_time_dimension(data, time_width=5,
                                          kernel_type='boxcar')
        assert smoothed.shape == data.shape
        # 平滑后标准差应减小
        assert np.std(smoothed) < np.std(data)

    def test_gaussian_smoothing(self):
        data = np.random.randn(8, 80)
        smoothed = smooth_time_dimension(data, time_width=5,
                                          kernel_type='gaussian')
        assert smoothed.shape == data.shape


# ============================================================
# hardness_data 模块
# ============================================================

class TestComputeFWHM:
    """测试 FWHM 计算"""

    def test_gaussian_fwhm(self):
        t = np.linspace(0, 10, 1000)
        sigma = 0.3
        peak_idx = 500
        ts = np.exp(-0.5 * ((t - t[peak_idx]) / sigma)**2) * 100
        left, right, fwhm = compute_fwhm(ts, peak_idx, interp=True)
        expected_fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma / (10.0 / 999)
        assert np.isclose(fwhm, expected_fwhm, rtol=0.05)


class TestNetIntensity:
    """测试净强度计算"""

    def test_positive_net(self):
        net, err = net_intensity_and_error(
            I_sum=100, noise_rms=2, n_bins=10,
            bg_mean=1, N_noise=50,
        )
        assert net > 0
        assert err > 0

    def test_zero_noise_points(self):
        net, err = net_intensity_and_error(
            I_sum=50, noise_rms=1.5, n_bins=5,
            bg_mean=0.5, N_noise=0,
        )
        assert err >= 0


class TestHardnessRatio:
    """测试硬度比计算"""

    def test_basic_hr(self):
        band_sums = [60, 40, 20]
        noise_rms = [1, 1, 1]
        n_bins = [10, 10, 10]
        bg_mean = [0.5, 0.5, 0.5]
        N_noise = [50, 50, 50]
        hr, hr_err = hardness_ratio(band_sums, noise_rms,
                                     n_bins, bg_mean, N_noise)
        assert len(hr) == 2
        assert 0 < hr[0] < 1  # 40/60 ≈ 0.67


class TestCompareHardnessRatios:
    """测试硬度比一致性检验"""

    def test_matching_hr(self):
        hr_left = [0.5, 0.8]
        hr_err_left = [0.05, 0.05]
        hr_right = [0.52, 0.78]
        hr_err_right = [0.05, 0.05]
        all_match, results = compare_hardness_ratios(
            hr_left, hr_err_left, hr_right, hr_err_right, n_sigma=1.0,
        )
        assert all_match

    def test_divergent_hr(self):
        hr_left = [0.5, 0.8]
        hr_err_left = [0.01, 0.01]
        hr_right = [1.5, 0.1]
        hr_err_right = [0.01, 0.01]
        all_match, results = compare_hardness_ratios(
            hr_left, hr_err_left, hr_right, hr_err_right, n_sigma=1.0,
        )
        assert not all_match


class TestLensMass:
    """测试透镜质量计算"""

    def test_positive_mass(self):
        M = lens_mass(dt_ms=8.82, f=2.0)
        assert M > 0

    def test_mass_scales_with_delay(self):
        M1 = lens_mass(dt_ms=5.0, f=2.0)
        M2 = lens_mass(dt_ms=10.0, f=2.0)
        assert np.isclose(M2 / M1, 2.0, rtol=0.01)

    def test_mass_in_solar_units(self):
        M = lens_mass(dt_ms=8.82, f=1.5)
        # 对于这些参数，质量应在几十到几百太阳质量量级
        assert 1 < M < 1e6


# ============================================================
# fpbh_data 模块
# ============================================================

class TestZM:
    """测试红移估计函数"""

    def test_positive_dm(self):
        z = zm(500)
        assert 0 < z < 5

    def test_small_dm(self):
        z = zm(10)
        assert 0 < z < 0.1

    def test_negative_dm(self):
        z = zm(-100)
        assert np.isnan(z)

    def test_monotonic(self):
        z1 = zm(200)
        z2 = zm(800)
        assert z2 > z1
