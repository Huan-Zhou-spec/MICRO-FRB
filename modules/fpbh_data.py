#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:06:17 2026

@author: ubuntu
"""

import numpy as np
from scipy import integrate
from colossus.cosmology import cosmology
from scipy.optimize import bisect

# ==================== 宇宙学参数 ====================
cosmo = cosmology.setCosmology('planck18')
G = 6.67e-11
Msun = 1.9891e30
c = 299792458
H0 = 67.66
h0 = 0.6766
Mpc = 3.08413e22
omgm = 0.2621

# ==================== 核心函数 ====================
def zm(dm_wo_mw):
    """
    根据总 DM 和银河系贡献 DM 估算红移 z。
    公式来源：z = [ (a-b2) + sqrt((b2-a)^2 - 4*b2*(c2-a)) ] / (2*b2)
    其中 a = dm - dmm, b2=855, c2=200
    """
    a = dm_wo_mw
    b2 = 855.0
    c2 = 200.0
    # 计算判别式
    delta = (b2 - a)**2 - 4 * b2 * (c2 - a)
    # 只保留物理上有实数解且 z>0 的情况
    if delta < 0:
        return np.nan
    z = ((a - b2) + np.sqrt(delta)) / (2 * b2)
    if z <= 0:
        return 1e-3
    return z

def ymi(zl, ML, w, yma=6):
    def dt(x):
        return (4 * G * ML * Msun / c**3) * (1 + zl) * (
            x/2 * np.sqrt(x**2 + 4) + np.log((np.sqrt(x**2 + 4) + x) / (np.sqrt(x**2 + 4) - x))
        ) * 1000 - w
    try:
        return bisect(dt, 0, yma, xtol=1e-12)
    except ValueError:
        return yma

def dtau(zs, zl, ML, w, Rf):
    Dls = (cosmo.comovingDistance(0, zs) - cosmo.comovingDistance(0, zl)) / (h0 * (zs + 1))
    Ds = cosmo.comovingDistance(0, zs) / (h0 * (zs + 1))
    Dl = cosmo.comovingDistance(0, zl) / (h0 * (zl + 1))
    hz = cosmo.Hz(zl)
    ymax = np.sqrt((1 + Rf) / np.sqrt(Rf) - 2)
    ymin = ymi(zl, ML, w)
    if ymax >= ymin and Rf >= 1:
        return 1.5 * omgm * H0**2 / hz / c * Dls * Dl / Ds * (ymax**2 - ymin**2) * (1 + zl)**2 * 1000
    return 0.0

def tau(zs, ML, w, Rf):
    result, _ = integrate.quad(lambda zl: dtau(zs, zl, ML, w, Rf), 0, zs)
    return result

def fpbh(ML, DM_vals, wi_vals, snr_vals):
    """计算 fpbh，输入为清理后的数组"""
    tautrack = []
    for dm, w, snr in zip(DM_vals, wi_vals, snr_vals):
        zs = zm(dm)   # 需要用户提供 zm 函数
        if zs >= 0:
            ts = tau(zs, ML, w, snr/10)
        else:
            ts = 0
        tautrack.append(ts)
    ttot = np.sum(tautrack)
    f = 1.0 / ttot if ttot > 0 else np.inf
    #Nlen2=np.log(1/(1-0.95))
    Nlen2 = 1
    f_up=np.log(1-Nlen2/len(DM_vals))/np.log(1-1/len(DM_vals))*f
    print(ML, f_up)
    return f_up

# ==================== 数据加载与清理 ====================
def load_and_clean_data(repeater_file, non_repeater_file):
    """加载两个npy文件，只保留 dm_exc_ymw16, wi, snr 均有效的源"""
    repeater_data = np.load(repeater_file, allow_pickle=True)
    non_repeater_data = np.load(non_repeater_file, allow_pickle=True)

    def clean_data(data, name):
        print(f"\n=== 检查 {name} 的无效数据 ===")
        # 必须存在的字段（只关心这三个）
        required_fields = ['dm_exc_ymw16', 'width_fitb', 'snr_fitb']
        for field in required_fields:
            if field not in data.dtype.names:
                raise KeyError(f"数据中缺少必需字段: {field}，无法继续")

        valid_mask = np.ones(len(data), dtype=bool)
        invalid_indices = []

        # 检查 snr
        snr_vals = data['snr_fitb']
        snr_invalid = ~np.isfinite(snr_vals) | (snr_vals <= 0)
        if np.any(snr_invalid):
            print(f"  snr: 发现 {np.sum(snr_invalid)} 个无效值 (NaN/Inf/<=0)")
            for idx in np.where(snr_invalid)[0]:
                if idx not in invalid_indices:
                    invalid_indices.append(idx)
                info = f"    索引 {idx} | snr: {snr_vals[idx]}"
                if 'tns_name' in data.dtype.names:
                    info += f" | TNS: {data['tns_name'][idx]}"
                print(info)
            valid_mask &= ~snr_invalid

        cleaned = data[valid_mask]
        print(f"  清理后剩余 {len(cleaned)} 个有效数据点 (共移除 {len(invalid_indices)} 个无效点)")
        return cleaned

    cleaned_repeater = clean_data(repeater_data, "重复暴")
    cleaned_non_repeater = clean_data(non_repeater_data, "非重复暴")

    all_data = np.concatenate([cleaned_repeater, cleaned_non_repeater])
    print(f"\n合并后总有效源数: {len(all_data)}")

    # 提取所需字段
    return {
        'DM_wo_mw': all_data['dm_exc_ymw16'],
        'wi': all_data['width_fitb']*1e3,
        'snr': all_data['snr_fitb']
    }






















