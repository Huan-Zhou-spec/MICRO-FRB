#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 21:35:56 2026

@author: ubuntu
"""
from modules import load_and_clean_data, fpbh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch

'''
# ==================== 使用示例 ====================
if __name__ == "__main__":
    repeater_file = 'FRB_data/CHIME_cat2_frb/chimefrbcat2_unique_first_repeaters.npy'
    non_repeater_file = 'FRB_data/CHIME_cat2_frb/chimefrbcat2_unique_non_repeater.npy'

    data = load_and_clean_data(repeater_file, non_repeater_file)
    if len(data['DM_wo_mw']) == 0:
        print("没有有效数据，退出")
        exit()

    # 质量取值：从 1 到 1000，对数均匀分布（30 个点）
    ML_values = np.logspace(0, 5, 50)  # 10^0 = 1 到 10^3 = 1000
    results = []

    print("\n开始计算不同透镜质量对应的 f_up...")
    for ML in ML_values:
        f_up = fpbh(ML, data['DM_wo_mw'], data['wi'], data['snr'])
        results.append([ML, f_up])

    results = np.array(results)

    # 保存结果
    output_file = 'fpbh_bound/FRB_fpbh_vs_ML.txt'
    np.savetxt(output_file, results, header='#FRB:ML_Msun, f_up', delimiter=' ', comments='')
    print(f"\n结果已保存至 {output_file}")

    
'''    

plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 18


def load_data(filename):
    return np.loadtxt(filename, unpack=True)


def create_fill_data(x_data, y_max=1.0):
    return [y_max] * len(x_data)


def setup_plot():
    fig, ax = plt.subplots(figsize=(14, 10))
    return fig, ax


def plot_constraint(ax, x_data, y_data, color, linestyle, marker, label, text_pos, rotation=0, alpha=0.2):
    ax.loglog(x_data, y_data, color=color, linestyle=linestyle, marker=marker, linewidth=2.5, label=label)
    y_fill = create_fill_data(x_data)
    ax.fill_between(x_data, y_data, y_fill, facecolor=color, alpha=alpha)
    ax.text(text_pos[0], text_pos[1], label, fontsize=24,
            ha='center', va='bottom', rotation=rotation, color=color)


def main():
    data_files = {
        'LSS': "fpbh_bound/LSS.txt",
        'Dynamical': "fpbh_bound/DynEff.txt",
        'Accretion': "fpbh_bound/acc.txt",
        'GWs': "fpbh_bound/LIGO.txt",
        'Microlensing': "fpbh_bound/OGLE.txt",
        'FRB': "fpbh_bound/FRB_fpbh_vs_ML.txt"
    }
    constraints = {}
    for name, filename in data_files.items():
        m, f = load_data(filename)
        if name == 'FRB':
            constraints[name] = {'mass': m, 'f_pbh': f}
        else:
            constraints[name] = {'mass': 10**m, 'f_pbh': 10**f}

    constraint_params = {
        'LSS': {'color': 'gray', 'linestyle': '-',  'marker': '^', 'text_pos': (5e11, 6e-4), 'rotation': 75},
        'Dynamical': {'color': 'gray', 'linestyle': '--', 'marker': 'None', 'text_pos': (3e5, 5e-4), 'rotation': -75},
        'Accretion': {'color': 'gray', 'linestyle': '-.', 'marker': 'None', 'text_pos': (5e2, 1.5e-5), 'rotation': 90},
        'GWs': {'color': 'gray', 'linestyle': '--', 'marker': '*', 'text_pos': (7e-1, 5e-4), 'rotation': -60},
        'Microlensing': {'color': 'gray', 'linestyle': ':', 'marker': 'None', 'text_pos': (9e-4, 3e-3), 'rotation': 0},
        'FRB': {'color': 'red', 'linestyle': '-', 'marker': 'None', 'text_pos': (1e4, 7e-2), 'rotation': 0}
    }

    fig, ax = setup_plot()
    for name, params in constraint_params.items():
        data = constraints[name]
        plot_constraint(ax, data['mass'], data['f_pbh'],
                        params['color'], params['linestyle'], params['marker'], name,
                        params['text_pos'], params['rotation'])

    # ========== 定义两组线段（分别指定颜色，支持斜线段） ==========
    segments = [
        {'x1': 539, 'x2': 609, 'y1': 0.042, 'y2': 0.042, 'color': 'green', 'label': None},
        {'x1': 280, 'x2': 467, 'y1': 0.043, 'y2': 0.042, 'color': 'blue', 'label': None}
    ]

    # 在主图中绘制所有线段及端点
    for seg in segments:
        x1, x2, y1, y2, color = seg['x1'], seg['x2'], seg['y1'], seg['y2'], seg['color']
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.5,
                label=seg['label'] if seg['label'] else "")
        ax.scatter([x1, x2], [y1, y2], color=color, s=60, edgecolors='black', zorder=10)

    # ========== 子图（放大区域） ==========
    inset_ax = fig.add_axes([0.8, 0.6, 0.3, 0.3])
    inset_ax.set_xlim(200, 700)
    inset_ax.set_ylim(0.040, 0.044)

    # 在子图中绘制所有线段及端点标记（使用各自颜色）
    seg_labels = {
    'green': 'FRB 20211115A',
    'blue': 'FRB 21090131D'
    }

    # 在子图中绘制所有线段及端点标记（使用各自颜色）
    for seg in segments:
        x1, x2, y1, y2, color = seg['x1'], seg['x2'], seg['y1'], seg['y2'], seg['color']
        inset_ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)
        inset_ax.scatter([x1, x2], [y1, y2], color=color, s=80, edgecolors='black', zorder=10)
        # 在线段上方中点标注
        x_mid = (x1 + x2) / 2-30
        y_mid_seg = (y1 + y2) / 2
        inset_ax.text(x_mid, y_mid_seg - 0.0006, seg_labels.get(color, color),
        color=color, fontsize=12.5, ha='center', va='bottom', fontweight='bold')

        


    #inset_ax.set_xlabel('$M_{\\rm PBH}~(M_{\\odot})$', fontsize=24)
    #inset_ax.set_ylabel('$f_{\\rm PBH}$', fontsize=24)
    inset_ax.grid(True, linestyle='--', alpha=0.5)

    # ========== 为主图中的第一个线段绘制矩形框和连接线 ==========
    first = segments[0]
    x1, x2, y1, y2 = first['x1'], first['x2'], first['y1'], first['y2']
    y_mid = (y1 + y2) / 2
    rect = Rectangle((x1 - 20, min(y1, y2) - 0.0005), (x2 - x1) + 40, abs(y2 - y1) + 0.001,
                     facecolor='none', edgecolor='black', linestyle='--', linewidth=1)
    ax.add_patch(rect)

    # 连接线1（指向子图右上角）
    xy_main = (x1 + 100, y_mid + 0.0003)
    xy_inset = (0.26, 0.66)
    con = ConnectionPatch(xyA=xy_main, xyB=xy_inset, coordsA='data', coordsB='axes fraction',
                          axesA=ax, axesB=inset_ax, color='black', linewidth=1.5,
                          arrowstyle='->', shrinkA=5, shrinkB=5)
    fig.add_artist(con)

    # 连接线2（指向子图左下角）
    xy_main2 = (x1 - 300, y_mid - 0.0003)
    xy_inset2 = (0.75, 0.45)
    con2 = ConnectionPatch(xyA=xy_main2, xyB=xy_inset2, coordsA='data', coordsB='axes fraction',
                           axesA=ax, axesB=inset_ax, color='black', linewidth=1.5,
                           arrowstyle='->', shrinkA=5, shrinkB=5)
    fig.add_artist(con2)

    # ========== 坐标轴设置 ==========
    ax.set_xlabel(r'$M_{\rm PBH}~(M_{\odot})$', labelpad=10, fontsize=24)
    ax.set_ylabel(r'$f_{\rm PBH}$', labelpad=10, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.set_xlim(1e-5, 1e14)
    ax.set_ylim(1e-5, 1)
    ax.legend(fontsize=21, loc='lower left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig("./fpbh_bound/fpbh.pdf", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()







    
    
    
    