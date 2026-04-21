#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 21:35:56 2026

@author: ubuntu
"""
from modules import load_and_clean_data, fpbh
import numpy as np
import matplotlib.pyplot as plt

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
# 设置全局字体和图形参数
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 18

def load_data(filename):
    """加载数据文件"""
    return np.loadtxt(filename, unpack=True)

def create_fill_data(x_data, y_max=1.0):
    """创建填充数据"""
    return [y_max] * len(x_data)

def setup_plot():
    """设置绘图参数"""
    fig, ax = plt.subplots(figsize=(14, 10))
    return fig, ax

def plot_constraint(ax, x_data, y_data, color, label, text_pos, rotation=0, alpha=0.2):
    """绘制单个约束曲线和填充区域"""
    ax.loglog(x_data, y_data, color=color, linestyle='-', linewidth=2.5, label=label)
    y_fill = create_fill_data(x_data)
    ax.fill_between(x_data, y_data, y_fill, facecolor=color, alpha=alpha)
    ax.text(text_pos[0], text_pos[1], label, fontsize=24,
            ha='center', va='bottom', rotation=rotation, color=color)

def main():
    """主函数"""
    # 加载所有约束数据
    data_files = {
        'LSS': "fpbh_bound/LSS.txt",
        'Dynamical': "fpbh_bound/Dynamical.txt", 
        'Accretion': "fpbh_bound/Accretion.txt",
        'Evaporation': "fpbh_bound/Evaporation.txt",
        'GWs': "fpbh_bound/GWs.txt",
        'Microlensing': "fpbh_bound/Microlensing.txt",
        'FRB': "fpbh_bound/FRB_fpbh_vs_ML.txt"
    }
    
    # 存储所有数据
    constraints = {}
    for name, filename in data_files.items():
        m, f = load_data(filename)
        if name == 'LSS':
            # LSS 文件中数据是以10为底的对数值
            constraints[name] = {
                'mass': 10**m,
                'f_pbh': 10**f,
            }
        else:
            constraints[name] = {
                'mass': m,
                'f_pbh': f,
            }
    
    # 设置约束参数（颜色、文本位置、旋转角度）
    constraint_params = {
        'LSS': {'color': 'gray', 'text_pos': (1e12, 9e-4), 'rotation': 75},
        'Dynamical': {'color': 'c', 'text_pos': (9e3, 2e-3), 'rotation': -75},
#        'Accretion': {'color': 'olive', 'text_pos': (1e1, 3e-4), 'rotation': 90},
#        'Evaporation': {'color': 'purple', 'text_pos': (1e-16, 7e-3), 'rotation': 85},
        'GWs': {'color': 'orange', 'text_pos': (1e-1, 7e-3), 'rotation': -60},
        'Microlensing': {'color': 'navy', 'text_pos': (9e-8, 2e-3), 'rotation': 60},
        'FRB': {'color': 'red', 'text_pos': (1e4, 7e-2), 'rotation': 0}
    }
    
    # 创建图形
    fig, ax = setup_plot()
    
    # 绘制所有约束
    for name, params in constraint_params.items():
        data = constraints[name]
        plot_constraint(
            ax, data['mass'], data['f_pbh'],
            params['color'], name,
            params['text_pos'], params['rotation']
        )
    
    # 设置坐标轴和标签
    ax.set_xlabel(r'$M_{\rm PBH}~(M_{\odot})$', labelpad=10, fontsize=24)
    ax.set_ylabel(r'$f_{\rm PBH}$', labelpad=10, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    
    # 设置坐标范围
    ax.set_xlim(1e-10, 1e14)
    ax.set_ylim(1e-4, 1)
    
    # 添加图例
    ax.legend(fontsize=21, loc='lower left', framealpha=0.9)
    
    # 添加网格
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # 调整布局并显示
    plt.tight_layout()
    
    # 保存图形
    plt.savefig("./fpbh_bound/fpbh.pdf", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()

    
    
    
    