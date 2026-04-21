#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 20:34:56 2026

@author: ubuntu
"""

from .read_data import separate_by_repeater, process_frb_data, quick_summary, list_saved_files, load_and_display
from .read_data import explore_h5_structure, read_frb_dynamic_spectrum, print_data_summary
from .plot_data import (plot_scatter_with_choice,
                        plot_dynamic_spectrum, 
                        plot_ks_heatmap,plot_qq_matrix, 
                        plot_autocorr_with_spikes)
from .load_data import download_with_urllib, download_with_wget, check_system_command, download_canfar_file
from .analysis_data import (process_data_ts, calculate_snr_peaks, 
                            extract_peak_spectra,extract_noise_spectra,
                            compare_spectra_ks, compute_autocorr_with_spikes)
from .smooth_data import smooth_time_dimension, find_optimal_time_smooth, smooth_frb_data


from .fpbh_data import load_and_clean_data, fpbh
__version__ = '1.0.0'