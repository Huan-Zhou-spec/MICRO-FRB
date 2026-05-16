This code is used to filter the data in the CHIME/FRB Catalog 2 that have been lensed by point mass lenses. We can refer to arXiv:......


Micro-Lensing_FRB/
└── modules                              # The collection of all basic computing modules
    ├── analysis_data.py                 # Analysis of functions with multiple peaks structure
    ├── load_data.py                     # Batch download of FRB dynamic spectrum data
    ├── plot_data.py                     # Plot FRB data
    ├── read_data.py                     # Read the data from CHIME/FRB and classify it
    ├── smooth_data.py                   # Smooth processing of FRB dynamic spectrum data
    ├── fpbh_data.py                     # constraints on PBH
    ├── hardness_data.py                 # Hardness test
    
        
└── SearchLensedFRB.py                   # Based on the point mass lensing model, selecting micro-lensing candidates
└── fpbh.py                              # constraints on PBH with the CHIME/FRB catalog2 data
└── Hardness_test.py                     # Hardness test on candidates

    
└── FRB_data                             # All input and output data sets
    ├── CHIME_cat2_frb                   # CHIME/FRB Catalogue 2 Data Classification
    ├── canfar_downloads                 # 340 sets of dynamic spectral data of multi-peaked FRBs
    ├──         
    

└── Figures                            # The collection of all output graphs
    ├──FRB_lensing_results_G           # The candidate analysis diagrams that were initially selected (Gaussian-smooth)
    ├──FRB_lensing_results_SG          # The candidate analysis diagrams that were initially selected ( Savitzky–Golay-smooth)


└──fpbh_bound                          # Various constraints on PBH

Continuously updated

Version 1.0.0
