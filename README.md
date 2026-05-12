This code is used to filter the data in the CHIME/FRB Catalog 2 that have been lensed by point mass lenses. We can refer to arXiv:......


Micro-Lensing_FRB/
└── modules                              # The collection of all basic computing modules
    ├── analysis_data.py                 # Analysis of functions with multiple peaks structure
    ├── load_data.py                     # Batch download of FRB dynamic spectrum data
    ├── plot_data.py                     # Plot FRB data
    ├── read_data.py                     # Read the data from CHIME/FRB and classify it
    ├── smooth_data.py                   # Smooth processing of FRB dynamic spectrum data
    ├── fpbh_data.py                     # constraints on PBH
    
                
└── FRB_data.py                          # The main function for classifying and downloading FRB data
└── FRB_Plots.py                         # The main functions for various drawing programs
└── SearchLensedFRB.py                   # Based on the point mass lensing model, selecting micro-lensing candidates
└── fpbh.py                              # constraints on PBH with the CHIME/FRB catalog2 data
└── 

    
└── FRB_data                             # All input and output data sets
    ├── CHIME_cat2_frb                   # CHIME/FRB Catalogue 2 Data Classification
    ├── canfar_downloads                 # 340 sets of dynamic spectral data of multi-peaked FRBs
    ├──         
    

└── Figures                            # The collection of all output graphs
    ├──canfar_downloads                # The dynamic spectrograms of more than 340 peak structures of FRB (Fast Radio Burst) downloaded from the CHIME/FRB website        
    ├──FRB_lensing_results             # The 14 candidate analysis diagrams that were initially selected


└──fpbh_bound                          # Various constraints on PBH

Continuously updated

Version 1.0.0
