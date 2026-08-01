# FDIA-SLC Joint Framework

This repository is associated with the manuscript:

**A Joint Framework for Measurement Error Detection, Missing Data Imputation, and FDIA/SLC Classification in Power Systems**

## Content

FDIA-SLC-Joint-Framework/
├── anomaly_detection/
│   ├── compute_residual.m               # compute EKF prediction residuals
│   ├── dual_threshold_detection.m       # dual-threshold measurement error detection
│   ├── ekf_imputation.m                 # EKF-based missing/error data imputation
│   ├── compute_sdm.m                    # SDM calculation
│   │
│   ├── full_flowchart_demo/
│   │   ├── main_AC_EKF.m                # main MATLAB demo
│   │   ├── wls.m                        # WLS state estimation
│   │   ├── wls_z.m                      # AC measurement model
│   │   ├── H_matrix.m                   # measurement Jacobian
│   │   ├── Linear_Exponential_Smoothing.m
│   │   ├── level_slope_calculation.m
│   │   └── readme.txt
│   │
│   └── generate_anomaly/
│       ├── generate_FDI/
│       │   └── generate_fdia.m          # FDIA data generation
│       └── generate_SLC/
│           └── generate_slc.m           # SLC data generation
│
├── data_generation/
│   └── generate_load/
│       ├── get_data.m                   # load-profile preprocessing
│       ├── generate_noise.m             # measurement-noise generation
│       ├── PF_calculate.m               # load allocation and power-flow calculation
│       └── load_values_20040101.mat
│
├── dataset/
│   ├── single_attack_201-Dim_Features.csv
│   ├── multi_attack_201-Dim_Features.csv
│   ├── single_attack_FDI_SLC.csv
│   ├── multi_attack_FDI_SLC.csv
│   ├── Active_Power.xlsx
│   ├── Reactive_Power.xlsx
│   ├── Voltage_Mag.xlsx
│   └── Voltage_Phase.xlsx
│
├── GEFCOM2012_Data/
│   ├── Load/                            # GEFCOM 2012 load data
│   └── Wind/                            # GEFCOM 2012 wind data
│
├── functions/
│   ├── classification.py                # LR, KNN, RF, and XGB classifiers
│   └── data_preparation.py              # data split and result-saving utilities
│
├── single_node/
│   └── SE_identification/
│       ├── SE_identification_1.ipynb    # single-node classification notebook
│       ├── classification_models/       # PCA-based trained models
│       ├── classification_models_1/     # all-feature trained models
│       ├── cross_validation/
│       │   ├── single_node_cv.py        # repeated stratified 5-fold CV
│       │   └── readme.txt
│       ├── score_1/                     # saved evaluation scores
│       └── time_1/                      # saved training times
│
├── multi_node/
│   └── SE_identification/
│       ├── SE_identification.ipynb      # multi-node classification notebook
│       ├── classification_models/       # all-feature and PCA-based models
│       ├── cross_validation/
│       │   ├── multi_node_cv.py         # repeated stratified 5-fold CV
│       │   └── readme.txt
│       ├── score/                       # saved evaluation scores
│       └── time/                        # saved training times
│
└── README.md                           # repository description

## Planned contents

This repository will be updated to include:

1. scripts for FDIA and SLC scenario generation;
2. scripts for measurement error detection and missing data imputation;
3. scripts for feature construction and PCA-based feature extraction;
4. machine-learning classification scripts;
5. generated FDIA/SLC datasets.

## Data source

The raw load data used in this study are derived from the public 2012 Global Energy Forecasting Competition (GEFC) dataset.

## Citation

After publication, the DOI and citation information of the paper will be added here.

