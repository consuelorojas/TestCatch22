# **Catch22 test limitations (title pending)**

Here we present a modular Python pipeline for generating, transforming, and classifying noisy time series signals, including synthetic sine waves and FitzHugh-Nagumo (FHN) models. The system supports different preprocessing and classification configurations.

## **Features**
* **Signal generation**: Modular system for noisy sine and FHN signals.
* **Feature extraction**: Uses ''pycatch22'' for time series feature representation.
* **Dimensionality reduction**: Built-in PCA with scaling.
* **Classification**: SVM-based, but accommodating any classifier with probability outputs.
* **Evaluation**: AUC metrics and accuracy are supported.

## **Structure**

```
├── README.md
├── runner.py                   # experiment runner
├── requirements.txt            # dependencies
├── report.mplstyle             # plotting styles
├── data/                       # dataset creation and fold management
│   └── dataset.py
├── signals/                    # signal generators (sine, FHN, lorenz, etc.)
│   ├── sine.py
│   ├── fhn.py
│   ├── fhn_obs.py
│   └── dispatcher.py               # utility to generate datasets based on configurations
├── preprocessing/              # preprocessing utilities (scaling, subsampling)
│   └── preprocessing.py
├── features/                   # feature extraction (pycatch22 wrappers)
│   └── features.py
├── FFTs/                       # frequency-domain transforms and results
│   ├── fhn_dyn/
│   ├── fhn_obs/
│   └── sine/
├── models/                     # classification and evaluation code
│   └── classification.py
├── figures/                    # plotting styles and output figures
├── notebooks/                  # analysis and experiments in notebooks
├── results/                    # structured experiment outputs
└── sweeps/                     # parameter sweep scripts. Inside contains the scripts that run the experiments with the different configurations; results are stored in the `results/` directory
```

## **How it works**
Each experiment evaluates classification performance under multiple representations and preprocessing configurations. In addition to time-domain and feature-based inputs, we compute frequency-domain (FFT) representations and make them available as model inputs.

Typical configurations include:
1. Raw signals
2. Raw signals + PCA
3. FFT representations (magnitude)
4. FFT representations (magnitude) + PCA
5. Feature-extracted data (pycatch22)
6. Feature-extracted data + PCA

All configurations share identical K-folds splits for a fair comparison.

FFT representations are computed and organized under the `FFTs/` directory (see structure above)