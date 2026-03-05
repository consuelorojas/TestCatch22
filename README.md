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
├── main_pipeline.py            # Main script to run the experiment
├── data/
│   └── dataset.py              # Creation of dataset, labeling and folds
├── signals/
│   ├── sine.py                 # Noisy sine wave generation
│   ├── fhn.py                  # Noisy FHN signal generator
│   └── dispatcher.py           # Maps model names to signal generators
├── FFTs/
|   ├── sine_dyn/            # FFT representations for sine wave dynamics 
|   ├── fhn_dyn/             # FFT representations for FHN dynamic
│   └── fhn_obs/             # FFT representations for FHN observational

├── preprocessing/
│   └── preprocessing.py       # PCA + scaling utilities and subsampling
├── features/
│   └── features.py            # pycatch22 feature extraction
├── models/
│   └── classification.py      # AUC, accuracy classification
└── README.md
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

FFT representations are computed and organized under the `FFTs/` directory (see structure above). They provide frequency-domain information (e.g. peak frequencies, spectral power) that can be used directly by classifiers or combined with the pycatch22 features.
