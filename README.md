# **Catch22 test limitations (title pending)**

Here we present a modular Python pipeline for generating, transforming, and classifying moisy time series signals, including synthetic sine waves and FitzHugh-Nagumo (FHN) models. The system supports different preprocessing and classification configurations.

## **Features**
* **Signal generation**: Modular system for noisy sine and FHN signals.
* **Feature extraction**: Uses ''pycatch22'' for time series feature representation
* **Dimensionality reduction**: Built-in PCA with scaling
* **Classification**: SVM-based, but accomodable to any classifier with probability outputs.
* **Evaluation**: AUC metricts and Accuracy are supported.

## **Structure**

```.
├── main_pipeline.py            # Main script to run the experiment
├── data
|    └── dataset.py              # Creation of dataset, labeling and folds.
├── signals/
│   ├── sine.py                 # Noisy sine wave generation
|   ├── fhn.py                  # Noisy FHN signal generator
│   └── dispatcher.py         	# Maps model names to signal generators
├── preprocessing/
│   └── preprocessing.py       # PCA + scaling utilities and subsampling
├── features/
│   └── features.py            # pycatch22 feature extraction
├── models/
│   └── classification.py      # AUC, ACC classification
└── README.md
```

## **How it works**
Each experiment evaluates classification performance under 4 cofigurations:
1. Raw signals
2. Raw signals + PCA
3. Feature-extracted data
4. Feature-extracted data + PCA
All configuration share identical cross-validations splits for fair comparison.

## **TO-DOs**
- [] Add support for additional classifiers (RandomForest, KNN, etc.)
- [] Extend to real-world physiological signals
- [] Add CLI/argparse support
