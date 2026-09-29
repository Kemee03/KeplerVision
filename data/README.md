# Kepler light-curve data

This project uses the labeled Kepler time-series dataset from your notebook. The two files are in this folder:

- `exoTrain.csv`: 5,087 light curves used for model fitting.
- `exoTest.csv`: 570 light curves held out for evaluation.

Each CSV has 3,198 columns: `LABEL` and `FLUX.1` through `FLUX.3197`. Source label `1` means no planet; source label `2` means a planet. KeplerVision maps those to model labels `0` and `1` respectively.

The source data is present locally but ignored by Git because the training file is large. Do not commit raw data or Kaggle API credentials. The test file contains only five planet examples, so its positive-class metrics are highly uncertain.
