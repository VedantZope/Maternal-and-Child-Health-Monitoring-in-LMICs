Maternal and Child Health (MCH) Indicator Estimation Using Satellite Imagery
==============================
This repository hosts a machine-learning project focused on estimating pivotal MCH indicators leveraging satellite imagery and geotagged data.

## Background

Many low- and middle-income countries (LMICs) lack reliable death registers, leading to significant gaps in the civil registration and vital statistics of maternal and child deaths. The traditional method of gathering basic MCH indicators and coverage of essential MCH services, such as childhood vaccinations, is through expensive nationally representative household surveys. However, these surveys often fall short in offering granular insights into the disparities of MCH indicators across communities, mainly due to their limited sample size and the outdated nature of their data.

## Status

![Project Status: In Progress](https://img.shields.io/badge/Project%20Status-In%20Progress-orange)
![Contributors](https://img.shields.io/github/contributors/VedantZope/Maternal-and-Child-Health-Monitoring-in-LMICs.svg)
![Number of Commits](https://img.shields.io/github/commit-activity/y/VedantZope/Maternal-and-Child-Health-Monitoring-in-LMICs.svg)

## Objective

The overarching goal of this research is to validate the capability of machine learning, combined with satellite imagery and other geotagged public data, in accurately estimating essential MCH indicators.

### Hypothesis

1. The dataset can offer precise estimates of village and neighborhood-level indicators, such as child undernutrition, anemia in children and women of reproductive age, child and maternal mortality, and episodes of childhood illnesses.
2. The indicators can be pinpointed accurately both for specific timeframes and longitudinally across a span of time.

## Authors

- [@VedantZope](https://www.github.com/VedantZope)

## Dataset

The datasets, both for training and testing, were majorly extracted from:
- Demographic and Health Surveys (DHS) covering 59 countries.
- Diverse sources of satellite imagery.

Additional datasets from sources [xxx,yyy] were used to improve the model's efficacy.

## Features and Target

At its core, this is a multi-target regression task where dataset features are employed to predict the following MCH indicators:
- `Mean_BMI`
- `Median_BMI`
- `Unmet_Need_Rate`
- `Under5_Mortality_Rate`
- `Skilled_Birth_Attendant_Rate`
- `Stunted_Rate`

## Project Structure and Usage
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Acknowledgements

- [Professor Pascal Geldsetzer](https://profiles.stanford.edu/pascal-geldsetzer) for his invaluable guidance throughout this project.
- [Doctor Haojie Wang](https://scholar.google.com.hk/citations?user=oU5bH20AAAAJ&hl=en) for generously providing data and consistent assistance during the project's lifecycle.
Project Organization
--------
