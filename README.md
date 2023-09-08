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

## Dataset

The datasets, both for training and testing, were majorly extracted from:
- Demographic and Health Surveys (DHS) covering 59 countries.
- Diverse sources of satellite imagery.

An additional dataset from the World Bank was used to improve the model's efficacy.

## Features and Target

At its core, this is a multi-target regression task where dataset features are employed to predict the following MCH indicators:
- `Mean_BMI`
- `Median_BMI`
- `Unmet_Need_Rate`
- `Under5_Mortality_Rate`
- `Skilled_Birth_Attendant_Rate`
- `Stunted_Rate`

## Project Structure and Usage
All the data files have been excluded
```
project_root/
│
├── README.md
├── data/
│   ├── external/
│   │   ├── WBdata_Sample/
│   │   │   ├── WB_Sample_workflow.py
│   │   │   └── WorldBank_sample.csv
│   │   ├── WBdata_gee/
│   │   │   ├── WB_gee_workflow.py
│   │   │   └── WorldBank_features.csv
│   ├── interim/
│   │   └── ... (intermediate data)
│   ├── processed/
│   │   └── ... (final canonical data sets)
│   └── raw/
│       └── ... (original immutable data dump)
│       └── GEE_Features.parquet
│       └── training_label.csv
│
├── models/
│   └── ... (trained and serialized models)
│
├── notebooks/
│   ├── final_workflow.ipynb   <- Your final workflow notebook
│   ├── imputation__evaluate.ipynb
│   └── hyperparameter-tuning.ipynb
│
├── reports/
│   ├── mid_term_report.pdf    <- Mid-term report
│   └── final_report.pdf       <- Final report
│
└── requirements.txt
```
## Authors

- [@VedantZope](https://www.github.com/VedantZope)

## Acknowledgements

- [Professor Pascal Geldsetzer](https://profiles.stanford.edu/pascal-geldsetzer) for his invaluable guidance throughout this project.
- [Doctor Haojie Wang](https://scholar.google.com.hk/citations?user=oU5bH20AAAAJ&hl=en) for generously providing data and consistent assistance during the project's lifecycle.
Project Organization
--------
