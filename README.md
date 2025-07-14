# Uncertainty Analysis through Machine Learning in Colombian Construction MSMEs
Analysis of uncertainty sources in construction projects using a combination of data resampling, feature importance with Random Forests, and rule extraction with Classification Trees. The project focuses on micro, small, and medium-sized enterprises (MSMEs) in the construction sector.

## Overview

The code supports:
- Loading and preparing labeled datasets.
- Balancing the dataset based on class distribution using bootstrapped resampling.
- Statistical validation of resampling with Mann-Whitney U tests.
- Feature importance analysis using 1000 bootstrap iterations of Random Forests.
- Extraction and visualization of the most frequent classification tree from bootstrap samples.
- Generation of customized plots and export of decision trees to PDF using Graphviz.

The classification task focuses on distinguishing between higher and lower perceived uncertainty levels, based on several uncertainty sources:
- Organizational
- Activity Durations
- Resources Use
- Changes in Requirements & Quality
- Resource Availability
- Logistics
- Environmental
- Sociopolitical
- Market
- Technological

## Requirements

To run the code, you will need:

- Python 3.8+
- Required libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - graphviz
  - scipy

## How to Use
Data is available only by requesting it to the author of the code.
Note: The datasets and sources analyzed are derived from surveys and expert assessments related to the Colombian construction sector. The results and decision trees are intended for exploratory insight rather than production deployment.

## Example of use: 

X_encoded, y, df_sig, df_aug, df_dist = preparar_datos('your_data.csv', num_merges)
df_plot = bootstrap_rf_ct(X_encoded, y, 'Organizational_uncertainty', 'organizational')

## Outputs:
- Feature importance plots with error bars and frequency color scales.
- PDF files of the most frequent classification tree.
- Statistical summaries of the resampling differences. 

## Applications

- Academic research on uncertainty in construction project management.
- Visual and quantitative support for expert-based decision-making.
- Methodological tool for resampling and robustness testing using ensemble methods.

## Author

Cristian Tobar
March 2025

