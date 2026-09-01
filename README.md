# MDH_RamanML

This repository contains two independent machine-learning codes developed for the project **“Machine Learning-Assisted Detection of Water Contaminants Using Conventional Raman Spectroscopy.”** Both use Raman spectra as input, but address different prediction tasks. In the same folder as the code, you will already find the trained models (The filtering model and the regressor/classifier model) with their respective graphs and points (in txt files) to evaluate the models performance.

## Goal

The **classification code** identifies water contaminants from Raman spectra. It can be used to distinguish pure from contaminated water and to determine which contaminant is present. The **regression code**, independently, predicts the concentration of a known contaminant from its Raman spectrum. 

## Workflow

Both codes follow a similar machine-learning approach but are trained and executed separately. Raman spectra are preprocessed (SVN, SGolay, LOESS, trained CAE model on the selected spectra) and divided into **80% training and 20% external test data**. TPOT then performs **5-fold stratified cross-validation** on the training set and searches for the best machine-learning pipeline, including further preprocessing, optional PCA, model selection, hyperparameter optimisation, and model stacking. The selected pipeline is finally evaluated on the independent test set.

## Usage

Run the desired script depending on the task:

    python <regression_script>.py

or

    python <classification_script>.py

Provide the requested Raman data folder and select the preprocessing options when prompted. Each code trains its TPOT model independently and automatically saves the selected pipeline, performance metrics, predictions, and result plots.
