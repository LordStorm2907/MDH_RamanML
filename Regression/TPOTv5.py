#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TPOT 0.12.2 Raman regression script.

Main idea:
    - TPOT does internal CV on the TRAIN set.
    - This script creates an external train/test split so you get real TEST R2.
    - Optional SNV normalization of spectra.
    - Optional Savitzky-Golay.
    - Optional LOESS.
    - Optional CAE.
    - PCA is inside TPOT config_dict, so TPOT decides whether to use PCA or not.
    - Optional y normalization fitted only on TRAIN and inverted for metrics.
    - Saves the trained TPOT object with joblib.
    - Saves the preprocessing objects and results.
    - Makes graphs similar to GPR3/GPR4: prediction scatter, violin, concentration histogram.
    - If the best TPOT pipeline contains PCA, it also plots PCA loadings.

Important for TPOT 0.12.2:
    - Use config_dict=... instead of search_space=...
    - Use scoring=... instead of scorers/scorers_weights.
    - StackingEstimator is available through tpot.builtins.
    - model.export(...) is available again.
"""

import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split

from tpot import TPOTRegressor
from tpot.builtins import StackingEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, LassoLarsCV, ElasticNetCV

from time import perf_counter

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Conv1D, Input, MaxPooling1D, UpSampling1D
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    LOWESS_AVAILABLE = True
except Exception:
    LOWESS_AVAILABLE = False


# =========================================================
# User settings
# =========================================================
WN_MAX = 4000.0
WN_MIN_DEFAULT = 0.0
RANDOM_STATE = 0

TEST_SIZE = 0.20
TPOT_CV = 5
TPOT_MAX_TIME_MINS = 60
TPOT_MAX_EVAL_TIME_MINS = 10
TPOT_N_JOBS = 1  # safer on Mac/Anaconda/Dask. Try 2 or 3 later.
TPOT_SCORER = "r2"  # alternatives: "neg_root_mean_squared_error", "neg_mean_absolute_error"

SAVGOL_FRACTION = 0.01
SAVGOL_POLYORDER = 2
CAE_EPOCHS = 10
CAE_BATCH_SIZE = 32
LOESS_FRAC = 0.05
LOESS_IT = 3

tic = perf_counter()

# =========================================================
# TPOT 0.12.2 setup
# =========================================================
def make_tpot():
    """
    TPOT 0.12.2 custom config_dict.

    Important:
        - This is the old TPOT API.
        - Do not use get_search_space(...) here.
        - StackingEstimator is allowed in config_dict.
        - GaussianProcessRegressor is added explicitly.
    """

    # TPOT 0.x uses sklearn-style string keys.
    # The evolutionary search can combine preprocessing, PCA/no-PCA,
    # regressors, and StackingEstimator nodes.
    custom_config_dict = {
        # -----------------------------
        # Optional preprocessing
        # -----------------------------
        "sklearn.preprocessing.RobustScaler": {},
        "sklearn.preprocessing.StandardScaler": {},
        "sklearn.preprocessing.MinMaxScaler": {},

        "sklearn.feature_selection.VarianceThreshold": {
            "threshold": [0.0, 1e-8, 1e-6, 1e-4],
        },

        # PCA is optional because TPOT may or may not place this operator
        # in the final pipeline.
        "sklearn.decomposition.PCA": {
            "svd_solver": ["auto", "full"],
            "n_components": [0.80, 0.90, 0.95, 0.985],
        },

        # -----------------------------
        # Regressors
        # -----------------------------
        "sklearn.gaussian_process.GaussianProcessRegressor": {
            "alpha": [1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
            "normalize_y": [True, False],
            "n_restarts_optimizer": [0],
        },

        "sklearn.svm.SVR": {
            "kernel": ["rbf", "linear"],
            "C": [0.1, 1.0, 10.0, 100.0],
            "epsilon": [0.001, 0.01, 0.1, 0.2],
            "gamma": ["scale", "auto"],
        },

        "sklearn.ensemble.RandomForestRegressor": {
            "n_estimators": [100],
            "max_features": [0.3, 0.6, 1.0],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        },

        "sklearn.ensemble.ExtraTreesRegressor": {
            "n_estimators": [100],
            "max_features": [0.3, 0.6, 1.0],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        },

        "sklearn.ensemble.GradientBoostingRegressor": {
            "n_estimators": [100],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 5],
            "subsample": [0.7, 1.0],
        },

        "sklearn.neighbors.KNeighborsRegressor": {
            "n_neighbors": [2, 3, 5, 8, 12],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },

        "sklearn.linear_model.Ridge": {
            "alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        },

        "sklearn.linear_model.LassoLarsCV": {
            "normalize": [False],
        },

        "sklearn.linear_model.ElasticNetCV": {
            "l1_ratio": [0.1, 0.5, 0.9],
            "tol": [1e-4, 1e-3],
        },

        "tpot.builtins.StackingEstimator": {
            "estimator": [
                {
                    "sklearn.gaussian_process.GaussianProcessRegressor": {
                        "alpha": [1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
                        "normalize_y": [True, False],
                        "n_restarts_optimizer": [0],
                    },
                },
                {
                    "sklearn.svm.SVR": {
                        "kernel": ["rbf", "linear"],
                        "C": [0.1, 1.0, 10.0, 100.0],
                        "epsilon": [0.001, 0.01, 0.1, 0.2],
                        "gamma": ["scale", "auto"],
                    },
                },
                {
                    "sklearn.ensemble.RandomForestRegressor": {
                        "n_estimators": [100],
                        "max_features": [0.3, 0.6, 1.0],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "bootstrap": [True, False],
                    },
                },
                {
                    "sklearn.ensemble.ExtraTreesRegressor": {
                        "n_estimators": [100],
                        "max_features": [0.3, 0.6, 1.0],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "bootstrap": [True, False],
                    },
                },
                {
                    "sklearn.ensemble.GradientBoostingRegressor": {
                        "n_estimators": [100],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "max_depth": [2, 3, 5],
                        "subsample": [0.7, 1.0],
                    },
                },
                {
                    "sklearn.neighbors.KNeighborsRegressor": {
                        "n_neighbors": [2, 3, 5, 8, 12],
                        "weights": ["uniform", "distance"],
                        "p": [1, 2],
                    },
                },
                {
                    "sklearn.linear_model.Ridge": {
                        "alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
                    },
                },
                {
                    "sklearn.linear_model.LassoLarsCV": {
                        "normalize": [False],
                    },
                },
                {
                    "sklearn.linear_model.ElasticNetCV": {
                        "l1_ratio": [0.1, 0.5, 0.9],
                        "tol": [1e-4, 1e-3],
                    },
                },
            ],
        },
    }

    return TPOTRegressor(
        generations=5,
        population_size=50,
        offspring_size=None,
        mutation_rate=0.9,
        crossover_rate=0.1,
        scoring="r2",
        cv=5,
        max_time_mins=3000,
        max_eval_time_mins=100,
        n_jobs=26,
        config_dict=custom_config_dict,
        verbosity=2,
    )

def get_best_pipeline_text(model):
    """
    TPOT 0.x normally exposes fitted_pipeline_, but this function stays defensive.
    This function safely tries the common ones and returns a text summary.
    """
    possible_attrs = [
        "fitted_pipeline_",
        "optimized_pipeline_",
        "best_pipeline_",
        "pipeline_",
    ]

    lines = []
    lines.append("===== TPOT MODEL SUMMARY =====")
    lines.append(f"TPOT object type: {type(model)}")

    for attr in possible_attrs:
        if hasattr(model, attr):
            value = getattr(model, attr)
            lines.append(f"\n===== {attr} =====")
            lines.append(repr(value))

    if len(lines) == 2:
        lines.append("\nNo fitted pipeline attribute was found. The full TPOT object was still saved with joblib.")

    return "\n".join(lines) + "\n"


# =========================================================
# Basic utilities
# =========================================================
def ask_yes_no(question, default="y"):
    answer = input(f"{question} [{'Y/n' if default == 'y' else 'y/N'}]: ").strip().lower()
    if answer == "":
        answer = default
    return answer in {"y", "yes", "s", "si", "sí"}


def snv_rows(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std



def extract_concentration_g_ml(filename):
    match = re.search(
        r"(?P<value>[0-9]+(?:[_\.][0-9]+)*)\s*"
        r"(?P<unit>g|mg|ug|μg|mcg)"
        r"(?:\s*(?:/|per|_)\s*m[lL])?",
        filename,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Could not extract concentration from filename: {filename}")

    concentration = float(match.group("value").replace("_", "."))
    unit = match.group("unit").lower()

    if unit == "g":
        return concentration
    if unit == "mg":
        return concentration / 1000.0
    if unit in {"ug", "μg", "mcg"}:
        return concentration / 1_000_000.0

    raise ValueError(f"Unsupported concentration unit in filename: {filename}")


def load_raman_folder(data_folder,nbOfAverage,percentage):
    files = sorted(Path(data_folder).glob("**/*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_folder}")

    wn_ref = None
    X_rows = []
    y_values = []

    for file in files:
        data = np.loadtxt(file)#sino en google colab no iba , encoding="latin1"
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"Expected at least two numeric columns in {file}")

        wn = data[:, 0]
        spectra = data[:, 1:]

        spectraTemp = augment_spectra(np.array(spectra).T, wn, percentage)
        spectra = spectraTemp.T

        nbOfAverage = int(nbOfAverage)
        if nbOfAverage<=1:
            mean_spectra=np.array(spectra.copy())
        else:
            mean_spectra =[]
            for i in range(spectra.shape[0]//nbOfAverage):
                mean_spectra.append(np.mean(spectra[:,i*nbOfAverage:(i+1)*nbOfAverage],axis=1))
            mean_spectra = np.array(mean_spectra)
            mean_spectra = mean_spectra.T

        if wn_ref is None:
            wn_ref = wn
        elif len(wn) != len(wn_ref) or not np.allclose(wn, wn_ref):
            raise ValueError(f"Wavenumber axis differs in {file}")

        n_spectra = mean_spectra.shape[1]#mean spectra shape = (n_points, n_spectra) cause spectra = X.T
        X_rows.append(mean_spectra.T)
        y_values.extend([extract_concentration_g_ml(file.name)] * n_spectra)

    return wn_ref, np.vstack(X_rows), np.array(y_values, dtype=float)

def augment_spectra(
    X: np.ndarray,
    wn: np.ndarray,
    percentage: int,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Create augmented spectra by adding baseline tilt, spectral shift and noise.
    """

    n_aug_per_sample = X.shape[0] * percentage // 100#X.shape = (n_spectra, n_points)
    rng = np.random.default_rng(random_state)

    X = np.asarray(X, dtype=float)
    wn = np.asarray(wn, dtype=float).ravel()

    n_samples, n_points = X.shape
    augmented = np.zeros((n_aug_per_sample, n_points), dtype=float)

    baseline_axis = np.linspace(-1, 1, n_points)
    output_index = 0

    for output_index in range(n_aug_per_sample):
        idx = output_index % n_samples
        spectrum = X[idx]

        baseline = 0.002 * rng.standard_normal() * baseline_axis

        shifted_spectrum = np.interp(wn - rng.uniform(-0.8, 0.8),wn,spectrum + baseline,left=spectrum[0],right=spectrum[-1],)

        noise_level = 0.003 * np.std(spectrum, ddof=1)
        augmented[output_index] = shifted_spectrum + noise_level * rng.standard_normal(n_points)

    return np.concatenate([X, augmented], axis=0)

def apply_savgol(X, fraction=SAVGOL_FRACTION, polyorder=SAVGOL_POLYORDER):
    window = max(5, int(fraction * X.shape[1]))
    if window % 2 == 0:
        window += 1
    if window <= polyorder:
        window = polyorder + 3
    if window % 2 == 0:
        window += 1
    if window >= X.shape[1]:
        window = X.shape[1] - 1 if X.shape[1] % 2 == 0 else X.shape[1]

    print(f"Savitzky-Golay window = {window}")
    return savgol_filter(X, window_length=window, polyorder=polyorder, axis=1)


def apply_loess(X, wn, frac=LOESS_FRAC, it=LOESS_IT):
    if not LOWESS_AVAILABLE:
        print("statsmodels is not installed: LOESS skipped.")
        return X

    print(f"Applying LOESS: frac={frac}, it={it}")
    X_loess = np.empty_like(X, dtype=float)

    for i, spectrum in enumerate(X):
        X_loess[i] = lowess(
            endog=spectrum,
            exog=wn,
            frac=frac,
            it=it,
            return_sorted=False,
        )

    return X_loess


# =========================================================
# Optional CAE
# =========================================================
def build_cae(input_length):
    inp = Input(shape=(input_length, 1))

    x = Conv1D(16, 3, activation="relu", padding="same")(inp)
    x = MaxPooling1D(2, padding="same")(x)
    x = Conv1D(8, 3, activation="relu", padding="same")(x)
    encoded = MaxPooling1D(2, padding="same")(x)

    x = Conv1D(8, 3, activation="relu", padding="same")(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 3, activation="relu", padding="same")(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 3, activation="linear", padding="same")(x)

    model = Model(inp, decoded)
    model.compile(optimizer="adam", loss="mse")
    return model


def apply_cae_train_test(Xtrain, Xtest, epochs=CAE_EPOCHS, batch_size=CAE_BATCH_SIZE):
    if not TF_AVAILABLE:
        print("TensorFlow is not installed: CAE skipped.")
        return Xtrain, Xtest, None

    cae = build_cae(Xtrain.shape[1])

    Xtr = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], 1).astype(np.float32)
    Xte = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], 1).astype(np.float32)

    cae.fit(Xtr, Xtr, epochs=epochs, batch_size=batch_size, verbose=0)

    Xtrain_cae = cae.predict(Xtr, batch_size=batch_size, verbose=0).squeeze(-1)
    Xtest_cae = cae.predict(Xte, batch_size=batch_size, verbose=0).squeeze(-1)

    Xtrain_cae = Xtrain_cae[:, : Xtrain.shape[1]]
    Xtest_cae = Xtest_cae[:, : Xtest.shape[1]]

    return Xtrain_cae, Xtest_cae, cae


# =========================================================
# Preprocessing
# =========================================================
def preprocess_global_filters(Xraw, wn, normalize_X, use_savgol, use_loess):
    """
    These filters are applied before the train/test split, as in your earlier scripts.
    SNV and Savitzky-Golay are not fitted estimators, so this is acceptable.
    """
    X = Xraw.copy()

    if normalize_X:
        X = snv_rows(X)

    if use_savgol:
        X = apply_savgol(X)
        if normalize_X:
            X = snv_rows(X)

    if use_loess:
        X = apply_loess(X, wn)
        if normalize_X:
            X = snv_rows(X)

    return X


def normalize_y_train_test(y_train, y_test, normalize_y):
    if not normalize_y:
        return y_train.copy(), y_test.copy(), 1.0

    y_scale = np.max(np.abs(y_train))
    if y_scale == 0:
        y_scale = 1.0

    return y_train / y_scale, y_test / y_scale, y_scale


# =========================================================
# Metrics and labels
# =========================================================
def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def safe_name(text):
    text = text.replace(" ", "_")
    text = text.replace("+", "_")
    text = text.replace("-", "_")
    text = re.sub(r"[^A-Za-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def preprocessing_label(normalize_X, normalize_y, use_savgol, use_loess, use_cae):
    steps = []
    if normalize_X:
        steps.append("SNV")
    if use_savgol:
        steps.append("Savitzky-Golay")
    if use_loess:
        steps.append("LOESS")
    if use_cae:
        steps.append("CAE")
    steps.append("TPOT 0.12.2 chooses PCA/no-PCA")
    if normalize_y:
        steps.append("y-normalized")
    return " + ".join(steps)


# =========================================================
# Plots
# =========================================================
def plot_predictions(y_train, y_pred_train, y_test, y_pred_test, y_label, model_label, output_path):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_train, y_pred_train, s=24, alpha=0.35, label="Train")
    plt.scatter(y_test, y_pred_test, s=36, alpha=0.75, label="Test")

    all_true = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_pred_train, y_pred_test])
    min_val = min(np.min(all_true), np.min(all_pred))
    max_val = max(np.max(all_true), np.max(all_pred))

    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2)
    plt.xlabel(f"True {y_label}")
    plt.ylabel(f"Predicted {y_label}")
    plt.title(f"Train/Test predictions - {model_label}")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

### save the scatter in txt
def txt_predictions(y_train, y_pred_train, y_test, y_pred_test, output_path):
    with open(output_path, "w", encoding="utf8") as f:
        f.write("Training\n")
        f.write("Real \t Predicted  \n")
        for real,pred in zip(y_train,y_pred_train):
            f.write(f"{real:.8g}\t{pred:.8g}\n")

        f.write("Test\n")
        f.write("Real \t Predicted  \n")
        for real,pred in zip(y_test,y_pred_test):
            f.write(f"{real:.8g}\t{pred:.8g}\n")

def plot_violin(y_true, y_pred, y_label, title, output_path):
    true_values = np.sort(np.unique(y_true))
    violin_data = [y_pred[y_true == value] for value in true_values]

    if len(true_values) > 1:
        diffs = np.diff(true_values)
        positive_diffs = diffs[diffs > 0]
        violin_width = 0.35 * np.min(positive_diffs) if len(positive_diffs) else 0.1
    else:
        violin_width = 0.1

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    plt.figure(figsize=(7, 6))
    plt.violinplot(
        violin_data,
        positions=true_values,
        widths=violin_width,
        showmeans=True,
        showextrema=True,
    )
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2)
    plt.xticks(true_values, [f"{value:.3g}" for value in true_values], rotation=45, ha="right")
    plt.xlabel(f"True {y_label}")
    plt.ylabel(f"Predicted {y_label}")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_train_test_concentration_histogram(y_conc, train_idx, test_idx, output_path):
    concentrations = np.sort(np.unique(y_conc))
    x = np.arange(len(concentrations))
    labels = [f"{value:.3g}" for value in concentrations]

    train_counts = np.array([np.sum(y_conc[train_idx] == c) for c in concentrations])
    test_counts = np.array([np.sum(y_conc[test_idx] == c) for c in concentrations])

    width = 0.42
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, train_counts, width=width, label="Train")
    plt.bar(x + width / 2, test_counts, width=width, label="Test")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Concentration (g/mL)")
    plt.ylabel("Number of spectra")
    plt.title("Spectra per concentration in train/test split")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def find_fitted_pca(model):
    """
    Try to find a fitted sklearn PCA object inside the best TPOT pipeline.
    This only returns something if TPOT actually selected PCA.
    """
    possible_attrs = [
        "fitted_pipeline_",
        "optimized_pipeline_",
        "best_pipeline_",
        "pipeline_",
    ]

    def walk(obj):
        if isinstance(obj, PCA) and hasattr(obj, "components_"):
            return obj

        if hasattr(obj, "steps"):
            for _, step in obj.steps:
                found = walk(step)
                if found is not None:
                    return found

        if hasattr(obj, "transformer_list"):
            for _, transformer in obj.transformer_list:
                found = walk(transformer)
                if found is not None:
                    return found

        if hasattr(obj, "estimators"):
            for estimator in obj.estimators:
                candidate = estimator[1] if isinstance(estimator, tuple) else estimator
                found = walk(candidate)
                if found is not None:
                    return found

        return None

    for attr in possible_attrs:
        if hasattr(model, attr):
            found = walk(getattr(model, attr))
            if found is not None:
                return found

    return None


def plot_pca_components(wn, Xraw, y_conc, pca, loadings, output_path):
    if pca is None or loadings is None:
        return

    plt.figure(figsize=(10, 6))

    for i in range(loadings.shape[0]):
        explained = pca.explained_variance_ratio_[i] * 100
        denom = np.max(np.abs(loadings[i]))
        if denom == 0:
            denom = 1.0
        plt.plot(
            wn,
            loadings[i] / denom,
            label=f"PC{i + 1}, explained variance: {explained:.2f}%",
        )

    unique_concentrations = np.sort(np.unique(y_conc))
    for concentration in unique_concentrations:
        idx = np.where(y_conc == concentration)[0][0]
        spectrum = Xraw[idx]
        denom = np.max(np.abs(spectrum))
        if denom == 0:
            denom = 1.0
        plt.plot(
            wn,
            spectrum / denom,
            label=f"Original spectrum, concentration {concentration:.3g} g/mL",
            alpha=0.5,
        )

    plt.xlabel("wn (cm⁻¹)")
    plt.ylabel("Normalized intensity / normalized loading")
    plt.title("Original spectra and PCA loadings")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# =========================================================
# Train/test split
# =========================================================
def make_external_train_test_split(X, y_conc):
    """
    Stratified split by concentration when possible.
    If a concentration has too few samples, fallback to normal random split.
    """
    y_strata = np.array([f"{v:.12g}" for v in y_conc])
    _, counts = np.unique(y_strata, return_counts=True)

    if counts.min() >= 2:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
        )
        train_idx, test_idx = next(splitter.split(X, y_strata))
        print("External split: StratifiedShuffleSplit by concentration")
    else:
        train_idx, test_idx = train_test_split(
            np.arange(X.shape[0]),
            test_size=TEST_SIZE,
            shuffle=True,
        )
        print("External split: train_test_split, because at least one concentration has fewer than 2 samples")

    return train_idx, test_idx


# =========================================================
# Main
# =========================================================
def main():
    print("\n===== Raman preprocessing + TPOT 0.12.2 + external test =====")

    data_folder = input("Path to Data folder: ").strip().strip('"').strip("'")
    data_folder = Path(data_folder)

    augment = ask_yes_no("augment spectra?", default="n")
    if augment:
        percentage = int(input("augmentation percentage (e.g. 100 for doubling the dataset): ").strip())
    else:
        percentage = 0

    nbOfAverage = input(f"average of how many spectra: ").strip().lower()
    if nbOfAverage == "":
        nbOfAverage = 1

    target_mode = input("Target mode: 1 = linear concentration, 2 = log10 concentration [2]: ").strip()
    use_log_y = target_mode != "1"

    use_wn_min = ask_yes_no("Use lower wavenumber limit?", default="n")
    if use_wn_min:
        wn_min = float(input("Minimum wavenumber: ").strip())
    else:
        wn_min = WN_MIN_DEFAULT

    use_wn_max = ask_yes_no("Use upper wavenumber limit?", default="y")
    if use_wn_max:
        wn_max = float(input(f"Maximum wavenumber [{WN_MAX}]: ").strip() or WN_MAX)
    else:
        wn_max = np.inf

    normalize_X = ask_yes_no("Normalize spectra with SNV?", default="y")
    normalize_y = ask_yes_no("Normalize target y using train scale?", default="y")
    use_savgol = ask_yes_no("Use Savitzky-Golay?", default="y")
    use_loess = ask_yes_no("Use LOESS?", default="n")
    use_cae = ask_yes_no("Use CAE?", default="n")

    model_label = preprocessing_label(
        normalize_X=normalize_X,
        normalize_y=normalize_y,
        use_savgol=use_savgol,
        use_loess=use_loess,
        use_cae=use_cae,
    )
    print(f"Pipeline: {model_label}")

    print("\n===== Loading data =====")
    wn, Xraw, y_conc = load_raman_folder(data_folder,nbOfAverage,percentage)

    idwn = np.ones_like(wn, dtype=bool)
    if use_wn_min:
        idwn = idwn & (wn > wn_min)
    if use_wn_max:
        idwn = idwn & (wn < wn_max)

    wn = wn[idwn]
    Xraw = Xraw[:, idwn]

    print(f"Loaded spectra: {Xraw.shape[0]}")
    print(f"Spectral points: {Xraw.shape[1]}")
    print("Concentrations loaded, g/mL:", np.sort(np.unique(y_conc)))

    print("\n===== Global filtering =====")
    Xfiltered = preprocess_global_filters(
        Xraw=Xraw,
        wn=wn,
        normalize_X=normalize_X,
        use_savgol=use_savgol,
        use_loess=use_loess,
    )

    y = y_conc.copy()
    if use_log_y:
        positive = y[y > 0]
        if len(positive) == 0:
            raise ValueError("Cannot use log10 because all concentrations are <= 0")
        y[y <= 0] = np.min(positive) / 10
        y = np.log10(y)
        y_label = "log10(concentration g/mL)"
    else:
        y_label = "concentration g/mL"

    print("\n===== External train/test split =====")
    train_idx, test_idx = make_external_train_test_split(Xfiltered, y_conc)

    Xtrain = Xfiltered[train_idx]
    Xtest = Xfiltered[test_idx]
    y_train_original = y[train_idx]
    y_test_original = y[test_idx]

    if use_cae:
        print("\n===== CAE fitted on train and applied to test =====")
        Xtrain, Xtest, cae_model = apply_cae_train_test(Xtrain, Xtest)
        if normalize_X:
            Xtrain = snv_rows(Xtrain)
            Xtest = snv_rows(Xtest)
    else:
        cae_model = None

    print("\n===== TPOT will choose PCA/no-PCA internally =====")
    Xtrain_model = Xtrain
    Xtest_model = Xtest
    n_features_model = Xtrain_model.shape[1]
    print(f"TPOT input variables before its internal transformations: {n_features_model}")

    y_train_model, y_test_model, y_scale = normalize_y_train_test(
        y_train=y_train_original,
        y_test=y_test_original,
        normalize_y=normalize_y,
    )

    print("\n===== Running TPOT on TRAIN set =====")
    print(f"TPOT internal CV = {TPOT_CV}")
    print(f"TPOT scorer = {TPOT_SCORER}")
    model = make_tpot()
    model.fit(Xtrain_model, y_train_model)

    print("\n===== Predicting train and external test =====")
    y_pred_train_model = model.predict(Xtrain_model)
    y_pred_test_model = model.predict(Xtest_model)

    y_pred_train = y_pred_train_model * y_scale
    y_pred_test = y_pred_test_model * y_scale

    m_train = metrics(y_train_original, y_pred_train)
    m_test = metrics(y_test_original, y_pred_test)

    print("\n===== FINAL RESULTS =====")
    print(f"Train R2   = {m_train['R2']:.4e}")
    print(f"Train RMSE = {m_train['RMSE']:.4e}")
    print(f"Train MAE  = {m_train['MAE']:.4e}")
    print(f"Test R2    = {m_test['R2']:.4e}")
    print(f"Test RMSE  = {m_test['RMSE']:.4e}")
    print(f"Test MAE   = {m_test['MAE']:.4e}")

    pipeline_name = safe_name(model_label)
    output_dir = Path(__file__).resolve().parent / f"results_{data_folder.name}__aug-{augment}_av-{nbOfAverage}"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_lines = [
        "===== TPOT 0.12.2 TRAIN/TEST RESULTS =====",
        f"Model = {getattr(model,'fitted_pipeline_', None)}",
        f"Train R2   = {m_train['R2']:.4e}",
        f"Train RMSE = {m_train['RMSE']:.4e}",
        f"Train MAE  = {m_train['MAE']:.4e}",
        f"Test R2    = {m_test['R2']:.4e}",
        f"Test RMSE  = {m_test['RMSE']:.4e}",
        f"Test MAE   = {m_test['MAE']:.4e}",
        "",
        "===== SETTINGS =====",
        f"data_folder = {data_folder}",
        f"target_mode = {'log10' if use_log_y else 'linear'}",
        f"use_wn_min = {use_wn_min}",
        f"wn_min = {wn_min if use_wn_min else None}",
        f"use_wn_max = {use_wn_max}",
        f"wn_max = {wn_max if use_wn_max else None}",
        f"normalize_X_SNV = {normalize_X}",
        f"normalize_y = {normalize_y}",
        f"y_scale = {y_scale}",
        f"use_savgol = {use_savgol}",
        f"use_loess = {use_loess}",
        f"use_cae = {use_cae}",
        "pca = chosen internally by TPOT 0.12.2 if useful",
        f"n_input_features_before_tpot = {n_features_model}",
        f"test_size = {TEST_SIZE}",
        f"tpot_cv = {TPOT_CV}",
        f"tpot_scorer = {TPOT_SCORER}",
        f"tpot_max_time_mins = {TPOT_MAX_TIME_MINS}",
        f"tpot_max_eval_time_mins = {TPOT_MAX_EVAL_TIME_MINS}",
        f"tpot_n_jobs = {TPOT_N_JOBS}",
        f"random_state = {RANDOM_STATE}",
    ]
    (output_dir / "train_test_results.txt").write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    # Save the fitted TPOT object, a readable summary, and the exported sklearn pipeline code.
    (output_dir / "best_tpot_pipeline_summary.txt").write_text(get_best_pipeline_text(model), encoding="utf-8")

    try:
        model.export(str(output_dir / "best_tpot_pipeline.py"))
    except Exception as exc:
        (output_dir / "best_tpot_pipeline_export_error.txt").write_text(
            f"Could not export TPOT pipeline: {exc}\n",
            encoding="utf-8",
        )
    final_pipeline = model.fitted_pipeline_
    joblib.dump(final_pipeline, output_dir / "best_tpot_final_pipeline.joblib")
    selected_pca = find_fitted_pca(model)
    if selected_pca is not None:
        joblib.dump(selected_pca, output_dir / "selected_pca_inside_tpot.joblib")
    if cae_model is not None:
        cae_model.save(output_dir / "cae_model.keras")

    joblib.dump(
        {
            "model": final_pipeline,
            "selected_pca_inside_tpot": selected_pca,
            "cae_model_saved_separately": cae_model is not None,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "wn": wn,
            "y_train_true": y_train_original,
            "y_train_pred": y_pred_train,
            "y_test_true": y_test_original,
            "y_test_pred": y_pred_test,
            "y_concentration_g_ml": y_conc,
            "metrics_train": m_train,
            "metrics_test": m_test,
            "settings": {
                "data_folder": str(data_folder),
                "target_mode": "log10" if use_log_y else "linear",
                "normalize_X_SNV": normalize_X,
                "normalize_y": normalize_y,
                "y_scale": y_scale,
                "use_savgol": use_savgol,
                "use_loess": use_loess,
                "use_cae": use_cae,
                "pca": "chosen internally by TPOT 0.12.2 if useful",
                "n_input_features_before_tpot": n_features_model,
                "test_size": TEST_SIZE,
                "tpot_cv": TPOT_CV,
                "tpot_scorer": TPOT_SCORER,
                "random_state": RANDOM_STATE,
            },
        },
        output_dir / "complete_results.joblib",
    )

    print("\n===== BEST PIPELINE FOUND BY TPOT =====")

    for attr in [
        "fitted_pipeline_",
        "optimized_pipeline_",
        "best_pipeline_",
        "pipeline_",
        ]:
        if hasattr(model, attr):
            print(getattr(model, attr))
            break

    plot_predictions(
        y_train=y_train_original,
        y_pred_train=y_pred_train,
        y_test=y_test_original,
        y_pred_test=y_pred_test,
        y_label=y_label,
        model_label=model_label,
        output_path=output_dir / "train_test_predictions.png",
    )

    txt_predictions(y_train_original, y_pred_train, y_test_original, y_pred_test, output_path=output_dir / "train_test_predictions.txt")

    plot_violin(
        y_true=y_test_original,
        y_pred=y_pred_test,
        y_label=y_label,
        title=f"External test prediction violins - {model_label}",
        output_path=output_dir / "test_violin.png",
    )

    plot_train_test_concentration_histogram(
        y_conc=y_conc,
        train_idx=train_idx,
        test_idx=test_idx,
        output_path=output_dir / "train_test_concentration_histogram.png",
    )

    if selected_pca is not None:
        plot_pca_components(
            wn=wn,
            Xraw=Xraw,
            y_conc=y_conc,
            pca=selected_pca,
            loadings=selected_pca.components_,
            output_path=output_dir / "PCA_components.png",
        )
    else:
        print("TPOT did not select PCA, so PCA_components.png was not created.")


    print(f"\nSaved results in: {output_dir}")
    print("Saved trained final TPOT pipeline: best_tpot_final_pipeline.joblib")
    print("Saved pipeline summary: best_tpot_pipeline_summary.txt")
    print("Saved exported sklearn pipeline if available: best_tpot_pipeline.py")

    toc = perf_counter()

    print(f"tiempo ejecución: {toc-tic:.3f} s")


if __name__ == "__main__":
    main()
