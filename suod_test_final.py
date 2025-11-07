# Final version of SUOD test for new datasets

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit

from pyod.models.lof import LOF
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.utils.data import evaluate_print

# SUOD import (via PyOD preferred, fallback to standalone)
try:
    from pyod.models.suod import SUOD
except Exception:
    from suod import SUOD  # type: ignore

# Optional: lightweight text vectorizer for datasets with a text column
try:
    from sklearn.feature_extraction.text import HashingVectorizer
    _HAS_HASHING = True
except Exception:
    _HAS_HASHING = False

# Configuring paths to datasets and setting parameters
DATASETS = [
    {"path": "new-data/existing/credit-card-fraud/creditcard.csv", "label_col": "Class"},  # credit card (label is 'Class')
    {"path": "new-data/generated/tabular-data/noisy-iris.csv", "label_col": "label"},
    {"path": "new-data/generated/concatenated-text/text-plus-code.csv", "label_col": "label", "text_col": "text"}
]

TEST_SIZE = 0.30
RANDOM_STATE = 42
N_JOBS = 1  # 1 for robustness
MAX_FEATURES_HASH = 2**12  # for HashingVectorizer if used

LABEL_CANDIDATES = [
    "label", "y", "target", "class", "Class", "fraud", "is_fraud",
    "anomaly", "is_anomaly", "outlier", "is_outlier"
]

# Utility functions
def find_label_col(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No label column found (looked for: {LABEL_CANDIDATES})")


def coerce_binary_labels(series: pd.Series) -> np.ndarray:
    """Coerce a label series to {0,1} ints."""
    s = series
    if s.dtype.kind in "biu":  # bool/int/uint
        return (s != 0).astype(int).to_numpy()
    s_lower = s.astype(str).str.lower()
    positive = {"1", "true", "yes", "fraud", "anomaly", "outlier"}
    return s_lower.isin(positive).astype(int).to_numpy()


def load_csv_features(path: str, label_override: str | None = None, text_col: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    label_col = find_label_col(df, preferred=label_override)
    y = coerce_binary_labels(df[label_col])

    # numeric features
    X_num = df.select_dtypes(include=[np.number]).drop(columns=[label_col], errors="ignore")
    if X_num.shape[1] > 0:
        X = X_num.to_numpy()
        return X, y

    # If no numeric features, try text hashing if configured
    if text_col and text_col in df.columns:
        if not _HAS_HASHING:
            raise ImportError("scikit-learn HashingVectorizer not available; cannot vectorize text.")
        texts = df[text_col].astype(str).fillna("")
        hv = HashingVectorizer(
            n_features=MAX_FEATURES_HASH,
            alternate_sign=False,
            norm=None,  # we'll scale later
        )
        X = hv.transform(texts).toarray()  # convert to dense for PyOD detectors
        return X, y

    raise ValueError(
        f"No numeric features found in {path} after dropping '{label_col}'. "
        f"Provide a text column via 'text_col' in DATASETS if you want hashing."
    )

# Previous version without fallback
# def fit_preproc(X_train: np.ndarray) -> tuple[np.ndarray, dict]:
#     """Learn the column mask and scaler on TRAIN, then transform and clip."""
#     X = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
#     keep = X.var(axis=0) > 1e-12  # remove constant columns (learned on TRAIN)
#     Xk = X[:, keep]

#     # Fit scaler on TRAIN
#     try:
#         scaler = StandardScaler().fit(Xk)
#     except Exception:
#         scaler = MinMaxScaler().fit(Xk)

#     Xk = scaler.transform(Xk)
#     Xk = np.clip(Xk, -100.0, 100.0)
#     return Xk, {"keep": keep, "scaler": scaler}

def fit_preproc(X_train: np.ndarray) -> tuple[np.ndarray, dict]:
    """Learn the column mask and scaler on TRAIN, then transform and clip."""
    X = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Primary mask: remove strictly constant columns
    var = X.var(axis=0)
    keep = var > 1e-12

    # Fallback: if everything looks constant, keep columns that are not all-zero
    if keep.sum() == 0:
        non_zero = (X != 0).any(axis=0)
        if non_zero.sum() == 0:
            raise ValueError(
                "All features are zero in the TRAIN split. "
                "Your text column may be empty or identical for all rows."
            )
        keep = non_zero

    Xk = X[:, keep]

    # Fit scaler on TRAIN (fallback to MinMax if StandardScaler fails)
    try:
        scaler = StandardScaler().fit(Xk)
    except Exception:
        scaler = MinMaxScaler().fit(Xk)

    Xk = scaler.transform(Xk)
    Xk = np.clip(Xk, -100.0, 100.0)
    return Xk, {"keep": keep, "scaler": scaler}


def transform_preproc(X_test: np.ndarray, pre: dict) -> np.ndarray:
    """Apply TRAIN-learned mask and scaler to TEST, then clip."""
    X = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    Xk = X[:, pre["keep"]]
    Xk = pre["scaler"].transform(Xk)
    Xk = np.clip(Xk, -100.0, 100.0)
    return Xk

# Previous version without safe k selection
# def make_base_detectors(contamination: float):
#     return [
#         LOF(n_neighbors=20, contamination=contamination, n_jobs=N_JOBS),
#         ABOD(n_neighbors=10, contamination=contamination),
#         KNN(n_neighbors=10, method="largest", contamination=contamination, n_jobs=N_JOBS),
#     ]

# Safe wrapper to sanitize decision_function scores
class SafeDetector:
    def __init__(self, base): self.base = base
    def __getattr__(self, name): return getattr(self.base, name)
    def fit(self, X, y=None): return self.base.fit(X, y)
    def decision_function(self, X):
        import numpy as np
        s = self.base.decision_function(X)
        return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)


def make_base_detectors(contamination: float, n_train: int):
    # Choosing safe k between 5 and min(50, n_train-1)
    k = max(5, min(50, n_train - 1))
    k_abod = max(5, min(30, n_train - 1))
    return [
        LOF(n_neighbors=k, contamination=contamination, n_jobs=N_JOBS),
        ABOD(n_neighbors=k_abod, contamination=contamination),
        KNN(n_neighbors=k, method="largest", contamination=contamination, n_jobs=N_JOBS),
    ]

# Previous version without score sanitization
# def train_eval_detector(clf, name: str, X_train: np.ndarray, y_train: np.ndarray,
#                         X_test: np.ndarray, y_test: np.ndarray):
#     t0 = time.perf_counter()
#     clf.fit(X_train)
#     t1 = time.perf_counter()
#     scores = clf.decision_function(X_test)
#     t2 = time.perf_counter()
#     evaluate_print(name, y_test, scores)  # prints ROC and PR metrics
#     print(f"[{name}] fit={t1 - t0:.3f}s  predict={t2 - t1:.3f}s\n")

def _sanitize_scores(scores: np.ndarray) -> np.ndarray:
    s = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    # If all scores are identical, we add tiny jitter to avoid metric edge-cases
    if np.ptp(s) == 0:
        rng = np.random.default_rng(0)
        s = s + 1e-9 * rng.standard_normal(size=s.shape[0])
    return s

# Updated version with score sanitization and fallback
def train_eval_detector(clf, name: str, X_train, y_train, X_test, y_test):
    t0 = time.perf_counter()
    clf.fit(X_train)
    t1 = time.perf_counter()
    # Prefer decision_function; fall back to predict_proba if available
    try:
        scores = clf.decision_function(X_test)
    except Exception:
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X_test)[:, 1]
        else:
            raise
    scores = _sanitize_scores(scores)
    t2 = time.perf_counter()
    evaluate_print(name, y_test, scores)
    print(f"[{name}] fit={t1 - t0:.3f}s  predict={t2 - t1:.3f}s\n")

def cap_size_stratified(X, y, max_n=5000, random_state=42):
    """Downsample to max_n rows with class proportions preserved."""
    n = X.shape[0]
    if n <= max_n:
        return X, y, False  # no cap applied

    try:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=max_n, random_state=random_state)
        idx_keep, _ = next(sss.split(X, y))
        return X[idx_keep], y[idx_keep], True
    except ValueError:
        # Fallback: sample per class manually (handles edge cases)
        rng = np.random.default_rng(random_state)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        # target counts proportional to original mix
        n_pos = int(round(len(pos) * max_n / n))
        n_pos = max(1, min(n_pos, len(pos)-1)) if len(pos) >= 2 else len(pos)
        n_neg = max_n - n_pos
        n_neg = max(1, min(n_neg, len(neg)))
        keep = np.concatenate([
            rng.choice(pos, size=n_pos, replace=False) if len(pos) >= n_pos and n_pos > 0 else pos,
            rng.choice(neg, size=n_neg, replace=False) if len(neg) >= n_neg and n_neg > 0 else neg
        ])
        rng.shuffle(keep)
        return X[keep], y[keep], True

if __name__ == "__main__":
    for cfg in DATASETS:
        csv_path = cfg["path"]
        label_col = cfg.get("label_col")
        text_col = cfg.get("text_col")  # optional for text+code dataset

        print("\n" + "=" * 92)
        print(f"Dataset: {csv_path}  |  label: {label_col or 'auto'}  |  text_col: {text_col or '-'}")

        # Load
        X, y = load_csv_features(csv_path, label_override=label_col, text_col=text_col)
        if X.size == 0:
            raise ValueError(f"Empty feature matrix from {csv_path}.")
        
        # If datasset too large, cap size to 5000 with contamination preserved
        X, y, capped = cap_size_stratified(X, y, max_n=5000, random_state=RANDOM_STATE)
        if capped:
            print(f"  -> Capped dataset to 5000 rows (preserved class ratio).")

        # Split (stratified to preserve anomaly rate in test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )

        # Preprocess with TRAIN-learned mask & scaler (apply to TEST)
        X_train_pp, pre = fit_preproc(X_train)
        X_test_pp = transform_preproc(X_test, pre)
        assert X_train_pp.shape[1] == X_test_pp.shape[1], "Train/Test feature dims differ after preprocessing!"

        # contamination = float((y_train == 1).mean())
        # print(f"Train: {X_train_pp.shape} | Test: {X_test_pp.shape} | Contamination(train)={contamination:.4f}")

        # # Baselines
        # base_detectors = make_base_detectors(contamination)
        # base_names = ["LOF", "ABOD", "KNN"]
        # for det, name in zip(base_detectors, base_names):
        #     train_eval_detector(det, name, X_train_pp, y_train, X_test_pp, y_test)

        # # SUOD ensemble over the same three detectors
        # suod = SUOD(
        #     base_estimators=make_base_detectors(contamination),
        #     n_jobs=N_JOBS,
        #     rp_flag_global=True,         # random projections (data-level accel)
        #     approx_flag_global=True,     # pseudo-supervised approximations (model-level accel)
        #     bps_flag=True,               # balanced parallel scheduling (execution-level accel)
        #     contamination=contamination,
        # )
        # train_eval_detector(suod, "SUOD Ensemble (LOF+ABOD+KNN)", X_train_pp, y_train, X_test_pp, y_test)
        
        # Updated test with safe k and score sanitization
        contamination = float((y_train == 1).mean())
        print(f"Train: {X_train_pp.shape} | Test: {X_test_pp.shape} | Contamination(train)={contamination:.4f}")
        base_detectors= [SafeDetector(d) for d in make_base_detectors(contamination, n_train=X_train_pp.shape[0])]
        base_names = ["LOF", "ABOD", "KNN"]
        for det, name in zip(base_detectors, base_names):
            train_eval_detector(det, name, X_train_pp, y_train, X_test_pp, y_test)

        suod = SUOD(
            base_estimators=make_base_detectors(contamination, n_train=X_train_pp.shape[0]),
            n_jobs=N_JOBS,
            rp_flag_global=True,
            approx_flag_global=True,
            bps_flag=True,
            contamination=contamination,
        )
        train_eval_detector(suod, "SUOD(LOF+ABOD+KNN)", X_train_pp, y_train, X_test_pp, y_test)

        print("-" * 92)
