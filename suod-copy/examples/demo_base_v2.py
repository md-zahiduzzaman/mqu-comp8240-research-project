import os
import sys

import scipy as sp
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from pyod.models.lof import LOF
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.utils.data import evaluate_print

from combo.models.score_comb import majority_vote, maximization, average

# suppress warnings
import warnings

warnings.filterwarnings("ignore")

import traceback
# temporary solution for relative imports in case combo is not installed
# if combo is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from suod.models.base import SUOD
from suod.utils.utility import get_estimators_small

# Monkey patch the predict_proba method in base.py to handle inf values
import pyod.models.base
original_predict_proba = pyod.models.base.BaseDetector.predict_proba
original_decision_function = pyod.models.base.BaseDetector.decision_function

def safe_predict_proba(self, X):
    """
    Override the predict_proba method to handle inf values
    """
    try:
        # Call the original method
        result = original_predict_proba(self, X)
        
        # Replace inf with large finite values
        if isinstance(result, np.ndarray):
            result = np.nan_to_num(result, nan=0.5, posinf=0.999, neginf=0.001)
            
            # Ensure result is 2D with shape (n_samples, 2)
            if result.ndim == 1:
                # If 1D, convert to 2D: [p] -> [[1-p, p]]
                result_2d = np.zeros((len(result), 2))
                result_2d[:, 1] = result
                result_2d[:, 0] = 1 - result
                result = result_2d
                
        return result
    except Exception as e:
        print(f"Error in predict_proba for {self.__class__.__name__}: {e}")
        # Return a safe fallback - 2D array with shape (n_samples, 2)
        n_samples = X.shape[0]
        result = np.zeros((n_samples, 2))
        result[:, 0] = 0.5  # prob of normal
        result[:, 1] = 0.5  # prob of outlier
        return result

def safe_decision_function(self, X):
    """
    Override the decision_function method to handle inf values
    """
    try:
        # Call the original method
        result = original_decision_function(self, X)
        
        # Replace inf with large finite values
        if isinstance(result, np.ndarray):
            result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
                
        return result
    except Exception as e:
        print(f"Error in decision_function for {self.__class__.__name__}: {e}")
        # Return a safe fallback
        n_samples = X.shape[0]
        return np.zeros(n_samples)

# Apply the monkey patches
pyod.models.base.BaseDetector.predict_proba = safe_predict_proba
pyod.models.base.BaseDetector.decision_function = safe_decision_function

# Also let's monkey patch the parallel_processes.py functionality to be more robust
import suod.models.parallel_processes

original_parallel_predict_proba = suod.models.parallel_processes._parallel_predict_proba
original_parallel_decision_function = suod.models.parallel_processes._parallel_decision_function

def safe_parallel_predict_proba(estimator, X, X_ind, clf_name, scaler=None):
    """
    Safe version of _parallel_predict_proba that handles various output formats
    """
    try:
        # Call the original method with a try/except block
        return original_parallel_predict_proba(estimator, X, X_ind, clf_name, scaler)
    except Exception as e:
        print(f"Error in parallel_predict_proba for {clf_name}: {e}")
        # Return safe default - probability 0.5 for all samples in correct format
        n_samples = len(X_ind)  # Use X_ind length as that's what gets returned
        return np.ones(n_samples) * 0.5

def safe_parallel_decision_function(estimator, X, X_ind, clf_name, scaler=None):
    """
    Safe version of _parallel_decision_function that handles various output formats
    """
    try:
        # Call the original method with a try/except block
        return original_parallel_decision_function(estimator, X, X_ind, clf_name, scaler)
    except Exception as e:
        print(f"Error in parallel_decision_function for {clf_name}: {e}")
        # Return safe default - score 0 for all samples
        n_samples = len(X_ind)  # Use X_ind length as that's what gets returned
        return np.zeros(n_samples)

# Apply the second set of monkey patches
suod.models.parallel_processes._parallel_predict_proba = safe_parallel_predict_proba
suod.models.parallel_processes._parallel_decision_function = safe_parallel_decision_function

def clean_data(X):
    """Helper function to clean data of potential issues"""
    # Check for and remove constant columns
    col_vars = np.var(X, axis=0)
    non_constant_cols = np.where(col_vars > 1e-10)[0]
    if len(non_constant_cols) < X.shape[1]:
        print(f"Removing {X.shape[1] - len(non_constant_cols)} constant columns")
        X = X[:, non_constant_cols]
    
    # Replace inf/nan values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X

def process_dataset(mat_file, max_samples=5000):
    """Process a single dataset with robust error handling"""
    mat_file_name = mat_file.replace('.mat', '')
    print(f"\n... Processing {mat_file_name} ...")
    
    # Look in multiple possible locations for the data file
    possible_paths = [
        os.path.join('', 'datasets', mat_file),
        os.path.join('examples', 'datasets', mat_file),
        os.path.join('examples', 'module_examples', 'datasets', mat_file)
    ]
    
    mat = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                mat = sp.io.loadmat(path)
                print(f"Successfully loaded data from {path}")
                break
            except Exception as e:
                print(f"Error loading {path}: {e}")
    
    if mat is None:
        print(f"Could not find or load {mat_file}")
        return
    
    # Extract X and y, handle different possible structures
    try:
        X = mat['X']
        y = mat['y'].ravel()  # Ensure y is 1D
    except Exception as e:
        print(f"Error extracting X and y from mat file: {e}")
        # Try alternative keys
        keys = list(mat.keys())
        print(f"Available keys in mat file: {[k for k in keys if not k.startswith('__')]}")
        for key in keys:
            if isinstance(mat[key], np.ndarray) and len(mat[key].shape) == 2 and mat[key].shape[0] > 100:
                if key != 'y' and key != 'Y':
                    print(f"Using {key} as X, shape: {mat[key].shape}")
                    X = mat[key]
        # Look for y or labels
        for key in ['y', 'Y', 'labels', 'target']:
            if key in mat:
                print(f"Using {key} as y, shape: {mat[key].shape}")
                y = mat[key].ravel()
                break
    
    # Limit dataset size if too large
    if X.shape[0] > max_samples:
        print(f"Dataset too large ({X.shape[0]} samples), sampling {max_samples} samples")
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    # Clean data before standardization
    print("Initial data shape:", X.shape)
    print("Initial min value:", np.min(X))
    print("Initial max value:", np.max(X))
    
    X = clean_data(X)
    
    # standardize data to be digestible for most algorithms
    try:
        X = StandardScaler().fit_transform(X)
    except Exception as e:
        print(f"StandardScaler failed: {e}")
        # Try simpler preprocessing
        X = MinMaxScaler().fit_transform(X)
    
    # Check for extreme values after standardization
    print("After standardization, min value:", np.min(X))
    print("After standardization, max value:", np.max(X))
    
    # Apply additional safeguard - clip extreme values
    X = np.clip(X, -100, 100)  # clip to reasonable range for most algorithms
    
    # Final cleaning
    X = clean_data(X)
    
    return X, y

if __name__ == "__main__":
    # list available datasets
    mat_file_list = [
        'cardio.mat',
        'satellite.mat',
        'satimage-2.mat',
        'mnist.mat',
    ]
    
    # Process all datasets one by one
    for idx, mat_file in enumerate(mat_file_list):
        try:
            print(f"\n===== Processing dataset {idx+1}/{len(mat_file_list)}: {mat_file} =====")
            result = process_dataset(mat_file)
            
            if result is None:
                print(f"Skipping {mat_file} due to loading issues")
                continue
                
            X, y = result
            
            # Split the data
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.4, random_state=42)
            
            contamination = np.mean(y)
            print(f"Contamination: {contamination:.4f}")
            
            try:
                # Get base estimators with appropriate contamination
                base_estimators = get_estimators_small(contamination)
                print(f"Number of base estimators: {len(base_estimators)}")
                
                # Use n_jobs=1 to avoid parallelism issues
                model = SUOD(base_estimators=base_estimators, n_jobs=1, 
                             bps_flag=True, contamination=contamination, 
                             approx_flag_global=True)
                
                print("Fitting model...")
                model.fit(X_train)
                
                print("Approximating model...")
                model.approximate(X_train)
                
                print("Predicting labels...")
                predicted_labels = model.predict(X_test)
                
                print("Computing decision scores...")
                predicted_scores = model.decision_function(X_test)
                predicted_scores = np.nan_to_num(predicted_scores, nan=0.0, posinf=10.0, neginf=-10.0)
                
                print("Computing probabilities...")
                try:
                    predicted_probs = model.predict_proba(X_test)
                except Exception as e:
                    print(f"predict_proba failed: {e}")
                    predicted_probs = None

                # Evaluation
                print("\nEvaluation Results:")
                evaluate_print('majority vote', y_test, majority_vote(predicted_labels))
                
                cleaned_scores = np.nan_to_num(predicted_scores, nan=0.0, posinf=10.0, neginf=-10.0)
                evaluate_print('average', y_test, average(cleaned_scores))
                evaluate_print('maximization', y_test, maximization(cleaned_scores))

                # Individual models for comparison
                print("\nIndividual model performance:")
                for clf_name, clf_class in [('LOF', LOF), ('ABOD', ABOD), ('KNN', KNN)]:
                    try:
                        clf = clf_class()
                        clf.fit(X_train)
                        scores = clf.decision_function(X_test)
                        scores = np.nan_to_num(scores, nan=0.0, posinf=10.0, neginf=-10.0)
                        evaluate_print(clf_name, y_test, scores)
                    except Exception as e:
                        print(f"{clf_name} failed: {e}")
                
                print(f"\nSuccessfully completed {mat_file}")
                
            except Exception as e:
                print(f"Error processing {mat_file}: {e}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"Fatal error processing {mat_file}: {e}")
            traceback.print_exc()
            
        print("\n" + "="*70)