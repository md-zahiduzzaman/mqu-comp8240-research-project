import scipy.io
import matplotlib.pyplot as plt
import pandas as pd

import os

def display_mat_data(mat_file):
    # Debug: Check current directory and datasets path
    print(f"Current working directory: {os.getcwd()}")

    # Different possible paths
    possible_paths = [
        os.path.join('datasets', mat_file),  
        os.path.join('examples', 'datasets', mat_file), 
        os.path.join('SUOD', 'examples', 'datasets', mat_file), 
    ]
    
    # Check each possible path
    mat_path = None
    for path in possible_paths:
        print(f"Checking: {os.path.abspath(path)}")
        if os.path.exists(path):
            mat_path = path
            print(f"✓ Found file at: {os.path.abspath(path)}")
            break
    
    # If file not found, raise an error
    if mat_path is None:
        print("\nFile not found in any expected location!")
        print("\nPlease check:")
        print("1. Does the datasets folder exist?")
        print(f"2. Does {mat_file} exist in the datasets folder?")
        print("3. What is the actual location of your datasets folder?")
        raise FileNotFoundError(f"Could not find {mat_file} in any expected location")
    
    # Loading .mat file
    mat = scipy.io.loadmat(mat_path)

    # Printing available keys
    print("Keys in MAT file:", mat.keys())
    
    # Choosing key storing data
    data_key = None
    for key in mat.keys():
        if not key.startswith("__"):
            data_key = key
            break
    
    data = mat[data_key]
    print(f"Data key: {data_key}")
    print(f"Data shape: {data.shape}")
    
    # If data is 2D with samples x features (tabular)
    if data.ndim == 2:
        print("Detected tabular data, showing first 5 rows:\n")
        df = pd.DataFrame(data)
        print(df.head())
    # If data is images or multi-dimensional 
    elif data.ndim >= 3:
        print("Detected image or multi-dimensional data, displaying first sample:")
        first_sample = data[0]
        # If 2D or 3D image data
        if first_sample.ndim == 2:
            plt.imshow(first_sample, cmap='gray')
            plt.title("First sample image (grayscale)")
            plt.show()
        elif first_sample.ndim == 3:
            # Assume channels last
            plt.imshow(first_sample)
            plt.title("First sample image (color)")
            plt.show()
        else:
            print("Complex data shape, visualisation not supported directly.")
    else:
        print("Data format not recognized for visualization.")

# Pass the .mat file name here
mat_file = "satimage-2.mat"
display_mat_data(mat_file)