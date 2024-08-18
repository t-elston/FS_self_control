import numpy as np 
import pandas as pd 
import h5py
import os
import glob



def find_h5_files(directory):
    """
    Search for HDF5 files (.h5 extension) in the specified directory.

    Parameters:
    - directory (str): Path to the directory to search for HDF5 files.

    Returns:
    - List[str]: A list of filenames (including paths) of HDF5 files found in the directory.
    """
    h5_files = []
    search_pattern = os.path.join(directory, '*.h5')  # Pattern to search for .h5 files

    for file_path in glob.glob(search_pattern):
        if os.path.isfile(file_path):
            h5_files.append(file_path)

    return h5_files


def pull_from_h5(file_path, data_to_extract):
    try:
        with h5py.File(file_path, 'r') as file:
            # Check if the data_to_extract exists in the HDF5 file
            if data_to_extract in file:
                data = file[data_to_extract][...]  # Extract the data
                return data
            else:
                print(f"'{data_to_extract}' not found in the file.")
                return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def list_hdf5_data(file_path):
    try:
        with h5py.File(file_path, 'r') as file:
            print(f"Datasets in '{file_path}':")
            for dataset in file:
                print(dataset)
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
def save_data_to_h5(directory, filename, datasets, dataset_names):
    """
    Saves multiple datasets to an HDF5 file, overwriting existing datasets.

    Args:
        directory (str): The directory where the file should be saved.
        filename (str): The name of the HDF5 file.
        datasets (list): A list of datasets to save.
        dataset_names (list): A list of names to assign to the datasets.
    """

    full_path = os.path.join(directory, filename)

    with h5py.File(full_path, 'a') as f:  # Open in append mode
        for dataset, dataset_name in zip(datasets, dataset_names):
            try:
                del f[dataset_name]  # Attempt to delete existing dataset
            except KeyError:
                pass  # Ignore if dataset doesn't exist
            f.create_dataset(dataset_name, data=dataset)  # Recreate dataset