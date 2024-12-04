# ~~~~~~~~~~~~~~ Libraries
import sys, os
import mne # Python package for processing and analyzing electrophysiological data
import numpy as np
from glob import glob # look for all the pathnames matching a specified pattern according to the rules
import matplotlib.pyplot as plt
from mne.preprocessing import ICA # ICA (Independent Component Analysis) algorithm, which is for artifact removal
import json
import re

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.stats import permutation_test
import matplotlib.pyplot as plt
from itertools import combinations

import pandas as pd
import seaborn as sns
import pickle



# ~~~~~~~~~~~~~~ Parameters
modality = 'visual' # 'visual' or 'audio'

# RDM parameters
window_size = 5 # 1 sample = 2ms, 5 samples = 10 ms

# Print out each parameter
print(f"{modality} data is processed")
print("RDM parameters:")
print(f"  Window size: {window_size} samples ({window_size * 2} ms)")
# ~~~~~~~~~~~~~~ Parameters ~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~ Set the working directory
path = f"/u/kazma/MINT/data/interim/{modality}"
sub_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
sub_folders_sorted = sorted(sub_folders, key=lambda x: int(re.search(r'\d+', x).group())) # Sort the folders based on the numeric part after "sub-"
# ~~~~~~~~~~~~~~ Set the working directory ~~~~~~~~~~~~~~



# SUB_LOOP
for subject in sub_folders_sorted:

    # subject folder 
    sub_filename = os.path.join(path, subject, 'RDM_epochs-epo.fif')

    # Load epochs
    epochs = mne.read_epochs(sub_filename, preload=True)

    # Get the information about the data
    conditions = list(epochs.event_id.keys())
    n_conditions = len(conditions)
    # n_trials = len(epochs)
    n_channels = epochs.get_data().shape[1]


    n_samples = epochs.get_data().shape[2]
    min_time = epochs.times[0]*1000   # First time point in milli seconds
    max_time = epochs.times[-1]*1000    # Last time point in milli seconds

    # store rdm for each time point
    all_rdm = []

    # WINDOW_LOOP
    for start in range(0, n_samples, window_size): # Loop over the epoch in steps of `window_size` to extract each 5-sample window
        # Check if there are enough samples left for a full window
        if start + window_size <= n_samples:

            df_dict = {}
            for condition in conditions:
                x = epochs[condition]
                x = x.get_data()
                x = x[:, :, start:start + window_size]
                x = x.reshape(x.shape[0], -1)
                df_dict[condition] = x


            # Define the order for conditions
            condition_order = {'singledot': 0, 'totaldot': 1, 'circum': 2}
            # Sort dictionary by first number and then by condition order
            sorted_condition_dict = dict(
                sorted(
                    df_dict.items(),
                    key=lambda item: (
                        int(item[0].split('_')[0]),
                        condition_order.get(item[0].split('_')[2], 99),
                        int(item[0].split('_')[-1].split('.')[0])  # Sort by condition order, 99 as default for unmatched
                    )
                )
            )

            pairwise_dissimilar_dict = {}

            # CONDITION_LOOP
            for i in range(n_conditions):
                for j in range(i + 1, n_conditions):
                    
                    cond1 = list(sorted_condition_dict.keys())[i]
                    cond2 = list(sorted_condition_dict.keys())[j]
                    data_cond1 = sorted_condition_dict[cond1]
                    data_cond2 = sorted_condition_dict[cond2]
                    # Correlation
                    corr_matrix = np.corrcoef(data_cond1, data_cond2)
                    pairwise_dissimilar_dict[(cond1, cond2)] = 1 - corr_matrix[0, 1] # 0 = most similar and 2 = most dissimilar
                    # print(f"Correlation for {cond1} vs {cond2} in time points {start + window_size}: {np.abs(corr_matrix[0, 1]):.2f}")
        
        # Average accuracy for this time point across CV folds
        all_rdm.append(pairwise_dissimilar_dict)
        print(f"Time point {start + window_size} is done")




    # ~~~~~~~~~~~~~~~~ Generate dissimilarity matrix

    all_dissimilarity_matrices = [] # List to store the resulting dissimilarity matrices

    # ~~~~~~~~~~~~~~~~  CONDITION_LOOP
    for idx, x in enumerate(all_rdm):
        # Extract all unique conditions
        conditions = sorted(set(key[0] for key in x.keys()).union(set(key[1] for key in x.keys())))
        # Define the order for conditions
        condition_order = {'singledot': 0, 'totaldot': 1, 'circum': 2}
        # Sort dictionary by first number and then by condition order
        sorted_conditions = sorted(
                conditions,
                key=lambda item: (
                    int(item.split('_')[0]),
                    condition_order.get(item.split('_')[2], 99),
                    int(item.split('_')[-1].split('.')[0])  # Sort by condition order, 99 as default for unmatched
                )
            )
        
        n_conditions = len(sorted_conditions)

        # Create a dictionary to map conditions to matrix indices
        condition_idx = {condition: idx for idx, condition in enumerate(sorted_conditions)}

        # Initialize a square matrix with NaN values (for easier filling)
        dissimilarity_matrix = np.full((n_conditions, n_conditions), np.nan)

        # Fill the matrix with accuracies
        for (cond1, cond2), dissimilarity in x.items(): # e.g., 'cond1' = 'numerosity 1' and 'cond2' = 'numerosity 2'
            i, j = condition_idx[cond1], condition_idx[cond2]
            dissimilarity_matrix[i, j] = dissimilarity
            dissimilarity_matrix[j, i] = dissimilarity  # Ensuring symmetry

        # Convert to DataFrame for readability
        dissimilarity_df = pd.DataFrame(dissimilarity_matrix, index=sorted_conditions, columns=sorted_conditions)
        
        # Store the DataFrame in the list
        all_dissimilarity_matrices.append(dissimilarity_df)

        # Optionally print each matrix to verify (can remove if not needed)
        print(f"Decoding dissimilarity Matrix for entry {idx + 1}:")
        print(dissimilarity_df)
        print("\n")
    # ~~~~~~~~~~~~~~~~  CONDITION_LOOP ~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~ Save the list of accuracy matrices
    save_folder = f"/u/kazma/MINT/data/processed/{modality}/{subject}"
    save_path = os.path.join(save_folder, "rdm_matrices.pkl") #  a pickle file
    with open(save_path, "wb") as f:
        pickle.dump(all_dissimilarity_matrices, f)
    print(f"{subject}: matrices saved in 'rdm_matrices.pkl'")
    # ~~~~~~~~~~~~~~~~ Save the list of accuracy matrices ~~~~~~~~~~~~~~~~