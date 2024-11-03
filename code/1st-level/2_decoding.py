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
# ~~~~~~~~~~~~~~ Libraries ~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~ Set the working directory
path = "/u/kazma/MINT/data/interim/visual"
sub_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
sub_folders_sorted = sorted(sub_folders, key=lambda x: int(re.search(r'\d+', x).group())) # Sort the folders based on the numeric part after "sub-"
# ~~~~~~~~~~~~~~ Set the working directory ~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~ ML Parameters
window_size = 5 # 1 sample = 2ms, 5 samples = 10 ms
n_splits = 10 # number of folds
clf = SVC(kernel='linear') # support vector machine (SVM) classifier with a linear kernel.

# Print out each parameter
print("Machine learning parameters:")
print(f"  Window size: {window_size} samples ({window_size * 2} ms)")
print(f"  Number of CV splits (folds): {n_splits}")
print(f"  Classifier: {clf.__class__.__name__} with kernel = '{clf.kernel}'")
# ~~~~~~~~~~~~~~ ML Parameters ~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~ SUB_LOOP
for subject in sub_folders_sorted:
    # each subject's file name
    sub_filename = os.path.join(path, subject, 'epochs-epo.fif') 
    # Load epochs
    epochs = mne.read_epochs(sub_filename, preload=True)

    # Get the info about the data
    conditions = list(epochs.event_id.keys()) # list of conditions
    n_conditions = len(conditions) # number of conditions
    n_trials = len(epochs) # number of trials
    n_samples = epochs.get_data().shape[2]
    min_time = epochs.times[0]*1000   # First time point in milli seconds
    max_time = epochs.times[-1]*1000    # Last time point in milli seconds

    # list for avg accuracy for each time window
    all_decoding_accuracy = [] 

    # ~~~~~~~~~~~~~~ WINDOW_LOOP
    for start in range(0, n_samples, window_size): # Loop over the epoch in steps of `window_size` to extract each 5-sample window
        if start + window_size <= n_samples: # Check if there are enough samples left for a full window

            # Initialize the decoding dictionary with each condition containing a list of flattened sample windows
            decoding_dict = {}

            for condition in conditions:
                x = epochs[condition] 
                x = x.get_data() # the shape: (n_trials, n_channels, n_times)
                x = x[:,:,start:start + window_size] # 5 sample is a window size
                x = x.reshape(x.shape[0], -1)
                decoding_dict[condition] = x

            # Dictionary to store decoding results for each pair
            pairwise_decoding_accuracies = {}

            for i in range(n_conditions):
                for j in range(i + 1, n_conditions):

                    cond1 = conditions[i]
                    cond2 = conditions[j]
                        
                    # Prepare data
                    data_cond1 = decoding_dict[cond1]
                    data_cond2 = decoding_dict[cond2]
                    data = np.vstack((data_cond1, data_cond2))
                    labels = np.hstack((np.zeros(len(data_cond1)), np.ones(len(data_cond2))))

                    # Set cross-validation
                    cv = StratifiedKFold(n_splits=n_splits) # StratifiedKFold ensures that each fold has a proportional representation of both classes, so each fold maintains a 50:50 balance of numerosity 1 and numerosity 2 trials.

                    # Time-resolved decoding storage
                    decoding_accuracies = []

                    # Cross-validation
                    for train_idx, test_idx in cv.split(data, labels):
                        X_train, X_test = data[train_idx], data[test_idx] # the EEG data for training and testing.
                        y_train, y_test = labels[train_idx], labels[test_idx] # corresponding labels for the training and testing data.
                        clf.fit(X_train, y_train) # trains the classifier on the training data (X_train) with labels (y_train).
                        y_pred = clf.predict(X_test) 
                        decoding_accuracies.append(accuracy_score(y_test, y_pred)) # calculates the classification accuracy by comparing the true labels (y_test) with the predicted labels (y_pred).

                    # average the accuracy within a time point
                    avg_accuracy = np.mean(decoding_accuracies)

                    # Store the average accuracy for the condition pair
                    pairwise_decoding_accuracies[(cond1, cond2)] = avg_accuracy

                    print(f"Average accuracy for {cond1} vs {cond2} in time points {start + window_size}: {avg_accuracy:.2f}")

        # Average accuracy for this time point across CV folds
        all_decoding_accuracy.append(pairwise_decoding_accuracies)
        print(f"Time point {start + window_size} is done")
    # ~~~~~~~~~~~~~~ WINDOW_LOOP



    # ~~~~~~~~~~~~~~~~ Generate Accuracy matrix
    
    all_accuracy_matrices = [] # List to store the resulting accuracy matrices


    # ~~~~~~~~~~~~~~~~  CONDITION_LOOP
    for idx, x in enumerate(all_decoding_accuracy):
        # Extract all unique conditions
        conditions = sorted(set(key[0] for key in x.keys()).union(set(key[1] for key in x.keys())))
        n_conditions = len(conditions)

        # Create a dictionary to map conditions to matrix indices
        condition_idx = {condition: idx for idx, condition in enumerate(conditions)}

        # Initialize a square matrix with NaN values (for easier filling)
        accuracy_matrix = np.full((n_conditions, n_conditions), np.nan)

        # Fill the matrix with accuracies
        for (cond1, cond2), accuracy in x.items(): # e.g., 'cond1' = 'numerosity 1' and 'cond2' = 'numerosity 2'
            i, j = condition_idx[cond1], condition_idx[cond2]
            accuracy_matrix[i, j] = accuracy
            accuracy_matrix[j, i] = accuracy  # Ensuring symmetry

        # Convert to DataFrame for readability
        accuracy_df = pd.DataFrame(accuracy_matrix, index=conditions, columns=conditions)
        
        # Store the DataFrame in the list
        all_accuracy_matrices.append(accuracy_df)

        # Optionally print each matrix to verify (can remove if not needed)
        print(f"Decoding Accuracy Matrix for entry {idx + 1}:")
        print(accuracy_df)
        print("\n")
    # ~~~~~~~~~~~~~~~~  CONDITION_LOOP ~~~~~~~~~~~~~~~~



    # ~~~~~~~~~~~~~~~~ Save the list of accuracy matrices
    save_folder = f"/u/kazma/MINT/data/processed/visual/{subject}"
    save_path = os.path.join(save_folder, "accuracy_matrices.pkl") #  a pickle file
    with open(save_path, "wb") as f:
        pickle.dump(all_accuracy_matrices, f)
    print(f"{subject}: matrices saved in 'accuracy_matrices.pkl'")
    # ~~~~~~~~~~~~~~~~ Save the list of accuracy matrices ~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~ SUB_LOOP ~~~~~~~~~~~~~~