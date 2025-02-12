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

import scipy.spatial.distance as dist
import statsmodels.api as sm
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.families.links import Identity  # Updated import


# ~~~~~~~~~~~~~~ Parameters
group = 'adult'
modality = 'visual' # 'visual' or 'audio'

# RDM parameter
whichRDM = 'dynamic' #'dynamic' 

# Print out each parameter
print(f"{modality} data is processed")
print("RDM parameters:")
print(f"  Type: {whichRDM}")
# ~~~~~~~~~~~~~~ Parameters ~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~ Set the working directory
path = f"/u/kazma/MINT/data/{group}/interim/{modality}"
sub_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
sub_folders_sorted = sorted(sub_folders, key=lambda x: int(re.search(r'\d+', x).group())) # Sort the folders based on the numeric part after "sub-"
# ~~~~~~~~~~~~~~ Set the working directory ~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~ Load model RDM
rdm_path = f"/u/kazma/MINT/code/1st-level/RDM"
# Load the .npy file
# Load the .npy file
RDM_numerosity = np.load(f"{rdm_path}/rdm_numerosity.npy")
RDM_spatialfrequency = np.load(f"{rdm_path}/rdm_spatial_frequency.npy")
RDM_signledot = np.load(f"{rdm_path}/rdm_area_of_a_single_dot.npy")
RDM_totaldot = np.load(f"{rdm_path}/rdm_area_of_total_dots.npy")
RDM_circumference = np.load(f"{rdm_path}/rdm_circumference_of_total_dots.npy")
# ~~~~~~~~~~~~~~ Load model RDM ~~~~~~~~~~~~~~


# SUB_LOOP
# for subject in sub_folders_sorted

subject = sub_folders_sorted[1]

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


rdm_order = ['intercept','numerosity_RDM', 'spatialfrequency_RDM', 'signledot_RDM', 'totaldot_RDM', 'circumference_RDM']

betas = np.zeros((len(rdm_order), n_samples)) # 6 = intercept plus betas for each moldel
p_values = np.zeros((len(rdm_order), n_samples)) # 6 = intercept plus betas for each moldel


### Time point LOOP
for timepoint in range(n_samples):

    df_dict = {}

    df = np.zeros((n_conditions, n_channels))

    for i, condition in enumerate(conditions):
        x = epochs[condition]
        x = x.get_data()
        x = x[:,:,timepoint] # because 0 means -100ms, so 50 means the onset
        s = x.reshape(x.shape[0], -1)
        df[i,:] = s
        df_dict[condition] = s


    df_pd = pd.DataFrame(df)
    df_pd.index = list(df_dict.keys())


    # Define the order for conditions
    condition_order = {'singledot': 0, 'totaldot': 1, 'circum': 2}

    df_sort = df_pd.loc[sorted(df_pd.index, key=lambda x: int(x.split('_')[-1].split('.')[0]))]
    df_sort = df_sort.loc[sorted(df_sort.index, key=lambda x: condition_order[x.split('_')[2]])]
    df_sort = df_sort.loc[sorted(df_sort.index, key=lambda x: int(x.split('_')[0]))]


    rdm = dist.pdist(df_sort, metric='correlation')
    rdm_square = dist.squareform(rdm)  
    upper_tri_mask = np.triu(np.ones(rdm_square.shape), k=0).astype(bool) # Create an upper triangular mask (including the diagonal)
    rdm_square[upper_tri_mask] = np.nan   # Apply the mask and set the upper triangle to NaN

    # Min-Max normalization: Normalize the entire matrix
    rdm_minmax_normalized = (rdm_square - np.nanmin(rdm_square)) / (np.nanmax(rdm_square) - np.nanmin(rdm_square))


    # Multiple regressions
    # Multiple regressions
    # vectorize the RDMs
    neural_RDM = rdm_minmax_normalized[np.tril_indices(rdm_minmax_normalized.shape[0], k=-1)]

    numerosity_RDM = RDM_numerosity[np.tril_indices(RDM_numerosity.shape[0], k=-1)]
    spatialfrequency_RDM = RDM_spatialfrequency[np.tril_indices(RDM_spatialfrequency.shape[0], k=-1)]
    signledot_RDM = RDM_signledot[np.tril_indices(RDM_signledot.shape[0], k=-1)]
    totaldot_RDM = RDM_totaldot[np.tril_indices(RDM_totaldot.shape[0], k=-1)]
    circumference_RDM = RDM_circumference[np.tril_indices(RDM_circumference.shape[0], k=-1)]


    Y = neural_RDM
    X = np.column_stack([numerosity_RDM, spatialfrequency_RDM, signledot_RDM, totaldot_RDM, circumference_RDM])
    X = sm.add_constant(X)
    model = sm.GLM(Y, X, family=Gaussian(link=Identity()))
    results = model.fit()

    # print(results.summary())
    coefficient = results.params  # Regression coefficients
    p_value = results.pvalues     # P-values for the coefficients

    betas[:,timepoint] = coefficient # 6 = intercept plus betas for each moldel
    p_values[:,timepoint] = p_value

    print(f"  {subject}: Timepoint {timepoint} proccessed")

### END: Time point LOOP

# convert it into pd.dataframe
betas_pd = pd.DataFrame(betas)
betas_pd.index =  rdm_order

p_values_pd = pd.DataFrame(p_values)
p_values_pd.index =  rdm_order


# ~~~~~~~~~~~~~~~~ Save the decoding accuracy
save_folder = f"/u/kazma/MINT/data/adult/processed/{modality}/{subject}"
save_path = os.path.join(save_folder, "rdm_beta.pkl") #  a pickle file
with open(save_path, "wb") as f:
    pickle.dump(betas_pd, f)
print(f"{subject}: saved in 'rdm_beta.pkl'")

save_path = os.path.join(save_folder, "rdm_beta_pvals.pkl") #  a pickle file
with open(save_path, "wb") as f:
    pickle.dump(p_values_pd, f)
print(f"{subject}: saved in 'rdm_beta_pvals.pkl'")
# ~~~~~~~~~~~~~~~~ Save the decoding accuracy ~~~~~~~~~~~~~~~~