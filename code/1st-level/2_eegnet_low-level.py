# ~~~~~~~~~~~~~~ Libraries
import sys, os
import mne # Python package for processing and analyzing electrophysiological data
import numpy as np
from glob import glob # look for all the pathnames matching a specified pattern according to the rules
import matplotlib.pyplot as plt
from mne.preprocessing import ICA # ICA (Independent Component Analysis) algorithm, which is for artifact removal
from autoreject import AutoReject # Python package for automatically rejecting bad epochs in EEG/MEG data
import json
import owncloud
import pandas as pd
import braindecode
import torch
import re
import pickle
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, cross_validate, StratifiedKFold

# Deep learning
from braindecode.models import EEGNetv4
from braindecode.preprocessing import exponential_moving_standardize
from braindecode.util import set_random_seeds
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier
# ~~~~~~~~~~~~~~ Libraries ~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~ Parameters


# EEGNET parameters
# Load the JSON file
code_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(code_directory) 
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open("config.json") as f: # import variables from config.json
    config = json.load(f) 
globals().update(config)


group = 'adult'
modality = 'visual' # 'visual' or 'audio'

### Subject loop
subject_lists = ['sub-06','sub-07','sub-08']


for subject in subject_lists:

    # Print out each parameter
    print(f"{modality} data of {subject} is processed")
    print("EEGNET parameters:")
    print(f"====== CV: {EEGNET_CV}")
    print(f"====== Max epochs: {EEGNET_MAX_EPOCHS}")
    print(f"====== Batch size: {EEGNET_BATCH_SIZE}")
    # ~~~~~~~~~~~~~~ Parameters ~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~ Set the working directory
    path = f"{COMPUTE_DIR}/data/{group}/interim/{modality}"
    sub_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    sub_folders_sorted = sorted(sub_folders, key=lambda x: int(re.search(r'\d+', x).group())) # Sort the folders based on the numeric part after "sub-"
    # ~~~~~~~~~~~~~~ Set the working directory ~~~~~~~~~~~~~~


    # ~~~~~~~~~~~~~~ Concatanating 3 sessions
    each_sub_path = f"{COMPUTE_DIR}/data/{group}/interim/{modality}/{subject}"
    each_sub_folders = [f for f in os.listdir(each_sub_path) if os.path.isdir(os.path.join(each_sub_path, f))]
    each_sub_folders_sorted = sorted(each_sub_folders, key=lambda x: int(re.search(r'\d+', x).group()))

    eegdata_dict = {}
    for x,ses_loop in enumerate(each_sub_folders_sorted):
        sub_filename = os.path.join(path, subject, ses_loop, 'epochs-epo.fif') 
        epochs = mne.read_epochs(sub_filename, preload=True)
        eegdata_dict[ses_loop] = epochs

    # concatanate all 3 sessions into 1 epoch
    epochs = mne.concatenate_epochs([eegdata_dict['ses-01'], eegdata_dict['ses-02'], eegdata_dict['ses-03']])
    # ~~~~~~~~~~~~~~ Concatanating 3 sessions ~~~~~~~~~~~~~~



    # ~~~~~~~~~~~~~~ Concatanating 3 sessions csv file
    each_sub_path = f"{COMPUTE_DIR}/data/{group}/raw/{modality}/{subject}"
    each_sub_folders = [f for f in os.listdir(each_sub_path) if os.path.isdir(os.path.join(each_sub_path, f))]
    each_sub_folders_sorted = sorted(each_sub_folders, key=lambda x: int(re.search(r'\d+', x).group()))

    csv_dict = {}
    for x,ses_loop in enumerate(each_sub_folders_sorted):
        csv_file = f"{subject}_{modality}_{ses_loop}.csv"
        sub_filename = os.path.join(each_sub_path, ses_loop, csv_file) 
        csv_df = pd.read_csv(sub_filename)
        csv_dict[ses_loop] = csv_df

    # concatanate all 3 sessions into 1 epoch
    csv_dfs = pd.concat([csv_dict['ses-01'], csv_dict['ses-02'], csv_dict['ses-03']], axis=0)
    # ~~~~~~~~~~~~~~ Concatanating 3 sessions csv file~~~~~~~~~~~~~~



    # ~~~~~~~~~~~~~~ Obtain target conditions' trials
    csv_dfs['Extracted'] = csv_dfs['condition'].str.split('_').str[-1]
    csv_idx = csv_dfs[['condition', 'Extracted']]
    condition_lists = csv_idx['Extracted'].unique()

    # condition loop
    for target_condition in condition_lists:

        idx_to_keep = csv_dfs.index[csv_dfs['Extracted'] == target_condition].tolist()
        selected_epochs = epochs[idx_to_keep]
        print(f"Selected trials: {len(selected_epochs)}")
        print(f"Target condition: {target_condition}")
        # ~~~~~~~~~~~~~~ Obtain target conditions' trials ~~~~~~~~~~~~~~


        # Crop epochs to the desired time range
        cropped_epochs = selected_epochs.copy().crop(tmin=EEGNET_MIN_TIME, tmax=EEGNET_MAX_TIME)

        # Get the info about the cropped data
        conditions = list(cropped_epochs.event_id.keys()) # list of conditions
        n_conditions = len(conditions) # number of conditions
        n_trials = len(cropped_epochs) # number of trials
        n_samples = cropped_epochs.get_data().shape[2]
        n_channels = cropped_epochs.get_data().shape[1]
        min_time = cropped_epochs.times[0]*1000   # First time point in milli seconds
        max_time = cropped_epochs.times[-1]*1000    # Last time point in milli seconds


        print("=====================================================")
        print("=====================================================")
        print("=====================================================")

        print(f" Condition lists: {conditions}")
        print(f" Total trials: {n_trials}")
        print(f" Time points: {n_samples}")
        print(f" Time window (ms): {min_time} - {max_time}")


        # check GPU availability
        cuda = torch.cuda.is_available()  # Check if a GPU is available
        device = "cuda" if cuda else "cpu"  # Use "cuda" if available, otherwise fallback to "cpu"

        if cuda:
            torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuning for performance


        # Set random seeds
        # PURPOSE: reproducibility across runs, random initializations (e.g., model weights) yield consistant results
        seed = 20200220
        set_random_seeds(seed=seed, cuda=cuda)


        # dictionaries for decoding accuracy
        pairwise_decoding_accuracies = {}
        # dictionaries for decoding accuracy of standard deviation
        pairwise_decoding_accuracies_std = {}
        # dictionaries for estimators
        pairwise_decoding_estimators = {}


        for i in range(n_conditions):
            for j in range(i + 1, n_conditions):

                    cond1 = conditions[i]
                    cond2 = conditions[j]

                    print(f"{cond1} vs. {cond2}")
                    
                    filtered_epochs = cropped_epochs[cond1,cond2]

                    # get EEG data
                    X = filtered_epochs.get_data(picks=mne.pick_types(filtered_epochs.info, eeg=True, eog=False, exclude='bads'))

                    # get labels (=numerosity)
                    unique_labels = np.unique(filtered_epochs.events[:,-1])
                    label_0, label_1 = unique_labels[0], unique_labels[1]  # Assign the first label to 0 and the second to 1

                    y = np.where(filtered_epochs.events[:, -1] == label_0, 0, 1)

                    skfold = StratifiedKFold(n_splits=EEGNET_CV, shuffle=True, random_state=23)

                    # exp. moving std. for each trial
                    for s in range(X.shape[0]):
                        X[s,:,:] = exponential_moving_standardize(X[s,:,:], factor_new=0.001, init_block_size=None, eps=1e-4)

                    # create the model
                    net = EEGClassifier(
                        "EEGNetv4", 
                                    module__n_chans=n_channels, # Number of EEG channels
                                    module__n_outputs=2,               # Number of outputs of the model. This is the number of classes in the case of classification.
                                    module__n_times=n_samples,         # Number of time samples of the input window.
                                    module__final_conv_length='auto',  # Length of the final convolution layer. If "auto", it is set based on the n_times.
                                    module__pool_mode='mean',          # Pooling method to use in pooling layers
                                    module__F1=8,                      # Number of temporal filters in the first convolutional layer.
                                    module__D=2,                       # Depth multiplier for the depthwise convolution.
                                    module__F2=16,                     # Number of pointwise filters in the separable convolution. Usually set to ``F1 * D``.
                                    module__kernel_length=64,         # Length of the temporal convolution kernel. Usally sampling rate / 2 = 500/2 = 250
                                    module__third_kernel_size=(8, 4), 
                                    module__drop_prob=0.25,            # Dropout probability after the second conv block and before the last layer. 0.5 for within-subject classification, 0.25 in cross-subject clasification
                                    module__chs_info=None,             # (list of dict) – Information about each individual EEG channel. This should be filled with info["chs"]. Refer to mne.Info for more details.
                                    module__input_window_seconds=None, # Length of the input window in seconds.
                                    module__sfreq=SFREQ,
                                    max_epochs=EEGNET_MAX_EPOCHS,
                                    batch_size=EEGNET_BATCH_SIZE,
                                    train_split=None,
                    )


                    cvs = cross_validate(net, 
                                        X, 
                                        y, 
                                        scoring="accuracy", # for balanced classes, this corresponds to accuracy,
                                        # chance level might be 0 (adjusted = False), or 0.X (adjusted = True)
                                        cv=skfold, 
                                        n_jobs=-1, # only 1 to avoid overload in parallel jobs, in non-par jobs it could be -1
                                        return_estimator=True, # if you need the model to estimate on another test set
                                        return_train_score=False,
                                        )
                    
                    pairwise_decoding_accuracies[(cond1,cond2)] = np.mean(cvs['test_score'])
                    pairwise_decoding_accuracies_std[(cond1,cond2)] = np.std(cvs['test_score'])
                    pairwise_decoding_estimators[(cond1,cond2)] = cvs['estimator']


        # ~~~~~~~~~~~~~~~~ Save the decoding accuracy
        save_folder = f"{COMPUTE_DIR}/data/{group}/processed/{modality}/{subject}"

        filename = f"EEGNet_accuracy_pairwise_{target_condition}.pkl"
        save_path = os.path.join(save_folder, filename) #  a pickle file
        with open(save_path, "wb") as f:
            pickle.dump(pairwise_decoding_accuracies, f)
        print(f"{subject}: saved in {filename}")
        # ~~~~~~~~~~~~~~~~ Save the decoding accuracy ~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~ Save the decoding accuracy standard deviation
        filename = f"EEGNet_std_pairwise_{target_condition}.pkl"
        save_path = os.path.join(save_folder, filename) #  a pickle file
        with open(save_path, "wb") as f:
            pickle.dump(pairwise_decoding_accuracies_std, f)
        print(f"{subject}: saved in {filename}")
        # ~~~~~~~~~~~~~~~~ Save the decoding accuracy standard deviation ~~~~~~~~~~~~~~~~


        # ~~~~~~~~~~~~~~~~ Save the EEGNet estimation
        filename = f"EEGNet_estimator_pairwise_{target_condition}.pkl"
        save_path = os.path.join(save_folder, filename) #  a pickle file
        with open(save_path, "wb") as f:
            pickle.dump(pairwise_decoding_estimators, f)
        print(f"{subject}: saved in {filename}")
        # ~~~~~~~~~~~~~~~~ Save the EEGNet estimation ~~~~~~~~~~~~~~~~