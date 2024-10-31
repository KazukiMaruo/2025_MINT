# ~~~~~~~~~~~~~~ Libraries
import sys, os
import mne # Python package for processing and analyzing electrophysiological data
import numpy as np
from glob import glob # look for all the pathnames matching a specified pattern according to the rules
import matplotlib.pyplot as plt
from mne.preprocessing import ICA # ICA (Independent Component Analysis) algorithm, which is for artifact removal
from autoreject import AutoReject # Python package for automatically rejecting bad epochs in EEG/MEG data
import json
# ~~~~~~~~~~~~~~ Libraries ~~~~~~~~~~~~~~



#############################################################################
#################### ENVIROMENT #############################################
#############################################################################

# ~~~~~~~~~~~~~~ open json paramete files and import functions
code_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(code_directory) 
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open("config.json") as f: # import variables from config.json
    config = json.load(f) 
globals().update(config)
sys.path.append(BASE_DIR) # import custom python packages
print(os.getcwd())
from utils import create_if_not_exist, download_datashare_dir, update_eeg_headers, make_31_montage, calculate_artificial_channels
# ~~~~~~~~~~~~~~ open json paramete files and import functions ~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~ Pre-processing Parameters
modality = 'visual'
session = 1
sub_name = 'sub-01_ses-01'
print(f"\n\n Processing {modality} EEG session {session} of {sub_name}\n\n")
# ~~~~~~~~~~~~~~ Pre-processing Parameters ~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~ Path settings and make folders
datashare_dir_path = os.path.join(DATASHARE_RAW_FOLDER, modality, sub_name) #  "DATASHARE_RAW_FOLDER": "MINT/raw/",
# create directories
raw_target_dir_path = os.path.join(BASE_DIR, 'data', 'raw', modality, sub_name)
interim_target_dir_path = os.path.join(BASE_DIR, 'data', 'interim', modality, sub_name)
processed_target_dir_path = os.path.join(BASE_DIR, 'data', 'processed', modality, sub_name)
create_if_not_exist(raw_target_dir_path) 
create_if_not_exist(interim_target_dir_path)
create_if_not_exist(processed_target_dir_path) 
# the interested file name
target_file_name = f"{raw_target_dir_path}/{sub_name}.vhdr"
# ~~~~~~~~~~~~~~ Path settings and make folders ~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~ load data from datashare
download_datashare_dir(datashare_dir = datashare_dir_path,
                       target_dir = raw_target_dir_path, 
                       datashare_user = DATASHARE_USER) # "DATASHARE_USER": "kazma",
# get eeg headers from datashare
update_eeg_headers(target_file_name) 
# ~~~~~~~~~~~~~~ load data from datashare ~~~~~~~~~~~~~~


###############################################################################
#################### LOADING DATA #############################################
###############################################################################


# ~~~~~~~~~~~~~~ Load data and the data ajustments
raw = mne.io.read_raw_brainvision(target_file_name,  # Brain vision file format contains three files .vhdr for the header, .eeg for the data, and .vmrk for events
                                  misc='auto', # misc = Specifies that channels labeled as miscellaneous (e.g., auxiliary or non-EEG channels) should be automatically detected.
                                  scale=1.0, # 1.0 means no change for scale
                                  preload=True, # the data is loaded directly into memory. This allows for faster processing 
                                  verbose=False) # Suppresses output during the loading process (helpful when you don’t want too much logging information
raw.rename_channels({'VP': 'Fp2',  # In this case, the channel labeled VP is renamed to Fp2, and VM is renamed to Fp1.
                     'VM': 'Fp1'})
raw.set_montage(make_31_montage(raw))

# Trigger value is converted to condition label name (e.g., from 106 to 6_con_totaldot)
raw.annotations.description = np.array(
    [
        CONDITION_MAP[i] if i in CONDITION_MAP else i
        for i in raw.annotations.description
    ]
)
# interested duration of each trial (e.g., 800ms after the onset of image)
raw.annotations.duration = np.array([POSTSTIM_WINDOW for _ in raw.annotations.description])

# delete remaining stimulus events / This part of the code removes specific types of events from the annotations in the EEG data
indices_to_remove = [i for i, j in enumerate(raw.annotations.description) if "Stimulus" in j or "Segment" in j or "actiCAP" in j]
raw.annotations.delete(indices_to_remove)

# get events (number of events, 3), each row represents [event time, previous_event, event_id]
events, event_id = mne.events_from_annotations(raw, event_id=EVENT_ID)

# resample
raw, events = raw.resample(SFREQ, events = events) # raw.resample(SFREQ, events=events): Resamples the raw EEG data to a new sampling frequency (SFREQ) while adjusting the timing of the events accordingly.
raw.events = events #  This assigns the resampled events back to the raw object, so the events remain associated with the data after resampling.
# ~~~~~~~~~~~~~~ Load data and the data ajustments ~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~ print and save
# print the number of trials for each condition
event_counts = {}
for key in EVENT_ID.keys():
    event_counts[key] = len(events[events[:, 2] == EVENT_ID[key]])
    print(key, len(events[events[:, 2] == EVENT_ID[key]]))

# Save event counts in jsonfile
with open(f"{interim_target_dir_path}/event_counts_before_drop.json", "w") as f:
    json.dump(event_counts, f)

# This saves the events and the event ID mapping as .npy (NumPy binary) files for future use.
np.save(f"{interim_target_dir_path}/events.npy", events) # This function saves the array or object
np.save(f"{interim_target_dir_path}/event_id.npy", EVENT_ID)

# This plots the events and saves the visualization as a .png image.
fig = mne.viz.plot_events(events, event_id=EVENT_ID, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, show=False)
fig.savefig(f"{interim_target_dir_path}/events.png")

# creates new artificial channels representing eye movements, using specific EEG channel pairs.
raw = calculate_artificial_channels(raw.copy(), pairs=[['Fp1', 'Fp2'],['F9', 'F10']], labels=['eyeV', 'eyeH']) # ['Fp1', 'Fp2']: These are the frontal electrodes typically used to capture vertical eye movements (denoted as eyeV in this case). ['F9', 'F10']: These electrodes are often placed near the eyes and can be used to capture horizontal eye movements (denoted as eyeH).
raw.drop_channels(['Fp1']) # former eye channel (dummy name)

# save
raw.save(f"{interim_target_dir_path}/raw.fif", overwrite=True) # .fif file, which is a standard file format used in MNE for storing EEG/MEG data.
# ~~~~~~~~~~~~~~ print and save


################################################################################
#################### Preprocessing #############################################
################################################################################

# band-pass filter
raw.filter(l_freq=PREPROC_PARAMS["hpf"], # high-pass
           h_freq=PREPROC_PARAMS["lpf"], # low-pass
           method='fir', #  Finite Impulse Response (FIR) filter 
           fir_design='firwin', # a popular window method for designing FIR filters
           skip_by_annotation='EDGE boundary', # This skips data points marked by annotations such as "EDGE" or "boundary"
           n_jobs=-1) # This allows the filtering to be done in parallel, using all available CPU cores