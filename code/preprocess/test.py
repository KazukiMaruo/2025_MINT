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


# open parent folder of this script
code_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(code_directory) 
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import variables from config.json
with open("config.json") as f: # The file object is referenced as f during the block.
    config = json.load(f) 
globals().update(config)
sys.path.append(BASE_DIR) # import custom python packages

print(os.getcwd())

# import cusom functions
from utils import create_if_not_exist, download_datashare_dir, update_eeg_headers, make_31_montage, calculate_artificial_channels



# get arguments
modality = 'visual'
session = 1
print(f"\n\n Processing {modality} EEG session {session} \n\n")

# create directories
""" create_if_not_exist(f"{BASE_DIR}/data/raw/{modality}/session_{session}") #  
create_if_not_exist(f"{BASE_DIR}/data/interim/{modality}/session_{session}")
create_if_not_exist(f"{BASE_DIR}/data/processed/{modality}/session_{session}") """


# download data from datashare
download_datashare_dir(datashare_dir = os.path.join(DATASHARE_RAW_FOLDER, modality, 'sub-01_ses-01'),
                       target_dir = os.path.join(BASE_DIR, 'data', 'raw', modality), #  "BASE_DIR": "/u/kazma/MINT/",
                       datashare_user = DATASHARE_USER) # "DATASHARE_USER": "kazma",


# get eeg headers from datashare
update_eeg_headers(f"{BASE_DIR}/data/raw/{modality}/sub-01_ses-01.vhdr") # BrainVision file format commonly used for EEG data.a