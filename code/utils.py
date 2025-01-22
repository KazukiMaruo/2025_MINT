import os
import numpy as np
import getpass
from pathlib import Path
import keyring
import owncloud  # pip install pyocclient
import mne

def create_if_not_exist(dirs):
    """
    Recursively create directories, if not already existent.

    Args:
        dirs (list): List of directories to be created.
    """
    if not isinstance(dirs, list):
        dirs = [dirs]
    for d in dirs:
        if not os.path.exists(d):
            try:
                os.makedirs(d)
            except FileExistsError:
                print(f"Could not create directory {d} because it existed. Maybe it was created by a parallel process.")


def datashare_establish_connection(datashare_user):
    """Establishes connection to MPCDF DataShare.
    adapted from: https://github.com/SkeideLab/bids_template_code/blob/b7751c5ed7419d4401b07112d7dd553a9b99ffca/helpers.py#L48
    
    Args:
        datashare_user : str
            Username for MPCDF Datashare. If None, the username will be
            retrieved from the system (only works in MPCDF HPC).

    Returns:
        owncloud.Client
    """

    # Get DataShare login credentials
    if not datashare_user:
        datashare_user = getpass.getuser()  # get username from system (only works in MPCDF HPC)

    datashare_pass = keyring.get_password('datashare', datashare_user)
    # datashare_pass = 'Xt98iCm4XQHH!vf'
    if datashare_pass is None:
        datashare_pass = getpass.getpass()
        keyring.set_password('datashare', datashare_user, datashare_pass)        
    
    # Login to DataShare
    domain = 'https://datashare.mpcdf.mpg.de'
    datashare = owncloud.Client(domain)
    datashare.login(datashare_user, datashare_pass)

    return datashare

def download_datashare_dir(datashare_dir, target_dir, datashare_user):  # target_dir was bids_ds
    """Downloads all files within a directory from MPCDF DataShare.
    adapted from: https://github.com/SkeideLab/bids_template_code/blob/b7751c5ed7419d4401b07112d7dd553a9b99ffca/helpers.py#L48
    Please change your datashare credentials in the code below.
    
    Args:
        datashare_dir (str): Path of the raw data starting from the DataShare root, like this:
            https://datashare.mpcdf.mpg.de/apps/files/?dir=<datashare_dir>. It
            is e.g. PRAWN/raw/<modelity>/sub-<sub>_ses-<ses> for the PRAWN study.
            All files in this folder will be downloaded.
        target_dir (str): Local directory where the files will be downloaded to. 
            The directory will be created if it does not exist.
    """
    # establish connection
    datashare = datashare_establish_connection(datashare_user=datashare_user)

    # Loop over session folders on DataShare
    files = datashare.list(datashare_dir)
    for file in files:
        # Explicity exclude certain file names
        if file.name.startswith('_'):
            continue

        # Download if it doesn't exist
        local_file = Path(f"{target_dir}/{file.name}")
        if not local_file.exists():  # and not exclude_file.exists():
            
            # Download zip file
            print(f'Downloading {file.path} to {local_file}')
            create_if_not_exist(target_dir)
            datashare.get_file(file, local_file)        
        else:
            print(f"File {local_file} already exists. Skipping download.")


def update_eeg_headers(file):
    """
    Uses a .vhdr file and updates the references to .eeg and .vmrk files based on the
    current filename of the .vhdr file.
    This must be done if a raw file was renamed, as BrainVision writes the original filename in the headers.
    
    Args:
        file (str): Path to the .vhdr file
    """
    # Read the .vhdr file
    with open(file, 'r') as f:
        lines = f.readlines()

    # Update the lines
    updated_lines = []
    for line in lines:
        if line.startswith("DataFile="):
            # Replace characters after "DataFile=" with the filename and ".eeg"
            line = "DataFile=" + os.path.basename(file).replace(os.path.splitext(file)[1], ".eeg") + "\n"
        elif line.startswith("MarkerFile="):
            # Replace characters after "MarkerFile=" with the filename and ".vmrk"
            line = "MarkerFile=" + os.path.basename(file).replace(os.path.splitext(file)[1], ".vmrk") + "\n"
        updated_lines.append(line)

    # Write the updated content back to the file
    with open(file, 'w') as f:
        f.writelines(updated_lines)

    # Open and update the .vmrk file
    vmrk_file = os.path.splitext(file)[0] + ".vmrk"
    if os.path.exists(vmrk_file):
        with open(vmrk_file, 'r') as f:
            vmrk_lines = f.readlines()

        # Update the lines
        updated_vmrk_lines = []
        for line in vmrk_lines:
            if line.startswith("DataFile="):
                # Replace content after "DataFile=" with the filename of "file" but with extension ".eeg"
                line = "DataFile=" + os.path.basename(file).replace(os.path.splitext(file)[1], ".eeg") + "\n"
            updated_vmrk_lines.append(line)

        # Write the updated content back to the .vmrk file
        with open(vmrk_file, 'w') as f:
            f.writelines(updated_vmrk_lines)
            
def make_31_montage(raw):
    """
    Defines the GHOST montage for PRAWN
    adapted from https://stackoverflow.com/questions/58783695/how-can-i-plot-a-montage-in-python-mne-using-a-specified-set-of-eeg-channels
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        plot (bool): Whether to plot the montage.
        save (bool): Whether to save the montage.
    
    Returns: montage
    """
    # Form the 10-20 montage
    mont1020 = mne.channels.make_standard_montage('standard_1020')
    # Choose what channels you want to keep
    # Make sure that these channels exist e.g. T1 does not exist in the standard 10-20 EEG system!
    kept_channels = raw.ch_names
    # additionally recode VP to Fp2
    add_channels = ['Fp2']

    ind = [i for (i, channel) in enumerate(mont1020.ch_names) if (channel in kept_channels) or (channel in add_channels)]
    mont = mont1020.copy()
    
    # Keep only the desired channels
    mont.ch_names = [mont1020.ch_names[x] for x in ind]
    kept_channel_info = [mont1020.dig[x + 3] for x in ind]
    
    # Keep the first three rows as they are the fiducial points information
    mont.dig = mont1020.dig[:3] + kept_channel_info

    return mont


def calculate_artificial_channels(raw, pairs=[['Fp1', 'Fp2'], ['F9', 'F10']], labels=['eyeV', 'eyeH']):
    """
    Calculate artificial eye movement electrodes, by subtracting two electrodes
    timeseries from each other.

    Substraction of fitting electrodes leads to boosting of the 
    (anticorrelated) eye movement signals,
    while canceling out brain-related signals or other signals.
    The electrodes need to be on opposite sites of the eye(s).

    For instance: VM and VP (above and below the right eye). Vertical movement
    (called Fp1 and Fp2, whereas one is the EOG electrode in reality).
    T9 and T10 (left and right from the eyes). Horizontal movement.

    Args:
        raw (mne.raw) raw data object
        pairs (nested list): list of list of electrode names which are subtracted from each other
        labels (list): list of labels (names) of the new artificial electrodes
    
    Returns:
        raw (mne.raw): processed raw data object
    """
    
    # Create a copy of the original raw data
    raw_new = raw.copy()

    for i in range(len(pairs)):
        
        # Specify the names of the existing channels you want to subtract
        channel1 = pairs[i][0] #'Fp1'
        channel2 = pairs[i][1] #'Fp2'  # was VP and VM, but got recorded as Fp1 and Fp2
        
        # Subtract the values of channel2 from channel1 and create a new channel
        new_channel_data = raw[channel1][0] - raw[channel2][0]
        new_channel_name = labels[i] #'new_channel'

        # Reshape the new channel data to have shape (1, n_samples)
        new_channel_data = np.reshape(new_channel_data, (1, -1))

        # Create a new info object for the new channel
        new_info = mne.create_info([new_channel_name], raw_new.info['sfreq'], ch_types='eog') # WAS EEG

        # Create a new RawArray object for the new channel
        new_channel_raw = mne.io.RawArray(new_channel_data, new_info)

        # Add the new channel to the raw data
        raw_new.add_channels([new_channel_raw], force_update_info=True)

    return raw_new



# find all sessions of the subject TODO in utils
def get_epochs_from_sessions(subject, BASE_DIR):
    sessions = [f.name for f in os.scandir(f"{BASE_DIR}/data/raw/") if f.is_dir() and subject in f.name]
    epochs = []
    for session in sessions:
        # load epochs
        epochs.append(mne.read_epochs(f"{BASE_DIR}/data/processed/{session}/epochs-epo.fif", verbose=False))
    epochs = mne.concatenate_epochs(epochs)
    return epochs
