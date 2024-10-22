
import sys, os
import mne
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from autoreject import AutoReject

# open parent folder of this script
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# import variables from config.json
import json
with open("config.json") as f:
    config = json.load(f)
globals().update(config)
sys.path.append(BASE_DIR)

# import cusom functions
from src.utils import create_if_not_exist, download_datashare_dir, update_eeg_headers, make_31_montage, calculate_artificial_channels

# DEBUG
#session="sub-001_ses-001"

# get arguments
if len(sys.argv) != 2:
   print("Usage: python script.py session")
   sys.exit(1)
else:
   session = sys.argv[1]

print(f"\n\n Processing session {session} \n\n")

# create directories
create_if_not_exist(f"{BASE_DIR}/data/raw/{session}")
create_if_not_exist(f"{BASE_DIR}/data/interim/{session}")
create_if_not_exist(f"{BASE_DIR}/data/processed/{session}")

# download data from datashare
download_datashare_dir(datashare_dir = f"{DATASHARE_RAW_FOLDER}eeg/{session}", 
                       target_dir = f"{BASE_DIR}data/raw/{session}",
                       datashare_user = DATASHARE_USER)

update_eeg_headers(f"{BASE_DIR}/data/raw/{session}/{session}.vhdr")

# load raw data
raw = mne.io.read_raw_brainvision(f"{BASE_DIR}/data/raw/{session}/{session}.vhdr", 
                                  misc='auto', scale=1.0, preload=True, verbose=False)

# create custom montage
raw.rename_channels({'VP': 'Fp2',  # this is a detour
                     'VM': 'Fp1'})
raw.set_montage(make_31_montage(raw))

# annotations / events
raw.annotations.description = np.array(
    [
        CONDITION_MAP[i] if i in CONDITION_MAP else i
        for i in raw.annotations.description
    ]
)
raw.annotations.duration = np.array([POSTSTIM_WINDOW for _ in raw.annotations.description])

# delete remaining stimulus events
indices_to_remove = [i for i, j in enumerate(raw.annotations.description) if "Stimulus" in j or "Segment" in j or "actiCAP" in j]
raw.annotations.delete(indices_to_remove)

# error correction in first 3 pilots
if session in ["sub-301_ses-001", "sub-303_ses-001", "sub-304_ses-001"]:
    raw.annotations.description = np.roll(raw.annotations.description, 1)
    
# get events and event_id
events, event_id = mne.events_from_annotations(raw, event_id=EVENT_ID)

# resample
raw, events = raw.resample(SFREQ, events = events)
raw.events = events

# print the number of events per condition
event_counts = {}
for key in EVENT_ID.keys():
    event_counts[key] = len(events[events[:, 2] == EVENT_ID[key]])
    print(key, len(events[events[:, 2] == EVENT_ID[key]]))

with open(f"{BASE_DIR}/data/interim/{session}/event_counts_before_drop.json", "w") as f:
    json.dump(event_counts, f)

np.save(f"{BASE_DIR}/data/interim/{session}/events.npy", events)
np.save(f"{BASE_DIR}/data/interim/{session}/event_id.npy", EVENT_ID)

fig = mne.viz.plot_events(events, event_id=EVENT_ID, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, show=False)
fig.savefig(f"{BASE_DIR}/data/interim/{session}/events.png")


# okular data
raw = calculate_artificial_channels(raw.copy(), pairs=[['Fp1', 'Fp2'],['F9', 'F10']], labels=['eyeV', 'eyeH'])
raw.drop_channels(['Fp1']) # former eye channel (dummy name)

# save
raw.save(f"{BASE_DIR}/data/interim/{session}/raw.fif", overwrite=True)

################################################################################
#################### Preprocessing #############################################
################################################################################

# filter
raw.filter(l_freq=PREPROC_PARAMS["hpf"], 
           h_freq=PREPROC_PARAMS["lpf"], 
           method='fir', fir_design='firwin', skip_by_annotation='EDGE boundary', n_jobs=-1)

# eye movement correction

if PREPROC_PARAMS["emc"] == "True":
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None, n_jobs=-1)
    
    ica = ICA(n_components=20, max_iter="auto", method='picard', random_state=97)
    ica.fit(filt_raw) # bads seem to be ignored by default

    # automatic detection of EOG/EMG components
    ica.exclude = []
    # find which ICs match the EOG pattern
    indices, scores = ica.find_bads_eog(raw)
    print(f'Found {len(indices)} independent components correlating with EOG signal.')
    ica.exclude.extend(indices) 

    # barplot of ICA component "EOG/EMG match" scores
    f = ica.plot_scores(scores,
                    title=f'IC correlation with EOG',
                    show=True)
    f.savefig(f"{BASE_DIR}/data/interim/{session}/ica_scores.png", dpi=100)

    # plot diagnostics
    if indices: # only if some components were found to correlate with EOG/EMG
        g = ica.plot_properties(raw, 
                                picks=indices, 
                                show=False)
        for gi, p in zip(g, indices):
            gi.savefig(f"{BASE_DIR}/data/interim/{session}/ica_diagnostics_ic{p}.png", dpi=100)
    plt.close('all')

    ica.apply(raw)

# reference
raw.set_eeg_reference(PREPROC_PARAMS["ref"], projection=False)

# epoching
epochs = mne.Epochs(raw, 
                    events, 
                    event_id=EVENT_ID, #event_id,
                    tmin=PREPROC_PARAMS["base"][0], 
                    tmax=PREPROC_PARAMS["tmax"], # new: longer interval
                    baseline=tuple(PREPROC_PARAMS["base"]),
                    detrend=PREPROC_PARAMS["det"],
                    proj=False,
                    reject_by_annotation=False, 
                    preload=True)

# autoreject
if PREPROC_PARAMS["ar"] != "False":
    
    if PREPROC_PARAMS["ar"] == "interpolate":        
        n_interpolate=[len(epochs.info['ch_names'])] # or False for default hyperparameter finding
        consensus=[len(epochs.info['ch_names'])]  # or False for default hyperparameter finding
    elif PREPROC_PARAMS["ar"] == "interpolate_reject":    
        n_interpolate = [4, 8, 12, 16]
        consensus = np.linspace(0, 1.0, 11)
    else:
        print("ERROR: No valid AR method provided in config.json, exiting!")
        sys.exit(1)
    
    # automated estimation of rejection threshold based on channel and trial per participant
    ar = AutoReject(n_interpolate=n_interpolate, 
                    consensus=consensus,
                    random_state=11,
                    n_jobs=-1, 
                    verbose=False)
    ar.fit(epochs)  # fit only a few epochs if you want to save time
    epochs, reject_log = ar.transform(epochs.copy(), return_log=True)

    # plot the rejection log and save plot
    rej_plot = reject_log.plot('horizontal', show=True)
    rej_plot.savefig(f"{BASE_DIR}/data/interim/{session}/autoreject.png", dpi=100)
    reject_log.save(f"{BASE_DIR}/data/interim/{session}/autoreject.npz", overwrite=True) # must end with .npz
    
    
    
# delete surplus trials --> at the end when formed epochs, because if some epoch forming failed (because eeg ended before tmax), then the rebalancing needs to be performed again
min_trials = np.min(list(event_counts.values()))
dropping_seed = 23
np.random.seed(dropping_seed)
epochs_equalized = []
for key in EVENT_ID.keys():
    if len(epochs[key]) > min_trials:
        n_surplus = len(epochs[key]) - min_trials
        ind_to_drop = np.random.choice(np.arange(len(epochs[key])), n_surplus, replace=False)     
        print(f"Dropped trials in {key}: {ind_to_drop}")   
        epochs_equalized.append(epochs[key].drop(ind_to_drop))
    else:
        epochs_equalized.append(epochs[key])        

epochs = mne.concatenate_epochs(epochs_equalized)

    
epochs.save(f"{BASE_DIR}/data/processed/{session}/epochs-epo.fif", overwrite=True)


################################################################################
#################### Evoked        #############################################
################################################################################

# epochs = mne.read_epochs(f"{BASE_DIR}/data/processed/{session}/epochs-epo.fif")

# evoked = [epochs[i].average(method="mean") for i in EVENT_ID.keys()] # TODO try median

# times=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# l_face_roi = ["P7", "P3", "TP9", "CP5"]
# r_face_roi = ["P8", "P4", "TP10", "CP6"]

# for i, condition in enumerate(EVENT_ID.keys()):
#     p1 = evoked[i].plot_joint(times=times, title=condition, show=False)
#     p1.savefig(f"{BASE_DIR}/data/interim/{session}/evoked_{condition}.png", dpi=100)

# for i, condition in enumerate(EVENT_ID.keys()):
#     p2 = epochs[condition].plot_image(picks=r_face_roi, show=False, title=condition, combine="mean")
#     p2[0].savefig(f"{BASE_DIR}/data/interim/{session}/epochs_r_face_roi_{condition}.png", dpi=100)
        
