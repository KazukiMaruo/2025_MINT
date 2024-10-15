# Protocol for the experiment

## Overview

This repository contains numerosity stimuli and outlines the steps for the experiment set up. The goal is to ensure transparency, reproducibility, and clarity of the experiment for all.

## Table of Contents

1. [Objective](#objective)
2. [Steps for Reproducibility](#steps-for-reproducibility)
3. [Directory Structure](#directory-structure)
4. [Contact](#contact)


## Objective

The objective of this experiment is to:
- Explore electrophysiological representation (= EEG) of numerosity in infants.
- Explore the cross-modality (auditory vs. visual).


## Steps for Reproducibility

This section outlines the pipeline, with step-by-step explanations of how to set up the experiment.

### 1. Download the experiment repository. 
- **Option 1**: Clone via Git (Recommended)

　If you have Git installed, open your terminal, go to the directory where you want to download, and run the following command:

  ```bash
  git clone --no-checkout https://github.com/SkeideLab/MINT.git
  cd MINT

  git sparse-checkout init --cone
  git sparse-checkout set experiment
  git checkout main
  ```

- **Option 2**: Download as a Zip file 

　From the repository's homepage on GitHub, click on the green '< >Code' button, and then downlaod the Zip.


### 2. Generate the visual stimuli based on your experiment setting

If you just want to take a look at the experiment, you can skip this process. All of the stimuli are alaready on the folder 
  ```bash
    cd MINT/experiment/stimuli/visual
  ```

1. Open 'visualstimuli_generator.py'
2. Change the parameters and run the script.

  ```bash
    # ~~~~~~~~~~~~~ Parameters
    width, height = 1920, 1080  # screen pixel dimensions
    distance_cm = 50  # Distance from the viewer in cm
    screen_width_cm = 50  # Screen width in cm
    visual_angle = 2 # visual angle
    circle_radius_pixels = compute_radius(width, screen_width_cm, distance_cm, visual_angle) # compute the raidus of circle within the specified visual angle
    # ~~~~~~~~~~~~~ Parameters ~~~~~~~~~~~~~
  ```


### 3. Check stimuli directory
The experiment folder should be configured like this:
  ```bash
    MINT/experiment
        ├── stimuli/
        │   ├── audio/  # Original raw data
        │   ├── catch/  # Cleaned and processed data
        │   └── visual/ # Description of the datasets
        ├── data/
        │   ├── .csv  # 
        │   ├── .log  # 
        │   └── .psydat 
        ├── main__adudio_mac.py  
        ├── main__visual_mac.py  
        ├── main__adudio.py 
        ├── main__visual.py
        ├── visualstimuli_generator.py  
        └── README.md  # Overview of the experiment directory
  ```






