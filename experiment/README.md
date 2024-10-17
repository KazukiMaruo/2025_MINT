# Protocol

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

If you just want to take a look at the experiment, you can skip this process. All of the stimuli are already on the following folder 
```bash
cd MINT/experiment/stimuli/visual
```

1. Open 'visualstimuli_generator.py'
2. Change the parameters and run the script.

```python
# ~~~~~~~~~~~~~ Parameters
width, height = 1920, 1080  # screen pixel dimensions
distance_cm = 50  # Distance from the viewer in cm
screen_width_cm = 50  # Screen width in cm
visual_angle = 2 # visual angle
circle_radius_pixels = compute_radius(width, screen_width_cm, distance_cm, visual_angle) # compute the raidus of circle within the specified visual angle
# ~~~~~~~~~~~~~ Parameters ~~~~~~~~~~~~~
```

### 3. Run the experiment on Psychopy
This experiment works well with
```python 
psychopyVersion = '2024.1.5'
```
1. Open Psychopy.
2. Open 'experiment.py' 
3. Chnage the folloing parameters.
```python 
# chose 'audio' or 'visual'
modality = 'visual'
# specify the monitor size (px), width (cm), and distance (cm) from the screen. 
monitor_data = [1440, 900, 30, 50]   # Ex. EEG lab: [1920, 1080, 30, 50], Mac laptop: [1440, 900, 30, 50] 
# specify whether you need trigger or not.
EEG_Trigger = False
```

## Directory Structure
The experiment folder should be configured like this:
```bash
experiment/
    ├── stimuli/   # contains all the stimuli you need
    │   ├── audio/   # auditory numerosity
    │   │   ├── beep.wav
    │   ├── catch/   # catchy video and audio
    │   │   ├── audio/
    │   │   │   ├── 1.wav
    │   │   │   │   ...
    │   │   │   └── 5.wav
    │   │   ├── video/
    │   │   │   ├── 1.mp4
    │   │   │   │   ...
    │   │   │   └── 16.mp4        
    │   │   └── pause.png   # an image for pause
    │   └── visual/   # visual numerosity
    │       ├── circumference_cont/   # total circumference is controlled across numerosity
    │       │   ├── numerosity_1/
    │       │   │   ...
    │       │   ├── umerosity_6/
    │       ├── singledotsize_cont/   # single dot size is controlled across numerosity
    │       │   ├── numerosity_1/
    │       │   │   ...
    │       │   ├── umerosity_6/               
    │       ├── totaldotsize_cont/   # total dot size is controlled across numerosity
    │       │   ├── numerosity_1/
    │       │   │   ...
    │       │   ├── umerosity_6/ 
    │       └── background.png   # background image
    │
    ├── data/   # output directory from psychopy 
    │   ├── .csv   
    │   ├── .log  
    │   └── .psydat 
    ├── experiment.py   # Psychopy: auditory experiment on Mac
    ├── functions.py   # Psychopy: visual experiment on Mac
    ├── visualstimuli_generator.py   # Generate the visual stimuli
    └── README.md  # Overview of the experiment directory
```


## Contact
If you have any questions, feel free to ask me!
 ```bash
kazuki@cbs.mpg.de
 ```



