# Protocol

## Overview

This repository contains scripts for data analysis and outlines the steps. The goal is to ensure transparency, reproducibility, and clarity of the data analysis for all.

## Table of Contents

1. [Objective](#objective)
2. [EEG preprocessing](#EEG-preprocessing)
3. [Single level analysis](#Single-level-analysis)
4. [Group level analysis](#Group-level-analysis)
5. [Directory Structure](#directory-structure)
6. [Contact](#contact)


## Objective

The objective is to test whether the EEG data is distinct by numerosity using:
- Time-resolved multivariate pattern analysis (MVPA).
- Representational similarity analysis (RSA).


## EEG preprocessing

This section outlines the preprocessing pipeline, with step-by-step explanations.

### 1. Download the code folder. 
1. Open https://download-directory.github.io/
2. Paste the URL of 'code' folder
3. Place it in the 'MINT' folder

### 2. Run preprocessing

1. go to 'pre-process' folder
2. Run 'preprocess.py'
 ```bash
python preprocess.py
 ```


## Single level analysis

### 1. Generate representational dissimilarity matrix.
1. go to '1st-level' folder
2. run '1_rdm_generator.py'
 ```bash
python 1_rdm_generator.py
 ```

- **NOTE:** 'param.json' in [stimuli folder](experiment/stimuli/visual/param.json) is necessary 


## Group level analysis

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
    │       ├── background.png   # background image
    │       └── param.json   # parameters of visual stimuli
    │
    ├── data/   # output directory from psychopy 
    │   ├── .csv   
    │   ├── .log  
    │   └── .psydat 
    ├── experiment.py   # Psychopy: visual and auditory experiments on Mac
    ├── functions.py   # function lists
    ├── visualstimuli_generator.py   # Generate the visual stimuli
    └── README.md  # Overview of the experiment directory
```


## Contact
If you have any questions, feel free to ask me!
 ```bash
kazuki@cbs.mpg.de
 ```



