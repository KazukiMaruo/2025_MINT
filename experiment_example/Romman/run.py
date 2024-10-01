#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 20 2024
@author: Roman Kessler, rkesslerx@gmail.com


TODOs:
    - check if dpi is low enough for short loading time on Argentinien
    - test the Argentinien2 functions (trigger etc)
    - stimulus trigger as well
    - save stimulus trigger code and stimulus file name in a table
MAYBE: 
    - save the new combination of stimulus parameters (incl sounds) to a file for later reference
Done
    - categories: manually find better categories that might be interesting to infants
    - pair each image with a version of the scrambled image
    - each image only 1x load in stimulus list (load all at once, and then get an image of those)
"""

# imports
import sys  
import os
from glob import glob
import timeit
import gc  # garbage collection
import datetime
import numpy as np
import random
import pandas as pd
import pickle
from shutil import rmtree
from psychopy import visual, core, data, event, logging, prefs, constants  # noqa
#prefs.hardware['audioLib'] = ['PTB', 'sounddevice', 'pyo', 'pygame']  # noqa 
prefs.hardware['audioLib'] = ['pygame'] # ''sounddevice''
from psychopy import sound, core
print('Using %s (with %s) for sounds' % (sound.audioLib, sound.audioDriver))
from psychopy.hardware import keyboard

global emulation

###### user definitions ###
emulation = False  # False
show_scrambled = False

#categories = ["face", "hand", "ball", "bike"] # TODO increase with the curated list of SHK
categories = [
    "cat",
    "face",
    "hand",
    "clothes",
    "bucket",
    "laptop",
    "apple",
    "banana",
    "chair",
    "spoon",
    "door",
    "flower",
    "rose",
    "ball",
    "car",
    "bus",
]



t_stim = 0.25
break_time = 0.10  # mean time of the break
background_sound_s = 0.03 # length in seconds of background sound
animal_every_nth = 20  # every n-th stimulus a random animal occurs

n_stimuli = len(categories)  # number of distinct stimuli
n_repetitions_per_stim = 60  # number of repetitions for each stimulus 
repetitions_per_subsession = 3  # within a sub-block, the stimuli are pseudo-randomized without restrictions
n_subsessions = 20  # must be an integer divisor of n_repetitions_per_stim / repetitions_per_subsession

animals = ["pig", "sheep", "chick", "parrot", "cat", "chicken", "horse", "mouse"]
# dog is too scary

###### FUNCTIONS ###


def shutdown(win, core):
    win.close()
    core.quit()

def get_keypress():
    keys = event.getKeys()
    if keys and keys[0] == 'Esc':
        shutdown()
    elif keys:
        return keys[0]
    else:
        return None

def garbage():
    collected = gc.collect()
    # Prints Garbage collector as 0 object
    #print("Garbage collector: collected",
    #      "%d objects." % collected)

def keyboard_input(win, core):
    # abort experiment if desired
    key = get_keypress()
    if key == 'q':  # esc
        shutdown(win, core)
    elif key == 'p':
        pause(win)

def pause(win):
    pause_text = visual.TextStim(
    win, text='wait for button press',
    font='', pos=(0,0),  #(-screen_size[0]/2+100, -screen_size[1]/2+30), 
    depth=0, color=(1.0, 1.0, 1.0),
    colorSpace='rgb', opacity=1.0, contrast=1.0, units='', ori=0.0,
    height=None, antialias=True, bold=False, italic=False,
    anchorHoriz='center', anchorVert='center', fontFiles=(),
    wrapWidth=None, flipHoriz=False, flipVert=False, name=None, autoLog=None)

    win.flip()
    pause_text.draw()
    win.flip()
    key = event.waitKeys(maxWait = np.inf)     # wait for a key press to continue
    if key == 'q':  # esc
        shutdown()
    #corner_text.draw()
    win.flip()

# create spaces breaks with catcher videos
def create_catch_breaks(n_trials, every_n, half_spacing=0.4): # after half_spacing of the experiment, the spacing is reduced
    break_trials = [i*every_n for i in range(n_trials) if i*every_n < n_trials]
    breaking_point = int(np.ceil(len(break_trials) * half_spacing))
    tighter_break_trials = []
    for i, point in enumerate(break_trials):
        tighter_break_trials.append(point)
        if i >= breaking_point:
            tighter_break_trials.append(int(point + every_n / 2))
    
    # fill it with the videos
    all_videos = ["1","2","3","4","5","6","7","11","12","13","14","15","16"]
    idxs = {}
    for point in tighter_break_trials[1:]: # new: skip the first one
        idxs[point] = random.sample(all_videos, 1)[0]
    
    return idxs

def is_close_to_any(number, number_list):
    for num in number_list:
        if abs(number - num) < 10:
            return True
    return False

def catchno(win, core, corner_text, no="1"):
    moviepath=f"movies{os.sep}{no}.mp4"
    catchmovie(win, moviepath, time=None)
    win.flip()


def catchmovie(win, moviepath, time=None):
    # keyboard to listen for keys
    kb = keyboard.Keyboard()
    
    mov = visual.MovieStim(  #visual.MovieStim3(  
        win,
        moviepath,    # path to video file
        size=(int(1920/2), int(1080/2)),  # (512, 512),
        flipVert=False,
        flipHoriz=False,
        loop=False, #True
        noAudio=False, #True,
        #volume=0.1,
        #autoStart=False
        )
    # main loop, exit when the status is finished
    while not mov.isFinished:
        # draw the movie
        mov.draw()
        # flip buffers so they appear on the window
        win.flip()

        # process keyboard input
        if kb.getKeys('q'):   # quit
            break
        elif kb.getKeys('r'):  # play/start
            mov.play()
        elif kb.getKeys('b'):  # pause
            mov.pause()
        elif kb.getKeys('s'):  # stop the movie
            mov.stop()

    # DEBUG
    mov.stop()
    # stop the movie, this frees resources too
    mov.unload()  # unloads when `mov.status == constants.FINISHED`

###### CREATE EXPERIMENT ENVIRONMENT ###
endExpNow = False  # flag for 'escape' or other condition => quit the exp

if not emulation:
    from psychopy import parallel # parallel
    from psychopy import gui

# set directory
if emulation: # DEBUG
    _thisDir = f"{os.sep}Users{os.sep}roman{os.sep}GitHub{os.sep}CRIME{os.sep}experiment{os.sep}"
else:
    _thisDir = os.path.dirname(os.path.abspath(__file__)) + os.sep
os.chdir(_thisDir)
sys.path.append(_thisDir)  # to be able to import custom functions

# set participant info
expName = f''  # from the Builder filename that created this script
expInfo = {'sub': '001', 'ses': '001'}

if not emulation:
    dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)    # GUI
    if not dlg.OK:
        core.quit()  # user pressed cancel
    p_port = parallel.ParallelPort(address='0x0378') #0x0378, 0x03BC
    p_port.setData(0)
else:
    expInfo['sub'] = expInfo['sub'] + "000"

expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

# logging (TODO also logfile?)
logging.console.setLevel(logging.CRITICAL)

# create experimental window
grayvalue = [0, 0, 0]
screen_size = (1024, 768)
if not emulation:  # todo: check is this fits to the new lab computer
    win = visual.Window(size=screen_size, fullscr=True, screen=0, allowGUI=False, allowStencil=False,
                        monitor='testMonitor', color=grayvalue, colorSpace='rgb',
                        winType='pyglet',  # new: should be faster
                        blendMode='avg', useFBO=True, units='pix',
                        )
else:  # on MacBook (built-in display)
    win = visual.Window(size=screen_size, fullscr=False,  # Better timing can be achieved in full-screen mode.
                        screen=0, allowGUI=True, allowStencil=False,
                        monitor='testMonitor', color=grayvalue, colorSpace='rgb',
                        winType='pyglet',  # new: should be faster
                        blendMode='avg', useFBO=True, units='pix',
                        )

# introduction
corner_text = visual.TextStim(
    win, text='',
    font='', 
    pos=(-screen_size[0]/2+30, -screen_size[1]/2+30), 
    depth=0, color=(1.0, 1.0, 1.0),
    colorSpace='rgb', opacity=1.0, contrast=1.0, units='', ori=0.0,
    height=None, antialias=True, bold=False, italic=False,
    anchorHoriz='center', anchorVert='center', fontFiles=(),
    wrapWidth=None, flipHoriz=False, flipVert=False, name=None, autoLog=None)
corner_text.size *= 2

# clocks and timers
clock_exp = core.Clock()  # clocks start at 0, experiment clock
logging.setDefaultClock(clock_exp)
stim_timer = core.CountdownTimer(t_stim)
break_timer = core.CountdownTimer(break_time)


######## CREATE STIMULI ETC ###

### catch video events
if show_scrambled == True:
    scrambled_multiplicator = 2
else:
    scrambled_multiplicator = 1

# check if reasonable numbers in header
if n_repetitions_per_stim / repetitions_per_subsession != n_subsessions: 
    print("Error: n_repetitions_per_stim / repetitions_per_subsession != n_subsessions")
    core.quit()
    win.close()
    
# make conditions etc
idxs = create_catch_breaks(n_trials=n_stimuli * n_repetitions_per_stim * scrambled_multiplicator, # 2 for scrambled 
                    every_n=90, 
                    half_spacing=0.4) # catch videos on fixed spacing

# DEBUG
print(idxs)
idxs[5] = "3"
idxs[8] = "3"
idxs[5] = "3"

### catch animals

# catcher animals
animal_sounds = []
animal_images = []
for i in animals:
    animal_sounds.append(
        sound.Sound(f"sounds{os.sep}animals{os.sep}{i}.wav", 
            secs=0.30, 
            volume = 1.0, # new, make this louder than the beeps
            hamming=True, 
            blockSize=512, 
            sampleRate=44100, # DEBUG
            name=f'animal_sound_{i}', 
            stereo = True, 
            autoLog=False)) 
    animal_images.append(
        visual.ImageStim(win=win, 
            name=f'animal_image_{i}', 
            units='pix',
            image=f"imgs{os.sep}animals{os.sep}{i}.png",
            mask=None,
            ori=0, pos=[0, 0], 
            size=[600, 600], 
            color=[1, 1, 1], colorSpace='rgb', opacity=1,
            flipHoriz=False, flipVert=False,
            texRes=128, interpolate=True, depth=0.0))


### condition list
n_trials = n_stimuli * n_repetitions_per_stim
conditions_intact = []
conditions_scrambled = []
for i in range(n_subsessions):
    conditions_subsession = []
    for j in range(repetitions_per_subsession):
        conditions_subsession += list(range(n_stimuli))
    conditions_subsessions_intact = conditions_subsession.copy()
    np.random.shuffle(conditions_subsessions_intact)
    conditions_intact.extend(conditions_subsessions_intact)
    if show_scrambled==True:
        conditions_subsessions_scrambled = conditions_subsession.copy()
        np.random.shuffle(conditions_subsessions_scrambled)
        conditions_scrambled.extend(conditions_subsessions_scrambled)

if show_scrambled==True:
    conditions = [item for pair in zip(conditions_intact, conditions_scrambled) for item in pair]
else:
    conditions = conditions_intact

### Create trial stimuli
TheStimulusList = {'intact': {},
                   'scrambled': {}}
TheStimulusList = {
    stimulus_type: {category: [] for category in categories}
    for stimulus_type in ['intact', 'scrambled']
}
maxStimPerCat = {'intact': {},
                   'scrambled': {}}
StimCatCounter = {'intact': {},
                   'scrambled': {}}
# for saving
#TriggerList = {'intact': {},
#               'scrambled': {}}
#TriggerList = {
#    stimulus_type: {category: [] for category in categories}
#    for stimulus_type in ['intact', 'scrambled']
#}

folders = {}
folders["intact"] = f"{_thisDir}{os.sep}imgs{os.sep}THINGS{os.sep}resized{os.sep}"
folders["scrambled"] = f"{_thisDir}{os.sep}imgs{os.sep}THINGS{os.sep}scrambled{os.sep}"
for stimulus_type in TheStimulusList.keys(): # intact, scrambled
    if (show_scrambled==False and stimulus_type=="scrambled"):
        continue
    for category in categories:
        # find all images of the category in the folder
        images = sorted(glob(f"{folders[stimulus_type]}{category}{os.sep}*.jpg"))
        # exclude manually some images that are not optimal
        images = [i for i in images if "EXCLUDE" not in i]
        maxStimPerCat[stimulus_type][category] = len(images)
        StimCatCounter[stimulus_type][category] = 0
        for i, image in enumerate(images):
            #TriggerList[stimulus_type][category].append(image)
            TheStimulusList[stimulus_type][category].append(
                # CREATE STIMULUS OBJECT - VISION
                visual.ImageStim(win=win, 
                                 name=f'{stimulus_type}_{category}_{i}', 
                                 units='pix',
                                 image=image, 
                                 mask=None,
                                 ori=0,  # orientation / rotation
                                 pos=[0, 0], 
                                 size=[800, 800],  #[768, 768],
                                 color=[1, 1, 1], colorSpace='rgb', opacity=1,
                                 flipHoriz=False, flipVert=False,
                                 texRes=128, 
                                 interpolate=False, # new: maybe False saves comp. time
                                 depth=0.0)
            )

# save trigger
# Flatten the dictionary into a list of records
#flattened_data = []
#for stimulus_type, items in TriggerList.items():
#    for category, thisList in items.items():
#        for i, image in enumerate(thisList):
#            record = {'stimulus_type': stimulus_type, 'category': category, 'image': os.path.basename(image), 'number': i+1}
#            flattened_data.append(record)

# Create the DataFrame
#df = pd.DataFrame(flattened_data)
#pd.set_option('display.max_columns', None)  # Show all columns
#df.head()


### sound stimuli
ticks = []
for i in range(400,850,50):
    ticks.append(sound.Sound(random.randint(400,800), # 600,1000 
                        volume=0.3, # new: less loud than the catchers
                        autoLog=False, # avoid Soundcard overflow on Argentinien computer
                        secs=background_sound_s, # time adjust by experiment
                        sampleRate=44100, #44100,
                        stereo=True)  # sample rate ignored because already set
    )
n_ticks = len(ticks)


###### Loop over trials

# wait for button press
pause(win)
for cond_counter, cond in enumerate(conditions):
    # if there is a regular catch video supposed to happen
    if (cond_counter in idxs.keys()) & (cond_counter < len(conditions) - 20):
        catchno(win, core, corner_text, no=idxs[cond_counter])
        break_timer.reset(t=1.)
        while break_timer.getTime() > 0:
            corner_text.draw()
            win.flip()
    
    # corner text
    corner_text.setText(f'{cond_counter+1}') 
    
    # animal catcher
    if (((cond_counter+1) % animal_every_nth) == 0) & (is_close_to_any(cond_counter+1, idxs.keys()) == False) & ((cond_counter+15) < len(conditions)): 
        # small break to not overlap with previous stimulus
        break_timer.reset(t=break_time + 0.7)
        while break_timer.getTime() > 0:
            corner_text.draw()
            keyboard_input(win, core)
            win.flip()
        # randomly select animal
        animal_i = np.random.randint(0, len(animals), size=None)
        if not emulation:
            win.callOnFlip(p_port.setData, 80+animal_i)
        # present animal sound
        animal_sounds[animal_i].play()
        # present animal image
        stim_timer.reset(t=t_stim + 1.5)
        while stim_timer.getTime() > 0:
            animal_images[animal_i].draw(win=win)  # debug to test autodraw
            corner_text.draw()
            keyboard_input(win, core)
            win.flip()
        # break
        break_timer.reset(t=break_time + 1.) 
        while break_timer.getTime() > 0:
            corner_text.draw()
            keyboard_input(win, core)
            win.flip()
    
    # check if "intact" or "scrambled"
    if ((cond_counter % 2) == 0 or (show_scrambled==False)): 
        version="intact"
    else: 
        version="scrambled"
    
    # send individual stimulus trigger (if needed later)
    if not emulation:
        win.callOnFlip(p_port.setData, 200 + StimCatCounter[version][categories[cond]])
    corner_text.draw()
    win.flip()
    if not emulation: 
        win.callOnFlip(p_port.setData, int(0))  # set all triggers to 0 after usage ->> is this really needed?
    win.flip()
    
    # send trial triggers
    if not emulation:
        if version=="intact":
            win.callOnFlip(p_port.setData, cond + 1)
        elif version=="scrambled":
            win.callOnFlip(p_port.setData, cond + 101)
    corner_text.draw()
    
    # show stimulus
    stim_timer.reset(t=t_stim)
    
    # start background sound
    which_tick = random.randint(0, n_ticks-1)
    ticks[which_tick].play() 
    
    # draw image stimulus
    while stim_timer.getTime() > 0:
        TheStimulusList[version][categories[cond]][StimCatCounter[version][categories[cond]]].draw(win=win)  # debug to test autodraw
        corner_text.draw()
        keyboard_input(win, core)
        win.flip()
    win.flip()
    if not emulation: # TODO: check if this position is right
        win.callOnFlip(p_port.setData, int(0))  # set all triggers to 0 after usage
    # increase the counter to show next one next time
    if StimCatCounter[version][categories[cond]] < (maxStimPerCat[version][categories[cond]] - 1):
        StimCatCounter[version][categories[cond]] += 1
    else: 
        StimCatCounter[version][categories[cond]] = 0

    # inter stimulus break
    break_timer.reset(t=break_time)
    while break_timer.getTime() > 0:
        corner_text.draw()
        keyboard_input(win, core)
        win.flip()
        
    # collect garbage
    garbage()

# short break
break_timer.reset(t=1.0)
while break_timer.getTime() > 0:
    corner_text.draw()
    keyboard_input(win, core)
    win.flip()

win.close()