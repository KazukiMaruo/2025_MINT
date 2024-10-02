#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.2.3),
    on September 12, 2024, at 15:42
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division
   
import psychopy
psychopy.useVersion('2021.2.3')


from psychopy import locale_setup
from psychopy import prefs
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2021.2.3'
expName = 'Replay_Things_v2021'  # from the Builder filename that created this script
expInfo = {'participant': '', 'trigger': '1', 'n_reps_localizer': '12', 'n_reps_animalbreak_localizer': '3', 'n_reps_videobreak_localizer': '5', 'n_reps_sequence': '5', 'n_reps_animalbreak_sequence': '2', 'n_reps_videobreak_sequence': '2', 'n_reps_rest_cues': '5', 'n_reps_sequence_rest': '2'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\\Users\\user\\Desktop\\REPLAY_Things\\Replay_Things_v2021.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1920, 1080], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Setup eyetracking
ioDevice = ioConfig = ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "init_vars"
init_varsClock = core.Clock()
from import_stimuli import create_cond_file_stim_pair

condition_file_localizer, condition_file_sequence, animal_catch, video_catch = create_cond_file_stim_pair(expInfo['participant'], int(expInfo['n_reps_localizer']), int(expInfo['n_reps_animalbreak_localizer'])*int(expInfo['n_reps_videobreak_localizer']), int(expInfo['n_reps_sequence']), int(expInfo['n_reps_sequence_rest'])*int(expInfo['n_reps_animalbreak_sequence'])*int(expInfo['n_reps_videobreak_sequence']))
selection_trial_localizer = np.arange(int(expInfo['n_reps_localizer']))
selection_trial_sequence = np.arange(int(expInfo['n_reps_sequence']))

# Display Position of Current Trial
current_trial_position = (-0.80, -0.45)

# Trigger Function
def get_current_trigger_value(t, values, timemap, duration):
    trigger_value = 0
    position_bool = (t >= timemap) * (t < (timemap + duration))
    if np.any(position_bool):
        trigger_value = values[position_bool][0]
    if isinstance(values[0],str):
        return trigger_value
    else:
        return int(trigger_value)

# Trigger Value Memory
trigger_old = 0

# Trigger Duration
trigger_duration = 0.040 # seconds

# Trigger Time Mappings
trigger_timemap_localizer = np.array([0,0.08,0.16])-frameTolerance # seconds
trigger_timemap_sequence = np.array([0,0.08,0.16,0.35,0.43,0.51,0.70,0.78,0.86])-frameTolerance # seconds
trigger_timemap_resting = np.array([0,0.08])-frameTolerance # seconds

# Initialize Parallel Port
if int(expInfo['trigger']) == 1:
    from psychopy import parallel
    p_port = parallel.ParallelPort(address='0x0378')


tone_dict = dict()
for i in [240, 440, 640]:
    tone_dict[i] = sound.Sound(i, # 600,1000 
                               volume=0.2, # new: less loud than the catchers
                               autoLog=False, # avoid Soundcard overflow on Argentinien computer
                               secs=0.25, # time adjust by experiment
                               sampleRate=44100, stereo=True # sample rate ignored because already set
                               )

# Initialize Videos
movie_dict = dict()
for path in video_catch:
    movie_dict[path] = visual.MovieStim3(
                                        win=win, name=path,
                                        noAudio = False,
                                        filename=path,
                                        volume=0.2,
                                        ori=0.0, pos=(0, 0), opacity=None,
                                        loop=False,
                                        depth=-2.0,
                                    )
    movie_dict[path].setAutoDraw(False)

text_init_vars_blank = visual.TextStim(win=win, name='text_init_vars_blank',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=0.0, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "start_localizer"
start_localizerClock = core.Clock()
text_start_localizer = visual.TextStim(win=win, name='text_start_localizer',
    text='Start Localizer with SPACE!',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
keyb_start_localizer = keyboard.Keyboard()

# Initialize components for Routine "trigger_start_localizer"
trigger_start_localizerClock = core.Clock()
text_trigger_start_localizer_blank = visual.TextStim(win=win, name='text_trigger_start_localizer_blank',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=0.0, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "rvsp_localizer"
rvsp_localizerClock = core.Clock()
current_trial_localizer = 0
n_reps_localizer = expInfo['n_reps_localizer']
n_reps_animalbreak_localizer = expInfo['n_reps_animalbreak_localizer']
n_reps_videobreak_localizer = expInfo['n_reps_videobreak_localizer']


image_localizer = visual.ImageStim(
    win=win,
    name='image_localizer', 
    image='sin', mask=None,
    ori=0.0, pos=(0, 0), size=(0.7, 0.7),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
text_current_trial_rsvp_localizer = visual.TextStim(win=win, name='text_current_trial_rsvp_localizer',
    text='',
    font='Open Sans',
    pos=current_trial_position, height=0.03, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "attentioncatch_animal_localizer"
attentioncatch_animal_localizerClock = core.Clock()
image_animal_localizer = visual.ImageStim(
    win=win,
    name='image_animal_localizer', 
    image='sin', mask=None,
    ori=0.0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
sound_1 = sound.Sound('A', secs=-1, stereo=True, hamming=True,
    name='sound_1')
sound_1.setVolume(0.2)
text_current_trial_animal_localizer = visual.TextStim(win=win, name='text_current_trial_animal_localizer',
    text='',
    font='Open Sans',
    pos=current_trial_position, height=0.03, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);

# Initialize components for Routine "attentioncatch_video_localizer"
attentioncatch_video_localizerClock = core.Clock()
text_current_trial_video_localizer = visual.TextStim(win=win, name='text_current_trial_video_localizer',
    text='',
    font='Open Sans',
    pos=current_trial_position, height=0.03, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "trigger_stop_localizer"
trigger_stop_localizerClock = core.Clock()
text_trigger_stop_localizer_blank = visual.TextStim(win=win, name='text_trigger_stop_localizer_blank',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=0.0, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "start_sequence"
start_sequenceClock = core.Clock()
text_start_sequence = visual.TextStim(win=win, name='text_start_sequence',
    text='Start Sequence Learning and Resting Phase with SPACE!',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
keyb_start_sequence = keyboard.Keyboard()

# Initialize components for Routine "trigger_start_sequence"
trigger_start_sequenceClock = core.Clock()
text_trigger_start_sequence_blank = visual.TextStim(win=win, name='text_trigger_start_sequence_blank',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=0.0, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "rvsp_sequence"
rvsp_sequenceClock = core.Clock()
current_trial_sequence = 0
n_reps_sequence = expInfo['n_reps_sequence']
n_reps_animalbreak_sequence = expInfo['n_reps_animalbreak_sequence']
n_reps_videobreak_sequence = expInfo['n_reps_videobreak_sequence']
n_reps_sequence_rest = expInfo['n_reps_sequence_rest']

image_1_sequence = visual.ImageStim(
    win=win,
    name='image_1_sequence', 
    image='sin', mask=None,
    ori=0.0, pos=(0, 0), size=(0.7, 0.7),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
image_2_sequence = visual.ImageStim(
    win=win,
    name='image_2_sequence', 
    image='sin', mask=None,
    ori=0.0, pos=(0, 0), size=(0.7, 0.7),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
image_3_sequence = visual.ImageStim(
    win=win,
    name='image_3_sequence', 
    image='sin', mask=None,
    ori=0.0, pos=(0, 0), size=(0.7, 0.7),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-3.0)
text_current_trial_rsvp_sequence = visual.TextStim(win=win, name='text_current_trial_rsvp_sequence',
    text='',
    font='Open Sans',
    pos=current_trial_position, height=0.03, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-4.0);

# Initialize components for Routine "iti_sequence"
iti_sequenceClock = core.Clock()
text_current_trial_iti_sequence = visual.TextStim(win=win, name='text_current_trial_iti_sequence',
    text='',
    font='Open Sans',
    pos=current_trial_position, height=0.03, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "attentioncatch_animal_sequence"
attentioncatch_animal_sequenceClock = core.Clock()
image_animal_sequence = visual.ImageStim(
    win=win,
    name='image_animal_sequence', 
    image='sin', mask=None,
    ori=0.0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
sound_animal_localizer = sound.Sound('A', secs=-1, stereo=True, hamming=True,
    name='sound_animal_localizer')
sound_animal_localizer.setVolume(1.0)
text_current_trial_animal_sequence = visual.TextStim(win=win, name='text_current_trial_animal_sequence',
    text='',
    font='Open Sans',
    pos=current_trial_position, height=0.03, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);

# Initialize components for Routine "attentioncatch_video_sequence"
attentioncatch_video_sequenceClock = core.Clock()
text_current_trial_video_sequence = visual.TextStim(win=win, name='text_current_trial_video_sequence',
    text='',
    font='Open Sans',
    pos=current_trial_position, height=0.03, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "trigger_stop_sequence"
trigger_stop_sequenceClock = core.Clock()
text_trigger_stop_sequence_blank = visual.TextStim(win=win, name='text_trigger_stop_sequence_blank',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=0.0, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "start_resting"
start_restingClock = core.Clock()
text_start_resting = visual.TextStim(win=win, name='text_start_resting',
    text='Resting Phase starts...',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "blank_resting"
blank_restingClock = core.Clock()
current_trial_resting = 0
n_reps_rest_cues = expInfo['n_reps_rest_cues']
text_current_trial_blank_resting = visual.TextStim(win=win, name='text_current_trial_blank_resting',
    text='',
    font='Open Sans',
    pos=current_trial_position, height=0.03, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "cue_resting"
cue_restingClock = core.Clock()
text_current_trial_cue_resting = visual.TextStim(win=win, name='text_current_trial_cue_resting',
    text='',
    font='Open Sans',
    pos=current_trial_position, height=0.03, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "stop_resting"
stop_restingClock = core.Clock()
text_end_resting = visual.TextStim(win=win, name='text_end_resting',
    text='Resting Phase ends...',
    font='Open Sans',
    pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "end_screen"
end_screenClock = core.Clock()
text_end_screen = visual.TextStim(win=win, name='text_end_screen',
    text='The experiment is now finished.\nThank you for your participation! :)',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
keyb_end_screen = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "init_vars"-------
continueRoutine = True
routineTimer.add(3.000000)
# update component parameters for each repeat
# keep track of which components have finished
init_varsComponents = [text_init_vars_blank]
for thisComponent in init_varsComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
init_varsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "init_vars"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = init_varsClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=init_varsClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_init_vars_blank* updates
    if text_init_vars_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_init_vars_blank.frameNStart = frameN  # exact frame index
        text_init_vars_blank.tStart = t  # local t and not account for scr refresh
        text_init_vars_blank.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_init_vars_blank, 'tStartRefresh')  # time at next scr refresh
        text_init_vars_blank.setAutoDraw(True)
    if text_init_vars_blank.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text_init_vars_blank.tStartRefresh + 3.0-frameTolerance:
            # keep track of stop time/frame for later
            text_init_vars_blank.tStop = t  # not accounting for scr refresh
            text_init_vars_blank.frameNStop = frameN  # exact frame index
            win.timeOnFlip(text_init_vars_blank, 'tStopRefresh')  # time at next scr refresh
            text_init_vars_blank.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in init_varsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "init_vars"-------
for thisComponent in init_varsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_init_vars_blank.started', text_init_vars_blank.tStartRefresh)
thisExp.addData('text_init_vars_blank.stopped', text_init_vars_blank.tStopRefresh)

# ------Prepare to start Routine "start_localizer"-------
continueRoutine = True
# update component parameters for each repeat
keyb_start_localizer.keys = []
keyb_start_localizer.rt = []
_keyb_start_localizer_allKeys = []
# keep track of which components have finished
start_localizerComponents = [text_start_localizer, keyb_start_localizer]
for thisComponent in start_localizerComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
start_localizerClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "start_localizer"-------
while continueRoutine:
    # get current time
    t = start_localizerClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=start_localizerClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_start_localizer* updates
    if text_start_localizer.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_start_localizer.frameNStart = frameN  # exact frame index
        text_start_localizer.tStart = t  # local t and not account for scr refresh
        text_start_localizer.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_start_localizer, 'tStartRefresh')  # time at next scr refresh
        text_start_localizer.setAutoDraw(True)
    
    # *keyb_start_localizer* updates
    waitOnFlip = False
    if keyb_start_localizer.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        keyb_start_localizer.frameNStart = frameN  # exact frame index
        keyb_start_localizer.tStart = t  # local t and not account for scr refresh
        keyb_start_localizer.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(keyb_start_localizer, 'tStartRefresh')  # time at next scr refresh
        keyb_start_localizer.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(keyb_start_localizer.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(keyb_start_localizer.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if keyb_start_localizer.status == STARTED and not waitOnFlip:
        theseKeys = keyb_start_localizer.getKeys(keyList=['space'], waitRelease=False)
        _keyb_start_localizer_allKeys.extend(theseKeys)
        if len(_keyb_start_localizer_allKeys):
            keyb_start_localizer.keys = _keyb_start_localizer_allKeys[-1].name  # just the last key pressed
            keyb_start_localizer.rt = _keyb_start_localizer_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in start_localizerComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "start_localizer"-------
for thisComponent in start_localizerComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_start_localizer.started', text_start_localizer.tStartRefresh)
thisExp.addData('text_start_localizer.stopped', text_start_localizer.tStopRefresh)
# check responses
if keyb_start_localizer.keys in ['', [], None]:  # No response was made
    keyb_start_localizer.keys = None
thisExp.addData('keyb_start_localizer.keys',keyb_start_localizer.keys)
if keyb_start_localizer.keys != None:  # we had a response
    thisExp.addData('keyb_start_localizer.rt', keyb_start_localizer.rt)
thisExp.addData('keyb_start_localizer.started', keyb_start_localizer.tStartRefresh)
thisExp.addData('keyb_start_localizer.stopped', keyb_start_localizer.tStopRefresh)
thisExp.nextEntry()
# the Routine "start_localizer" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "trigger_start_localizer"-------
continueRoutine = True
routineTimer.add(3.000000)
# update component parameters for each repeat
# keep track of which components have finished
trigger_start_localizerComponents = [text_trigger_start_localizer_blank]
for thisComponent in trigger_start_localizerComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
trigger_start_localizerClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "trigger_start_localizer"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = trigger_start_localizerClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=trigger_start_localizerClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    if int(expInfo['trigger']) == 1:
        trigger_current = get_current_trigger_value(tThisFlip, np.array([20]), np.array([0.0]), trigger_duration)
        if trigger_current != trigger_old:
            win.callOnFlip(p_port.setData, trigger_current)
        trigger_old = trigger_current
    
    if int(expInfo['trigger']) == 2:
        trigger_current = get_current_trigger_value(tThisFlip, np.array(['Start_Localizer']), np.array([0.0]), trigger_duration)
        if trigger_current != trigger_old:
            print(f'Localizer  ---  Time: {1000*tThisFlip:5.2f}  ---  Trigger Value: {trigger_current}')
        trigger_old = trigger_current
    
    # *text_trigger_start_localizer_blank* updates
    if text_trigger_start_localizer_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_trigger_start_localizer_blank.frameNStart = frameN  # exact frame index
        text_trigger_start_localizer_blank.tStart = t  # local t and not account for scr refresh
        text_trigger_start_localizer_blank.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_trigger_start_localizer_blank, 'tStartRefresh')  # time at next scr refresh
        text_trigger_start_localizer_blank.setAutoDraw(True)
    if text_trigger_start_localizer_blank.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text_trigger_start_localizer_blank.tStartRefresh + 3.0-frameTolerance:
            # keep track of stop time/frame for later
            text_trigger_start_localizer_blank.tStop = t  # not accounting for scr refresh
            text_trigger_start_localizer_blank.frameNStop = frameN  # exact frame index
            win.timeOnFlip(text_trigger_start_localizer_blank, 'tStopRefresh')  # time at next scr refresh
            text_trigger_start_localizer_blank.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in trigger_start_localizerComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "trigger_start_localizer"-------
for thisComponent in trigger_start_localizerComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_trigger_start_localizer_blank.started', text_trigger_start_localizer_blank.tStartRefresh)
thisExp.addData('text_trigger_start_localizer_blank.stopped', text_trigger_start_localizer_blank.tStopRefresh)

# set up handler to look after randomisation of conditions etc
loop_videobreak_localizer = data.TrialHandler(nReps=n_reps_videobreak_localizer, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='loop_videobreak_localizer')
thisExp.addLoop(loop_videobreak_localizer)  # add the loop to the experiment
thisLoop_videobreak_localizer = loop_videobreak_localizer.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisLoop_videobreak_localizer.rgb)
if thisLoop_videobreak_localizer != None:
    for paramName in thisLoop_videobreak_localizer:
        exec('{} = thisLoop_videobreak_localizer[paramName]'.format(paramName))

for thisLoop_videobreak_localizer in loop_videobreak_localizer:
    currentLoop = loop_videobreak_localizer
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_videobreak_localizer.rgb)
    if thisLoop_videobreak_localizer != None:
        for paramName in thisLoop_videobreak_localizer:
            exec('{} = thisLoop_videobreak_localizer[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    loop_animalbreak_localizer = data.TrialHandler(nReps=n_reps_animalbreak_localizer, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='loop_animalbreak_localizer')
    thisExp.addLoop(loop_animalbreak_localizer)  # add the loop to the experiment
    thisLoop_animalbreak_localizer = loop_animalbreak_localizer.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_animalbreak_localizer.rgb)
    if thisLoop_animalbreak_localizer != None:
        for paramName in thisLoop_animalbreak_localizer:
            exec('{} = thisLoop_animalbreak_localizer[paramName]'.format(paramName))
    
    for thisLoop_animalbreak_localizer in loop_animalbreak_localizer:
        currentLoop = loop_animalbreak_localizer
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_animalbreak_localizer.rgb)
        if thisLoop_animalbreak_localizer != None:
            for paramName in thisLoop_animalbreak_localizer:
                exec('{} = thisLoop_animalbreak_localizer[paramName]'.format(paramName))
        
        # set up handler to look after randomisation of conditions etc
        loop_rvsp_localizer = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions(condition_file_localizer, selection=selection_trial_localizer.tolist()),
            seed=None, name='loop_rvsp_localizer')
        thisExp.addLoop(loop_rvsp_localizer)  # add the loop to the experiment
        thisLoop_rvsp_localizer = loop_rvsp_localizer.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_rvsp_localizer.rgb)
        if thisLoop_rvsp_localizer != None:
            for paramName in thisLoop_rvsp_localizer:
                exec('{} = thisLoop_rvsp_localizer[paramName]'.format(paramName))
        
        for thisLoop_rvsp_localizer in loop_rvsp_localizer:
            currentLoop = loop_rvsp_localizer
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_rvsp_localizer.rgb)
            if thisLoop_rvsp_localizer != None:
                for paramName in thisLoop_rvsp_localizer:
                    exec('{} = thisLoop_rvsp_localizer[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "rvsp_localizer"-------
            continueRoutine = True
            routineTimer.add(0.350000)
            # update component parameters for each repeat
            current_trial_localizer += 1
            if int(expInfo['trigger']) == 1:
                trigger_values_localizer = np.array([21, int(Loc_Trigger_Value_Image), int(Loc_Trigger_Value_Sound)])
            if int(expInfo['trigger']) == 2:
                trigger_values_localizer = np.array(['Onset', 'Img', 'Aud'])
            
            play_tone = True
            image_localizer.setImage(Loc_Image)
            text_current_trial_rsvp_localizer.setText(current_trial_localizer)
            # keep track of which components have finished
            rvsp_localizerComponents = [image_localizer, text_current_trial_rsvp_localizer]
            for thisComponent in rvsp_localizerComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            rvsp_localizerClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "rvsp_localizer"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = rvsp_localizerClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=rvsp_localizerClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                if int(expInfo['trigger']) == 1:
                    trigger_current = get_current_trigger_value(tThisFlip, trigger_values_localizer, trigger_timemap_localizer, trigger_duration)
                    if trigger_current != trigger_old:
                        win.callOnFlip(p_port.setData, trigger_current)
                    trigger_old = trigger_current
                
                if int(expInfo['trigger']) == 2:
                    trigger_current = get_current_trigger_value(tThisFlip, trigger_values_localizer, trigger_timemap_localizer, trigger_duration)
                    if trigger_current != trigger_old:
                        print(f'Localizer  ---  Time: {1000*tThisFlip:5.2f}  ---  Trigger Value: {trigger_current}')
                    trigger_old = trigger_current
                
                if play_tone:
                    tone_dict[Loc_Sound].play()
                    play_tone = False
                
                
                
                # *image_localizer* updates
                if image_localizer.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_localizer.frameNStart = frameN  # exact frame index
                    image_localizer.tStart = t  # local t and not account for scr refresh
                    image_localizer.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_localizer, 'tStartRefresh')  # time at next scr refresh
                    image_localizer.setAutoDraw(True)
                if image_localizer.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_localizer.tStartRefresh + 0.250-frameTolerance:
                        # keep track of stop time/frame for later
                        image_localizer.tStop = t  # not accounting for scr refresh
                        image_localizer.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(image_localizer, 'tStopRefresh')  # time at next scr refresh
                        image_localizer.setAutoDraw(False)
                
                # *text_current_trial_rsvp_localizer* updates
                if text_current_trial_rsvp_localizer.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_current_trial_rsvp_localizer.frameNStart = frameN  # exact frame index
                    text_current_trial_rsvp_localizer.tStart = t  # local t and not account for scr refresh
                    text_current_trial_rsvp_localizer.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_current_trial_rsvp_localizer, 'tStartRefresh')  # time at next scr refresh
                    text_current_trial_rsvp_localizer.setAutoDraw(True)
                if text_current_trial_rsvp_localizer.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_current_trial_rsvp_localizer.tStartRefresh + 0.350-frameTolerance:
                        # keep track of stop time/frame for later
                        text_current_trial_rsvp_localizer.tStop = t  # not accounting for scr refresh
                        text_current_trial_rsvp_localizer.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(text_current_trial_rsvp_localizer, 'tStopRefresh')  # time at next scr refresh
                        text_current_trial_rsvp_localizer.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in rvsp_localizerComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "rvsp_localizer"-------
            for thisComponent in rvsp_localizerComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            loop_rvsp_localizer.addData('image_localizer.started', image_localizer.tStartRefresh)
            loop_rvsp_localizer.addData('image_localizer.stopped', image_localizer.tStopRefresh)
            loop_rvsp_localizer.addData('text_current_trial_rsvp_localizer.started', text_current_trial_rsvp_localizer.tStartRefresh)
            loop_rvsp_localizer.addData('text_current_trial_rsvp_localizer.stopped', text_current_trial_rsvp_localizer.tStopRefresh)
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'loop_rvsp_localizer'
        
        
        # ------Prepare to start Routine "attentioncatch_animal_localizer"-------
        continueRoutine = True
        routineTimer.add(5.000000)
        # update component parameters for each repeat
        selection_trial_localizer += selection_trial_localizer.shape[0]
        rnd_idx = np.random.randint(animal_catch.shape[0])
        image_animal_localizer.setImage(animal_catch[rnd_idx,0])
        sound_1.setSound(animal_catch[rnd_idx,1], secs=3.0, hamming=True)
        sound_1.setVolume(0.2, log=False)
        text_current_trial_animal_localizer.setText(current_trial_localizer)
        # keep track of which components have finished
        attentioncatch_animal_localizerComponents = [image_animal_localizer, sound_1, text_current_trial_animal_localizer]
        for thisComponent in attentioncatch_animal_localizerComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        attentioncatch_animal_localizerClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "attentioncatch_animal_localizer"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = attentioncatch_animal_localizerClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=attentioncatch_animal_localizerClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Send Trigger
            if int(expInfo['trigger']) == 1:
                trigger_current = get_current_trigger_value(tThisFlip, np.array([240, 249]), np.array([1.0, 4.0]), trigger_duration)
                if trigger_current != trigger_old:
                    win.callOnFlip(p_port.setData, trigger_current)
                trigger_old = trigger_current
            
            if int(expInfo['trigger']) == 2:
                trigger_current = get_current_trigger_value(tThisFlip, np.array(['AnimalBreak_Start','AnimalBreak_Stop']), np.array([0, 4.9]), trigger_duration)
                if trigger_current != trigger_old:
                    print(f'Localizer  ---  Time: {1000*tThisFlip:5.2f}  ---  Trigger Value: {trigger_current}')
                trigger_old = trigger_current
            
            
            # *image_animal_localizer* updates
            if image_animal_localizer.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                # keep track of start time/frame for later
                image_animal_localizer.frameNStart = frameN  # exact frame index
                image_animal_localizer.tStart = t  # local t and not account for scr refresh
                image_animal_localizer.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_animal_localizer, 'tStartRefresh')  # time at next scr refresh
                image_animal_localizer.setAutoDraw(True)
            if image_animal_localizer.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_animal_localizer.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    image_animal_localizer.tStop = t  # not accounting for scr refresh
                    image_animal_localizer.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(image_animal_localizer, 'tStopRefresh')  # time at next scr refresh
                    image_animal_localizer.setAutoDraw(False)
            # start/stop sound_1
            if sound_1.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                # keep track of start time/frame for later
                sound_1.frameNStart = frameN  # exact frame index
                sound_1.tStart = t  # local t and not account for scr refresh
                sound_1.tStartRefresh = tThisFlipGlobal  # on global time
                sound_1.play(when=win)  # sync with win flip
            if sound_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_1.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_1.tStop = t  # not accounting for scr refresh
                    sound_1.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(sound_1, 'tStopRefresh')  # time at next scr refresh
                    sound_1.stop()
            
            # *text_current_trial_animal_localizer* updates
            if text_current_trial_animal_localizer.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_current_trial_animal_localizer.frameNStart = frameN  # exact frame index
                text_current_trial_animal_localizer.tStart = t  # local t and not account for scr refresh
                text_current_trial_animal_localizer.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_current_trial_animal_localizer, 'tStartRefresh')  # time at next scr refresh
                text_current_trial_animal_localizer.setAutoDraw(True)
            if text_current_trial_animal_localizer.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_current_trial_animal_localizer.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_current_trial_animal_localizer.tStop = t  # not accounting for scr refresh
                    text_current_trial_animal_localizer.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(text_current_trial_animal_localizer, 'tStopRefresh')  # time at next scr refresh
                    text_current_trial_animal_localizer.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in attentioncatch_animal_localizerComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "attentioncatch_animal_localizer"-------
        for thisComponent in attentioncatch_animal_localizerComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        loop_animalbreak_localizer.addData('image_animal_localizer.started', image_animal_localizer.tStartRefresh)
        loop_animalbreak_localizer.addData('image_animal_localizer.stopped', image_animal_localizer.tStopRefresh)
        sound_1.stop()  # ensure sound has stopped at end of routine
        loop_animalbreak_localizer.addData('sound_1.started', sound_1.tStartRefresh)
        loop_animalbreak_localizer.addData('sound_1.stopped', sound_1.tStopRefresh)
        loop_animalbreak_localizer.addData('text_current_trial_animal_localizer.started', text_current_trial_animal_localizer.tStartRefresh)
        loop_animalbreak_localizer.addData('text_current_trial_animal_localizer.stopped', text_current_trial_animal_localizer.tStopRefresh)
        thisExp.nextEntry()
        
    # completed n_reps_animalbreak_localizer repeats of 'loop_animalbreak_localizer'
    
    
    # ------Prepare to start Routine "attentioncatch_video_localizer"-------
    continueRoutine = True
    routineTimer.add(8.000000)
    # update component parameters for each repeat
    rnd_idx = randint(video_catch.shape[0])
    
    # Start Video
    start_video = True
    #movie_dict[video_catch[rnd_idx]].setAutoDraw(True)
    text_current_trial_video_localizer.setText(current_trial_localizer)
    # keep track of which components have finished
    attentioncatch_video_localizerComponents = [text_current_trial_video_localizer]
    for thisComponent in attentioncatch_video_localizerComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    attentioncatch_video_localizerClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "attentioncatch_video_localizer"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = attentioncatch_video_localizerClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=attentioncatch_video_localizerClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Send Trigger
        if int(expInfo['trigger']) == 1:
            trigger_current = get_current_trigger_value(tThisFlip, np.array([230,239]), np.array([1.0,7.0]), trigger_duration)
            if trigger_current != trigger_old:
                win.callOnFlip(p_port.setData, trigger_current)
            trigger_old = trigger_current
        
        if int(expInfo['trigger']) == 2:
            trigger_current = get_current_trigger_value(tThisFlip, np.array(['VideoBreak_Start','VideoBreak_Stop']), np.array([0,4.9]), trigger_duration)
            if trigger_current != trigger_old:
                print(f'Localizer  ---  Time: {1000*tThisFlip:5.2f}  ---  Trigger Value: {trigger_current}')
            trigger_old = trigger_current
        
        if start_video and (tThisFlip >= 1.0-frameTolerance):
            movie_dict[video_catch[rnd_idx]].setAutoDraw(True)
            start_video = False
        
        if not start_video and (tThisFlip >= 7.0-frameTolerance):
            movie_dict[video_catch[rnd_idx]].setAutoDraw(False)
        
        # *text_current_trial_video_localizer* updates
        if text_current_trial_video_localizer.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_current_trial_video_localizer.frameNStart = frameN  # exact frame index
            text_current_trial_video_localizer.tStart = t  # local t and not account for scr refresh
            text_current_trial_video_localizer.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_current_trial_video_localizer, 'tStartRefresh')  # time at next scr refresh
            text_current_trial_video_localizer.setAutoDraw(True)
        if text_current_trial_video_localizer.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_current_trial_video_localizer.tStartRefresh + 8.0-frameTolerance:
                # keep track of stop time/frame for later
                text_current_trial_video_localizer.tStop = t  # not accounting for scr refresh
                text_current_trial_video_localizer.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_current_trial_video_localizer, 'tStopRefresh')  # time at next scr refresh
                text_current_trial_video_localizer.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in attentioncatch_video_localizerComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "attentioncatch_video_localizer"-------
    for thisComponent in attentioncatch_video_localizerComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # Stop Video
    #movie_dict[video_catch[rnd_idx]].setAutoDraw(False)
    loop_videobreak_localizer.addData('text_current_trial_video_localizer.started', text_current_trial_video_localizer.tStartRefresh)
    loop_videobreak_localizer.addData('text_current_trial_video_localizer.stopped', text_current_trial_video_localizer.tStopRefresh)
    thisExp.nextEntry()
    
# completed n_reps_videobreak_localizer repeats of 'loop_videobreak_localizer'


# ------Prepare to start Routine "trigger_stop_localizer"-------
continueRoutine = True
routineTimer.add(4.000000)
# update component parameters for each repeat
# keep track of which components have finished
trigger_stop_localizerComponents = [text_trigger_stop_localizer_blank]
for thisComponent in trigger_stop_localizerComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
trigger_stop_localizerClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "trigger_stop_localizer"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = trigger_stop_localizerClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=trigger_stop_localizerClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    if int(expInfo['trigger']) == 1:
        trigger_current = get_current_trigger_value(tThisFlip, np.array([29]), np.array([3.0]), trigger_duration)
        if trigger_current != trigger_old:
            win.callOnFlip(p_port.setData, trigger_current)
        trigger_old = trigger_current
    
    if int(expInfo['trigger']) == 2:
        trigger_current = get_current_trigger_value(tThisFlip, np.array(['Stop_Localizer']), np.array([3.0]), trigger_duration)
        if trigger_current != trigger_old:
            print(f'Localizer  ---  Time: {1000*tThisFlip:5.2f}  ---  Trigger Value: {trigger_current}')
        trigger_old = trigger_current
    
    # *text_trigger_stop_localizer_blank* updates
    if text_trigger_stop_localizer_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_trigger_stop_localizer_blank.frameNStart = frameN  # exact frame index
        text_trigger_stop_localizer_blank.tStart = t  # local t and not account for scr refresh
        text_trigger_stop_localizer_blank.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_trigger_stop_localizer_blank, 'tStartRefresh')  # time at next scr refresh
        text_trigger_stop_localizer_blank.setAutoDraw(True)
    if text_trigger_stop_localizer_blank.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text_trigger_stop_localizer_blank.tStartRefresh + 4.0-frameTolerance:
            # keep track of stop time/frame for later
            text_trigger_stop_localizer_blank.tStop = t  # not accounting for scr refresh
            text_trigger_stop_localizer_blank.frameNStop = frameN  # exact frame index
            win.timeOnFlip(text_trigger_stop_localizer_blank, 'tStopRefresh')  # time at next scr refresh
            text_trigger_stop_localizer_blank.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in trigger_stop_localizerComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "trigger_stop_localizer"-------
for thisComponent in trigger_stop_localizerComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_trigger_stop_localizer_blank.started', text_trigger_stop_localizer_blank.tStartRefresh)
thisExp.addData('text_trigger_stop_localizer_blank.stopped', text_trigger_stop_localizer_blank.tStopRefresh)

# ------Prepare to start Routine "start_sequence"-------
continueRoutine = True
# update component parameters for each repeat
keyb_start_sequence.keys = []
keyb_start_sequence.rt = []
_keyb_start_sequence_allKeys = []
# keep track of which components have finished
start_sequenceComponents = [text_start_sequence, keyb_start_sequence]
for thisComponent in start_sequenceComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
start_sequenceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "start_sequence"-------
while continueRoutine:
    # get current time
    t = start_sequenceClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=start_sequenceClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_start_sequence* updates
    if text_start_sequence.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_start_sequence.frameNStart = frameN  # exact frame index
        text_start_sequence.tStart = t  # local t and not account for scr refresh
        text_start_sequence.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_start_sequence, 'tStartRefresh')  # time at next scr refresh
        text_start_sequence.setAutoDraw(True)
    
    # *keyb_start_sequence* updates
    waitOnFlip = False
    if keyb_start_sequence.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        keyb_start_sequence.frameNStart = frameN  # exact frame index
        keyb_start_sequence.tStart = t  # local t and not account for scr refresh
        keyb_start_sequence.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(keyb_start_sequence, 'tStartRefresh')  # time at next scr refresh
        keyb_start_sequence.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(keyb_start_sequence.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(keyb_start_sequence.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if keyb_start_sequence.status == STARTED and not waitOnFlip:
        theseKeys = keyb_start_sequence.getKeys(keyList=['space'], waitRelease=False)
        _keyb_start_sequence_allKeys.extend(theseKeys)
        if len(_keyb_start_sequence_allKeys):
            keyb_start_sequence.keys = _keyb_start_sequence_allKeys[-1].name  # just the last key pressed
            keyb_start_sequence.rt = _keyb_start_sequence_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in start_sequenceComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "start_sequence"-------
for thisComponent in start_sequenceComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_start_sequence.started', text_start_sequence.tStartRefresh)
thisExp.addData('text_start_sequence.stopped', text_start_sequence.tStopRefresh)
# check responses
if keyb_start_sequence.keys in ['', [], None]:  # No response was made
    keyb_start_sequence.keys = None
thisExp.addData('keyb_start_sequence.keys',keyb_start_sequence.keys)
if keyb_start_sequence.keys != None:  # we had a response
    thisExp.addData('keyb_start_sequence.rt', keyb_start_sequence.rt)
thisExp.addData('keyb_start_sequence.started', keyb_start_sequence.tStartRefresh)
thisExp.addData('keyb_start_sequence.stopped', keyb_start_sequence.tStopRefresh)
thisExp.nextEntry()
# the Routine "start_sequence" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
loop_repeat_sequence_rest = data.TrialHandler(nReps=n_reps_sequence_rest, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='loop_repeat_sequence_rest')
thisExp.addLoop(loop_repeat_sequence_rest)  # add the loop to the experiment
thisLoop_repeat_sequence_rest = loop_repeat_sequence_rest.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisLoop_repeat_sequence_rest.rgb)
if thisLoop_repeat_sequence_rest != None:
    for paramName in thisLoop_repeat_sequence_rest:
        exec('{} = thisLoop_repeat_sequence_rest[paramName]'.format(paramName))

for thisLoop_repeat_sequence_rest in loop_repeat_sequence_rest:
    currentLoop = loop_repeat_sequence_rest
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_repeat_sequence_rest.rgb)
    if thisLoop_repeat_sequence_rest != None:
        for paramName in thisLoop_repeat_sequence_rest:
            exec('{} = thisLoop_repeat_sequence_rest[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "trigger_start_sequence"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    # keep track of which components have finished
    trigger_start_sequenceComponents = [text_trigger_start_sequence_blank]
    for thisComponent in trigger_start_sequenceComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trigger_start_sequenceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "trigger_start_sequence"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = trigger_start_sequenceClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trigger_start_sequenceClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        if int(expInfo['trigger']) == 1:
            trigger_current = get_current_trigger_value(tThisFlip, np.array([40]), np.array([0.0]), trigger_duration)
            if trigger_current != trigger_old:
                win.callOnFlip(p_port.setData, trigger_current)
            trigger_old = trigger_current
        
        if int(expInfo['trigger']) == 2:
            trigger_current = get_current_trigger_value(tThisFlip, np.array(['Start_Sequence']), np.array([0.0]), trigger_duration)
            if trigger_current != trigger_old:
                print(f'Sequence  ---  Time: {1000*tThisFlip:5.2f}  ---  Trigger Value: {trigger_current}')
            trigger_old = trigger_current
        
        # *text_trigger_start_sequence_blank* updates
        if text_trigger_start_sequence_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_trigger_start_sequence_blank.frameNStart = frameN  # exact frame index
            text_trigger_start_sequence_blank.tStart = t  # local t and not account for scr refresh
            text_trigger_start_sequence_blank.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_trigger_start_sequence_blank, 'tStartRefresh')  # time at next scr refresh
            text_trigger_start_sequence_blank.setAutoDraw(True)
        if text_trigger_start_sequence_blank.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_trigger_start_sequence_blank.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                text_trigger_start_sequence_blank.tStop = t  # not accounting for scr refresh
                text_trigger_start_sequence_blank.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_trigger_start_sequence_blank, 'tStopRefresh')  # time at next scr refresh
                text_trigger_start_sequence_blank.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trigger_start_sequenceComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trigger_start_sequence"-------
    for thisComponent in trigger_start_sequenceComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    loop_repeat_sequence_rest.addData('text_trigger_start_sequence_blank.started', text_trigger_start_sequence_blank.tStartRefresh)
    loop_repeat_sequence_rest.addData('text_trigger_start_sequence_blank.stopped', text_trigger_start_sequence_blank.tStopRefresh)
    
    # set up handler to look after randomisation of conditions etc
    loop_videobreak_sequence = data.TrialHandler(nReps=n_reps_videobreak_sequence, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='loop_videobreak_sequence')
    thisExp.addLoop(loop_videobreak_sequence)  # add the loop to the experiment
    thisLoop_videobreak_sequence = loop_videobreak_sequence.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_videobreak_sequence.rgb)
    if thisLoop_videobreak_sequence != None:
        for paramName in thisLoop_videobreak_sequence:
            exec('{} = thisLoop_videobreak_sequence[paramName]'.format(paramName))
    
    for thisLoop_videobreak_sequence in loop_videobreak_sequence:
        currentLoop = loop_videobreak_sequence
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_videobreak_sequence.rgb)
        if thisLoop_videobreak_sequence != None:
            for paramName in thisLoop_videobreak_sequence:
                exec('{} = thisLoop_videobreak_sequence[paramName]'.format(paramName))
        
        # set up handler to look after randomisation of conditions etc
        loop_animalbreak_sequence = data.TrialHandler(nReps=n_reps_animalbreak_sequence, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='loop_animalbreak_sequence')
        thisExp.addLoop(loop_animalbreak_sequence)  # add the loop to the experiment
        thisLoop_animalbreak_sequence = loop_animalbreak_sequence.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_animalbreak_sequence.rgb)
        if thisLoop_animalbreak_sequence != None:
            for paramName in thisLoop_animalbreak_sequence:
                exec('{} = thisLoop_animalbreak_sequence[paramName]'.format(paramName))
        
        for thisLoop_animalbreak_sequence in loop_animalbreak_sequence:
            currentLoop = loop_animalbreak_sequence
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_animalbreak_sequence.rgb)
            if thisLoop_animalbreak_sequence != None:
                for paramName in thisLoop_animalbreak_sequence:
                    exec('{} = thisLoop_animalbreak_sequence[paramName]'.format(paramName))
            
            # set up handler to look after randomisation of conditions etc
            loop_rvsp_sequence = data.TrialHandler(nReps=1.0, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=data.importConditions(condition_file_sequence, selection=selection_trial_sequence.tolist()),
                seed=None, name='loop_rvsp_sequence')
            thisExp.addLoop(loop_rvsp_sequence)  # add the loop to the experiment
            thisLoop_rvsp_sequence = loop_rvsp_sequence.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_rvsp_sequence.rgb)
            if thisLoop_rvsp_sequence != None:
                for paramName in thisLoop_rvsp_sequence:
                    exec('{} = thisLoop_rvsp_sequence[paramName]'.format(paramName))
            
            for thisLoop_rvsp_sequence in loop_rvsp_sequence:
                currentLoop = loop_rvsp_sequence
                # abbreviate parameter names if possible (e.g. rgb = thisLoop_rvsp_sequence.rgb)
                if thisLoop_rvsp_sequence != None:
                    for paramName in thisLoop_rvsp_sequence:
                        exec('{} = thisLoop_rvsp_sequence[paramName]'.format(paramName))
                
                # ------Prepare to start Routine "rvsp_sequence"-------
                continueRoutine = True
                routineTimer.add(1.050000)
                # update component parameters for each repeat
                current_trial_sequence += 1
                iti_duration_sequence = 0.3*np.random.rand() + 0.3
                if int(expInfo['trigger']) == 1:
                    trigger_values_sequence = np.array([41, Seq_Trigger_Value_Image_1, Seq_Trigger_Value_Sound_1, 42, Seq_Trigger_Value_Image_2, Seq_Trigger_Value_Sound_2, 43, Seq_Trigger_Value_Image_3, Seq_Trigger_Value_Sound_3])
                if int(expInfo['trigger']) == 2:
                    trigger_values_sequence = np.array(['Onset1','Img1','Aud1','Onset2','Img2','Aud2','Onset3','Img3','Aud3'])
                    
                play_tone1 = True
                play_tone2 = True
                play_tone3 = True
                image_1_sequence.setImage(Seq_Image_1)
                image_2_sequence.setImage(Seq_Image_2)
                image_3_sequence.setImage(Seq_Image_3)
                text_current_trial_rsvp_sequence.setText(current_trial_sequence)
                # keep track of which components have finished
                rvsp_sequenceComponents = [image_1_sequence, image_2_sequence, image_3_sequence, text_current_trial_rsvp_sequence]
                for thisComponent in rvsp_sequenceComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                rvsp_sequenceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
                frameN = -1
                
                # -------Run Routine "rvsp_sequence"-------
                while continueRoutine and routineTimer.getTime() > 0:
                    # get current time
                    t = rvsp_sequenceClock.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=rvsp_sequenceClock)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    if int(expInfo['trigger']) == 1:
                        trigger_current = get_current_trigger_value(tThisFlip, trigger_values_sequence, trigger_timemap_sequence, trigger_duration)
                        if trigger_current != trigger_old:
                            win.callOnFlip(p_port.setData, trigger_current)
                        trigger_old = trigger_current
                    
                    if int(expInfo['trigger']) == 2:
                        trigger_current = get_current_trigger_value(tThisFlip, trigger_values_sequence, trigger_timemap_sequence, trigger_duration)
                        if trigger_current != trigger_old:
                            print(f'Sequence  ---  Time: {1000*tThisFlip:5.2f}  ---  Trigger Value: {trigger_current}')
                        trigger_old = trigger_current
                    
                    if play_tone1 and tThisFlip >= 0.0-frameTolerance:
                        tone_dict[Seq_Sound_1].play()
                        play_tone1 = False
                    
                    if play_tone2 and tThisFlip >= 0.35-frameTolerance:
                        tone_dict[Seq_Sound_2].play()
                        play_tone2 = False
                    
                    if play_tone3 and tThisFlip >= 0.7-frameTolerance:
                        tone_dict[Seq_Sound_3].play()
                        play_tone3 = False
                    
                    # *image_1_sequence* updates
                    if image_1_sequence.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_1_sequence.frameNStart = frameN  # exact frame index
                        image_1_sequence.tStart = t  # local t and not account for scr refresh
                        image_1_sequence.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_1_sequence, 'tStartRefresh')  # time at next scr refresh
                        image_1_sequence.setAutoDraw(True)
                    if image_1_sequence.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_1_sequence.tStartRefresh + 0.250-frameTolerance:
                            # keep track of stop time/frame for later
                            image_1_sequence.tStop = t  # not accounting for scr refresh
                            image_1_sequence.frameNStop = frameN  # exact frame index
                            win.timeOnFlip(image_1_sequence, 'tStopRefresh')  # time at next scr refresh
                            image_1_sequence.setAutoDraw(False)
                    
                    # *image_2_sequence* updates
                    if image_2_sequence.status == NOT_STARTED and tThisFlip >= 0.350-frameTolerance:
                        # keep track of start time/frame for later
                        image_2_sequence.frameNStart = frameN  # exact frame index
                        image_2_sequence.tStart = t  # local t and not account for scr refresh
                        image_2_sequence.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_2_sequence, 'tStartRefresh')  # time at next scr refresh
                        image_2_sequence.setAutoDraw(True)
                    if image_2_sequence.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_2_sequence.tStartRefresh + 0.250-frameTolerance:
                            # keep track of stop time/frame for later
                            image_2_sequence.tStop = t  # not accounting for scr refresh
                            image_2_sequence.frameNStop = frameN  # exact frame index
                            win.timeOnFlip(image_2_sequence, 'tStopRefresh')  # time at next scr refresh
                            image_2_sequence.setAutoDraw(False)
                    
                    # *image_3_sequence* updates
                    if image_3_sequence.status == NOT_STARTED and tThisFlip >= 0.70-frameTolerance:
                        # keep track of start time/frame for later
                        image_3_sequence.frameNStart = frameN  # exact frame index
                        image_3_sequence.tStart = t  # local t and not account for scr refresh
                        image_3_sequence.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_3_sequence, 'tStartRefresh')  # time at next scr refresh
                        image_3_sequence.setAutoDraw(True)
                    if image_3_sequence.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_3_sequence.tStartRefresh + 0.25-frameTolerance:
                            # keep track of stop time/frame for later
                            image_3_sequence.tStop = t  # not accounting for scr refresh
                            image_3_sequence.frameNStop = frameN  # exact frame index
                            win.timeOnFlip(image_3_sequence, 'tStopRefresh')  # time at next scr refresh
                            image_3_sequence.setAutoDraw(False)
                    
                    # *text_current_trial_rsvp_sequence* updates
                    if text_current_trial_rsvp_sequence.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text_current_trial_rsvp_sequence.frameNStart = frameN  # exact frame index
                        text_current_trial_rsvp_sequence.tStart = t  # local t and not account for scr refresh
                        text_current_trial_rsvp_sequence.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_current_trial_rsvp_sequence, 'tStartRefresh')  # time at next scr refresh
                        text_current_trial_rsvp_sequence.setAutoDraw(True)
                    if text_current_trial_rsvp_sequence.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text_current_trial_rsvp_sequence.tStartRefresh + 1.05-frameTolerance:
                            # keep track of stop time/frame for later
                            text_current_trial_rsvp_sequence.tStop = t  # not accounting for scr refresh
                            text_current_trial_rsvp_sequence.frameNStop = frameN  # exact frame index
                            win.timeOnFlip(text_current_trial_rsvp_sequence, 'tStopRefresh')  # time at next scr refresh
                            text_current_trial_rsvp_sequence.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                        core.quit()
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in rvsp_sequenceComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # -------Ending Routine "rvsp_sequence"-------
                for thisComponent in rvsp_sequenceComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                loop_rvsp_sequence.addData('image_1_sequence.started', image_1_sequence.tStartRefresh)
                loop_rvsp_sequence.addData('image_1_sequence.stopped', image_1_sequence.tStopRefresh)
                loop_rvsp_sequence.addData('image_2_sequence.started', image_2_sequence.tStartRefresh)
                loop_rvsp_sequence.addData('image_2_sequence.stopped', image_2_sequence.tStopRefresh)
                loop_rvsp_sequence.addData('image_3_sequence.started', image_3_sequence.tStartRefresh)
                loop_rvsp_sequence.addData('image_3_sequence.stopped', image_3_sequence.tStopRefresh)
                loop_rvsp_sequence.addData('text_current_trial_rsvp_sequence.started', text_current_trial_rsvp_sequence.tStartRefresh)
                loop_rvsp_sequence.addData('text_current_trial_rsvp_sequence.stopped', text_current_trial_rsvp_sequence.tStopRefresh)
                
                # ------Prepare to start Routine "iti_sequence"-------
                continueRoutine = True
                # update component parameters for each repeat
                text_current_trial_iti_sequence.setText(current_trial_sequence)
                # keep track of which components have finished
                iti_sequenceComponents = [text_current_trial_iti_sequence]
                for thisComponent in iti_sequenceComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                iti_sequenceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
                frameN = -1
                
                # -------Run Routine "iti_sequence"-------
                while continueRoutine:
                    # get current time
                    t = iti_sequenceClock.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=iti_sequenceClock)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *text_current_trial_iti_sequence* updates
                    if text_current_trial_iti_sequence.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text_current_trial_iti_sequence.frameNStart = frameN  # exact frame index
                        text_current_trial_iti_sequence.tStart = t  # local t and not account for scr refresh
                        text_current_trial_iti_sequence.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_current_trial_iti_sequence, 'tStartRefresh')  # time at next scr refresh
                        text_current_trial_iti_sequence.setAutoDraw(True)
                    if text_current_trial_iti_sequence.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text_current_trial_iti_sequence.tStartRefresh + iti_duration_sequence-frameTolerance:
                            # keep track of stop time/frame for later
                            text_current_trial_iti_sequence.tStop = t  # not accounting for scr refresh
                            text_current_trial_iti_sequence.frameNStop = frameN  # exact frame index
                            win.timeOnFlip(text_current_trial_iti_sequence, 'tStopRefresh')  # time at next scr refresh
                            text_current_trial_iti_sequence.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                        core.quit()
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in iti_sequenceComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # -------Ending Routine "iti_sequence"-------
                for thisComponent in iti_sequenceComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                loop_rvsp_sequence.addData('text_current_trial_iti_sequence.started', text_current_trial_iti_sequence.tStartRefresh)
                loop_rvsp_sequence.addData('text_current_trial_iti_sequence.stopped', text_current_trial_iti_sequence.tStopRefresh)
                # the Routine "iti_sequence" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
            # completed 1.0 repeats of 'loop_rvsp_sequence'
            
            
            # ------Prepare to start Routine "attentioncatch_animal_sequence"-------
            continueRoutine = True
            routineTimer.add(5.000000)
            # update component parameters for each repeat
            selection_trial_sequence += selection_trial_sequence.shape[0]
            rnd_idx = np.random.randint(animal_catch.shape[0])
            image_animal_sequence.setImage(animal_catch[rnd_idx,0])
            sound_animal_localizer.setSound(animal_catch[rnd_idx,1], secs=3.0, hamming=True)
            sound_animal_localizer.setVolume(1.0, log=False)
            text_current_trial_animal_sequence.setText(current_trial_sequence)
            # keep track of which components have finished
            attentioncatch_animal_sequenceComponents = [image_animal_sequence, sound_animal_localizer, text_current_trial_animal_sequence]
            for thisComponent in attentioncatch_animal_sequenceComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            attentioncatch_animal_sequenceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "attentioncatch_animal_sequence"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = attentioncatch_animal_sequenceClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=attentioncatch_animal_sequenceClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Send Trigger
                if int(expInfo['trigger']) == 1:
                    trigger_current = get_current_trigger_value(tThisFlip, np.array([240, 249]), np.array([1.0,4.0]), trigger_duration)
                    if trigger_current != trigger_old:
                        win.callOnFlip(p_port.setData, trigger_current)
                    trigger_old = trigger_current
                
                if int(expInfo['trigger']) == 2:
                    trigger_current = get_current_trigger_value(tThisFlip, np.array(['AnimalBreak_Start','AnimalBreak_Stop']), np.array([0,4.9]), trigger_duration)
                    if trigger_current != trigger_old:
                        print(f'Sequence  ---  Time: {1000*tThisFlip:5.2f}  ---  Trigger Value: {trigger_current}')
                    trigger_old = trigger_current
                
                # *image_animal_sequence* updates
                if image_animal_sequence.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_animal_sequence.frameNStart = frameN  # exact frame index
                    image_animal_sequence.tStart = t  # local t and not account for scr refresh
                    image_animal_sequence.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_animal_sequence, 'tStartRefresh')  # time at next scr refresh
                    image_animal_sequence.setAutoDraw(True)
                if image_animal_sequence.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_animal_sequence.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_animal_sequence.tStop = t  # not accounting for scr refresh
                        image_animal_sequence.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(image_animal_sequence, 'tStopRefresh')  # time at next scr refresh
                        image_animal_sequence.setAutoDraw(False)
                # start/stop sound_animal_localizer
                if sound_animal_localizer.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                    # keep track of start time/frame for later
                    sound_animal_localizer.frameNStart = frameN  # exact frame index
                    sound_animal_localizer.tStart = t  # local t and not account for scr refresh
                    sound_animal_localizer.tStartRefresh = tThisFlipGlobal  # on global time
                    sound_animal_localizer.play(when=win)  # sync with win flip
                if sound_animal_localizer.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sound_animal_localizer.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        sound_animal_localizer.tStop = t  # not accounting for scr refresh
                        sound_animal_localizer.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(sound_animal_localizer, 'tStopRefresh')  # time at next scr refresh
                        sound_animal_localizer.stop()
                
                # *text_current_trial_animal_sequence* updates
                if text_current_trial_animal_sequence.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_current_trial_animal_sequence.frameNStart = frameN  # exact frame index
                    text_current_trial_animal_sequence.tStart = t  # local t and not account for scr refresh
                    text_current_trial_animal_sequence.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_current_trial_animal_sequence, 'tStartRefresh')  # time at next scr refresh
                    text_current_trial_animal_sequence.setAutoDraw(True)
                if text_current_trial_animal_sequence.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_current_trial_animal_sequence.tStartRefresh + 5.0-frameTolerance:
                        # keep track of stop time/frame for later
                        text_current_trial_animal_sequence.tStop = t  # not accounting for scr refresh
                        text_current_trial_animal_sequence.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(text_current_trial_animal_sequence, 'tStopRefresh')  # time at next scr refresh
                        text_current_trial_animal_sequence.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in attentioncatch_animal_sequenceComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "attentioncatch_animal_sequence"-------
            for thisComponent in attentioncatch_animal_sequenceComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            loop_animalbreak_sequence.addData('image_animal_sequence.started', image_animal_sequence.tStartRefresh)
            loop_animalbreak_sequence.addData('image_animal_sequence.stopped', image_animal_sequence.tStopRefresh)
            sound_animal_localizer.stop()  # ensure sound has stopped at end of routine
            loop_animalbreak_sequence.addData('sound_animal_localizer.started', sound_animal_localizer.tStartRefresh)
            loop_animalbreak_sequence.addData('sound_animal_localizer.stopped', sound_animal_localizer.tStopRefresh)
            loop_animalbreak_sequence.addData('text_current_trial_animal_sequence.started', text_current_trial_animal_sequence.tStartRefresh)
            loop_animalbreak_sequence.addData('text_current_trial_animal_sequence.stopped', text_current_trial_animal_sequence.tStopRefresh)
            thisExp.nextEntry()
            
        # completed n_reps_animalbreak_sequence repeats of 'loop_animalbreak_sequence'
        
        
        # ------Prepare to start Routine "attentioncatch_video_sequence"-------
        continueRoutine = True
        routineTimer.add(8.000000)
        # update component parameters for each repeat
        rnd_idx = randint(video_catch.shape[0])
        
        # Start Video
        start_video = True
        #movie_dict[video_catch[rnd_idx]].setAutoDraw(True)
        text_current_trial_video_sequence.setText(current_trial_sequence)
        # keep track of which components have finished
        attentioncatch_video_sequenceComponents = [text_current_trial_video_sequence]
        for thisComponent in attentioncatch_video_sequenceComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        attentioncatch_video_sequenceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "attentioncatch_video_sequence"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = attentioncatch_video_sequenceClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=attentioncatch_video_sequenceClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Send Trigger
            if int(expInfo['trigger']) == 1:
                trigger_current = get_current_trigger_value(tThisFlip, np.array([230, 239]), np.array([1.0, 7.0]), trigger_duration)
                if trigger_current != trigger_old:
                    win.callOnFlip(p_port.setData, trigger_current)
                trigger_old = trigger_current
            
            if int(expInfo['trigger']) == 2:
                trigger_current = get_current_trigger_value(tThisFlip, np.array(['VideoBreak_Start','VideoBreak_Stop']), np.array([0, 4.9]), trigger_duration)
                if trigger_current != trigger_old:
                    print(f'Sequence  ---  Time: {1000*tThisFlip:5.2f}  ---  Trigger Value: {trigger_current}')
                trigger_old = trigger_current
            
            if start_video and (tThisFlip >= 1.0-frameTolerance):
                movie_dict[video_catch[rnd_idx]].setAutoDraw(True)
                start_video = False
            
            if not start_video and (tThisFlip >= 7.0-frameTolerance):
                movie_dict[video_catch[rnd_idx]].setAutoDraw(False)
            
            # *text_current_trial_video_sequence* updates
            if text_current_trial_video_sequence.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_current_trial_video_sequence.frameNStart = frameN  # exact frame index
                text_current_trial_video_sequence.tStart = t  # local t and not account for scr refresh
                text_current_trial_video_sequence.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_current_trial_video_sequence, 'tStartRefresh')  # time at next scr refresh
                text_current_trial_video_sequence.setAutoDraw(True)
            if text_current_trial_video_sequence.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_current_trial_video_sequence.tStartRefresh + 8.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_current_trial_video_sequence.tStop = t  # not accounting for scr refresh
                    text_current_trial_video_sequence.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(text_current_trial_video_sequence, 'tStopRefresh')  # time at next scr refresh
                    text_current_trial_video_sequence.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in attentioncatch_video_sequenceComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "attentioncatch_video_sequence"-------
        for thisComponent in attentioncatch_video_sequenceComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Stop Video
        #movie_dict[video_catch[rnd_idx]].setAutoDraw(False)
        loop_videobreak_sequence.addData('text_current_trial_video_sequence.started', text_current_trial_video_sequence.tStartRefresh)
        loop_videobreak_sequence.addData('text_current_trial_video_sequence.stopped', text_current_trial_video_sequence.tStopRefresh)
        thisExp.nextEntry()
        
    # completed n_reps_videobreak_sequence repeats of 'loop_videobreak_sequence'
    
    
    # ------Prepare to start Routine "trigger_stop_sequence"-------
    continueRoutine = True
    routineTimer.add(3.000000)
    # update component parameters for each repeat
    # keep track of which components have finished
    trigger_stop_sequenceComponents = [text_trigger_stop_sequence_blank]
    for thisComponent in trigger_stop_sequenceComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trigger_stop_sequenceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "trigger_stop_sequence"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = trigger_stop_sequenceClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trigger_stop_sequenceClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        if int(expInfo['trigger']) == 1:
            trigger_current = get_current_trigger_value(tThisFlip, np.array([49]), np.array([2.0]), trigger_duration)
            if trigger_current != trigger_old:
                win.callOnFlip(p_port.setData, trigger_current)
            trigger_old = trigger_current
        
        if int(expInfo['trigger']) == 2:
            trigger_current = get_current_trigger_value(tThisFlip, np.array(['Stop_Sequence']), np.array([2.0]), trigger_duration)
            if trigger_current != trigger_old:
                print(f'Sequence  ---  Time: {1000*tThisFlip:5.2f}  ---  Trigger Value: {trigger_current}')
            trigger_old = trigger_current
        
        # *text_trigger_stop_sequence_blank* updates
        if text_trigger_stop_sequence_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_trigger_stop_sequence_blank.frameNStart = frameN  # exact frame index
            text_trigger_stop_sequence_blank.tStart = t  # local t and not account for scr refresh
            text_trigger_stop_sequence_blank.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_trigger_stop_sequence_blank, 'tStartRefresh')  # time at next scr refresh
            text_trigger_stop_sequence_blank.setAutoDraw(True)
        if text_trigger_stop_sequence_blank.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_trigger_stop_sequence_blank.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                text_trigger_stop_sequence_blank.tStop = t  # not accounting for scr refresh
                text_trigger_stop_sequence_blank.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_trigger_stop_sequence_blank, 'tStopRefresh')  # time at next scr refresh
                text_trigger_stop_sequence_blank.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trigger_stop_sequenceComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trigger_stop_sequence"-------
    for thisComponent in trigger_stop_sequenceComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    loop_repeat_sequence_rest.addData('text_trigger_stop_sequence_blank.started', text_trigger_stop_sequence_blank.tStartRefresh)
    loop_repeat_sequence_rest.addData('text_trigger_stop_sequence_blank.stopped', text_trigger_stop_sequence_blank.tStopRefresh)
    
    # ------Prepare to start Routine "start_resting"-------
    continueRoutine = True
    routineTimer.add(1.000000)
    # update component parameters for each repeat
    # keep track of which components have finished
    start_restingComponents = [text_start_resting]
    for thisComponent in start_restingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    start_restingClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "start_resting"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = start_restingClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=start_restingClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        if int(expInfo['trigger']) == 1:
            trigger_current = get_current_trigger_value(tThisFlip, np.array([60]), np.array([0.0]), trigger_duration)
            if trigger_current != trigger_old:
                win.callOnFlip(p_port.setData, trigger_current)
            trigger_old = trigger_current
        
        if int(expInfo['trigger']) == 2:
            trigger_current = get_current_trigger_value(tThisFlip, np.array(['Start_Resting']), np.array([0.0]), trigger_duration)
            if trigger_current != trigger_old:
                print(f'Resting  ---  Time: {1000*tThisFlip:5.2f}  ---  Trigger Value: {trigger_current}')
            trigger_old = trigger_current
        
        # *text_start_resting* updates
        if text_start_resting.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_start_resting.frameNStart = frameN  # exact frame index
            text_start_resting.tStart = t  # local t and not account for scr refresh
            text_start_resting.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_start_resting, 'tStartRefresh')  # time at next scr refresh
            text_start_resting.setAutoDraw(True)
        if text_start_resting.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_start_resting.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_start_resting.tStop = t  # not accounting for scr refresh
                text_start_resting.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_start_resting, 'tStopRefresh')  # time at next scr refresh
                text_start_resting.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start_restingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "start_resting"-------
    for thisComponent in start_restingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    loop_repeat_sequence_rest.addData('text_start_resting.started', text_start_resting.tStartRefresh)
    loop_repeat_sequence_rest.addData('text_start_resting.stopped', text_start_resting.tStopRefresh)
    
    # set up handler to look after randomisation of conditions etc
    loop_cue_resting = data.TrialHandler(nReps=n_reps_rest_cues, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='loop_cue_resting')
    thisExp.addLoop(loop_cue_resting)  # add the loop to the experiment
    thisLoop_cue_resting = loop_cue_resting.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_cue_resting.rgb)
    if thisLoop_cue_resting != None:
        for paramName in thisLoop_cue_resting:
            exec('{} = thisLoop_cue_resting[paramName]'.format(paramName))
    
    for thisLoop_cue_resting in loop_cue_resting:
        currentLoop = loop_cue_resting
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_cue_resting.rgb)
        if thisLoop_cue_resting != None:
            for paramName in thisLoop_cue_resting:
                exec('{} = thisLoop_cue_resting[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "blank_resting"-------
        continueRoutine = True
        # update component parameters for each repeat
        current_trial_resting += 1
        current_duration_resting = 3.0 + 2.0*np.random.rand()
        text_current_trial_blank_resting.setText(current_trial_resting)
        # keep track of which components have finished
        blank_restingComponents = [text_current_trial_blank_resting]
        for thisComponent in blank_restingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        blank_restingClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "blank_resting"-------
        while continueRoutine:
            # get current time
            t = blank_restingClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=blank_restingClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_current_trial_blank_resting* updates
            if text_current_trial_blank_resting.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_current_trial_blank_resting.frameNStart = frameN  # exact frame index
                text_current_trial_blank_resting.tStart = t  # local t and not account for scr refresh
                text_current_trial_blank_resting.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_current_trial_blank_resting, 'tStartRefresh')  # time at next scr refresh
                text_current_trial_blank_resting.setAutoDraw(True)
            if text_current_trial_blank_resting.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_current_trial_blank_resting.tStartRefresh + current_duration_resting-frameTolerance:
                    # keep track of stop time/frame for later
                    text_current_trial_blank_resting.tStop = t  # not accounting for scr refresh
                    text_current_trial_blank_resting.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(text_current_trial_blank_resting, 'tStopRefresh')  # time at next scr refresh
                    text_current_trial_blank_resting.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank_restingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "blank_resting"-------
        for thisComponent in blank_restingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        loop_cue_resting.addData('text_current_trial_blank_resting.started', text_current_trial_blank_resting.tStartRefresh)
        loop_cue_resting.addData('text_current_trial_blank_resting.stopped', text_current_trial_blank_resting.tStopRefresh)
        # the Routine "blank_resting" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # ------Prepare to start Routine "cue_resting"-------
        continueRoutine = True
        routineTimer.add(0.250000)
        # update component parameters for each repeat
        if int(expInfo['trigger']) == 1:
            trigger_values_resting = np.array([61, 152])
        if int(expInfo['trigger']) == 2:
            trigger_values_resting = np.array(['Cue', 'Aud'])
        
        play_tone = True
        text_current_trial_cue_resting.setText(current_trial_resting)
        # keep track of which components have finished
        cue_restingComponents = [text_current_trial_cue_resting]
        for thisComponent in cue_restingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        cue_restingClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "cue_resting"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = cue_restingClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=cue_restingClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            if int(expInfo['trigger']) == 1:
                trigger_current = get_current_trigger_value(tThisFlip, trigger_values_resting, trigger_timemap_resting, trigger_duration)
                if trigger_current != trigger_old:
                    win.callOnFlip(p_port.setData, trigger_current)
                trigger_old = trigger_current
            
            if int(expInfo['trigger']) == 2:
                trigger_current = get_current_trigger_value(tThisFlip, trigger_values_resting, trigger_timemap_resting, trigger_duration)
                if trigger_current != trigger_old:
                    print(f'Resting  ---  Time: {1000*tThisFlip:5.2f}  ---  Trigger Value: {trigger_current}')
                trigger_old = trigger_current
            
            if play_tone:
                tone_dict[240].play()
                play_tone = False
            
            # *text_current_trial_cue_resting* updates
            if text_current_trial_cue_resting.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_current_trial_cue_resting.frameNStart = frameN  # exact frame index
                text_current_trial_cue_resting.tStart = t  # local t and not account for scr refresh
                text_current_trial_cue_resting.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_current_trial_cue_resting, 'tStartRefresh')  # time at next scr refresh
                text_current_trial_cue_resting.setAutoDraw(True)
            if text_current_trial_cue_resting.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_current_trial_cue_resting.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    text_current_trial_cue_resting.tStop = t  # not accounting for scr refresh
                    text_current_trial_cue_resting.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(text_current_trial_cue_resting, 'tStopRefresh')  # time at next scr refresh
                    text_current_trial_cue_resting.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in cue_restingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "cue_resting"-------
        for thisComponent in cue_restingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        loop_cue_resting.addData('text_current_trial_cue_resting.started', text_current_trial_cue_resting.tStartRefresh)
        loop_cue_resting.addData('text_current_trial_cue_resting.stopped', text_current_trial_cue_resting.tStopRefresh)
        thisExp.nextEntry()
        
    # completed n_reps_rest_cues repeats of 'loop_cue_resting'
    
    
    # ------Prepare to start Routine "stop_resting"-------
    continueRoutine = True
    routineTimer.add(1.000000)
    # update component parameters for each repeat
    # keep track of which components have finished
    stop_restingComponents = [text_end_resting]
    for thisComponent in stop_restingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    stop_restingClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "stop_resting"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = stop_restingClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=stop_restingClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Send Trigger
        if int(expInfo['trigger']) == 1:
            trigger_current = get_current_trigger_value(tThisFlip, np.array([69]), np.array([0.5]), trigger_duration)
            if trigger_current != trigger_old:
                win.callOnFlip(p_port.setData, trigger_current)
            trigger_old = trigger_current
        
        if int(expInfo['trigger']) == 2:
            trigger_current = get_current_trigger_value(tThisFlip, np.array(['Stop_Resting']), np.array([0.5]), trigger_duration)
            if trigger_current != trigger_old:
                print(f'Resting  ---  Time: {1000*tThisFlip:5.2f}  ---  Trigger Value: {trigger_current}')
            trigger_old = trigger_current
        
        # *text_end_resting* updates
        if text_end_resting.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end_resting.frameNStart = frameN  # exact frame index
            text_end_resting.tStart = t  # local t and not account for scr refresh
            text_end_resting.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end_resting, 'tStartRefresh')  # time at next scr refresh
            text_end_resting.setAutoDraw(True)
        if text_end_resting.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_end_resting.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_end_resting.tStop = t  # not accounting for scr refresh
                text_end_resting.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_end_resting, 'tStopRefresh')  # time at next scr refresh
                text_end_resting.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in stop_restingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "stop_resting"-------
    for thisComponent in stop_restingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    loop_repeat_sequence_rest.addData('text_end_resting.started', text_end_resting.tStartRefresh)
    loop_repeat_sequence_rest.addData('text_end_resting.stopped', text_end_resting.tStopRefresh)
    thisExp.nextEntry()
    
# completed n_reps_sequence_rest repeats of 'loop_repeat_sequence_rest'


# ------Prepare to start Routine "end_screen"-------
continueRoutine = True
# update component parameters for each repeat
keyb_end_screen.keys = []
keyb_end_screen.rt = []
_keyb_end_screen_allKeys = []
# keep track of which components have finished
end_screenComponents = [text_end_screen, keyb_end_screen]
for thisComponent in end_screenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
end_screenClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "end_screen"-------
while continueRoutine:
    # get current time
    t = end_screenClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=end_screenClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_end_screen* updates
    if text_end_screen.status == NOT_STARTED and tThisFlip >= 4.0-frameTolerance:
        # keep track of start time/frame for later
        text_end_screen.frameNStart = frameN  # exact frame index
        text_end_screen.tStart = t  # local t and not account for scr refresh
        text_end_screen.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_end_screen, 'tStartRefresh')  # time at next scr refresh
        text_end_screen.setAutoDraw(True)
    
    # *keyb_end_screen* updates
    waitOnFlip = False
    if keyb_end_screen.status == NOT_STARTED and tThisFlip >= 4.0-frameTolerance:
        # keep track of start time/frame for later
        keyb_end_screen.frameNStart = frameN  # exact frame index
        keyb_end_screen.tStart = t  # local t and not account for scr refresh
        keyb_end_screen.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(keyb_end_screen, 'tStartRefresh')  # time at next scr refresh
        keyb_end_screen.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(keyb_end_screen.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(keyb_end_screen.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if keyb_end_screen.status == STARTED and not waitOnFlip:
        theseKeys = keyb_end_screen.getKeys(keyList=['space'], waitRelease=False)
        _keyb_end_screen_allKeys.extend(theseKeys)
        if len(_keyb_end_screen_allKeys):
            keyb_end_screen.keys = _keyb_end_screen_allKeys[-1].name  # just the last key pressed
            keyb_end_screen.rt = _keyb_end_screen_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in end_screenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "end_screen"-------
for thisComponent in end_screenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_end_screen.started', text_end_screen.tStartRefresh)
thisExp.addData('text_end_screen.stopped', text_end_screen.tStopRefresh)
# check responses
if keyb_end_screen.keys in ['', [], None]:  # No response was made
    keyb_end_screen.keys = None
thisExp.addData('keyb_end_screen.keys',keyb_end_screen.keys)
if keyb_end_screen.keys != None:  # we had a response
    thisExp.addData('keyb_end_screen.rt', keyb_end_screen.rt)
thisExp.addData('keyb_end_screen.started', keyb_end_screen.tStartRefresh)
thisExp.addData('keyb_end_screen.stopped', keyb_end_screen.tStopRefresh)
thisExp.nextEntry()
# the Routine "end_screen" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
