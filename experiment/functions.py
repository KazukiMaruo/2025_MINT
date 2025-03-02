
"""
TODOS:
    - Eye-tracker set-up
Info:
    - Numerosity from 1 ~ 6
     Day 1
        - Audio: 35 trials for each condition (12 conditions) = 10 min 
        - Visual: 70 trials for each condition (18 conditions) = 10 min 

    Day 2
        - Audio: 35 trials for each condition (12 conditions) = 10 min

"""





def run_audio(monitor_data = [1920, 1080, 50, 90], EEG_trigger = True):

    # ~~~~~~~~~~~~~~~ Import packages
    from psychopy import locale_setup
    from psychopy import prefs
    from psychopy import plugins
    plugins.activatePlugins()
    prefs.hardware['audioLib'] = 'ptb'
    prefs.hardware['audioLatencyMode'] = '3'
    from psychopy import sound, gui, parallel, visual, core, data, event, logging, clock, colors, layout, monitors
    from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                    STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

    import numpy as np  # whole numpy lib is available, prepend 'np.'
    from numpy import (sin, cos, tan, log, log10, pi, average,
                    sqrt, std, deg2rad, rad2deg, linspace, asarray)
    from numpy.random import random, randint, normal, shuffle, choice as randchoice
    import os  # handy system and path functions
    import sys  # to get file system encoding
    import random
    import time
    import re
    import psychopy.iohub as io
    from psychopy.hardware import keyboard


    # ~~~~~~~~~~~~~~~ Directory path setting
    _thisDir = os.path.dirname(os.path.abspath(__file__)) # '__file__' indicate the current directory of this script
    os.chdir(_thisDir) # change the current working directory to the directory specified by '_thisdir'
    # Store info about the experiment session
    psychopyVersion = '2024.1.5'
    expName = 'audio'  # from the Builder filename that created this script
    expInfo = {
        'participant': '',
        'session': '1',
        'date|hid': data.getDateStr(), # "|hid" indicate this info is hidden in the GUI
        'expName|hid': expName,
        'psychopyVersion|hid': psychopyVersion,
        }

    # ~~~~~~~~~~~~~~~ Show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel

    # define csv file name 
    filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date|hid']) # 'os.sep' is a system independent separator for both window and mac 

    # An ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, 
        version='',
        extraInfo=expInfo, 
        runtimeInfo=None,
        originPath='',
        savePickle=True, # saved in a binary pickle file (.psydat)
        saveWideText=True, # saved in a wide-format text file, usually a CSV file
        dataFileName=filename
        )

    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP) # This creates a log file to store detailed information about the experiment’s execution, such as events, errors, and system messages.
    logging.console.setLevel(logging.WARNING)  # only warning messages and errors will be printed to the screen (console)

    endExpNow = False  # whether the experiment should be terminated immediately, You can later set this flag to True in response to certain conditions
    frameTolerance = 0.001  # how close to onset before 'same' frame



    # ~~~~~~~~~~~~~~~ Setup the Window
    Monitor_data = monitor_data # [1920, 1080, 30, 50], [1440, 900, 30, 50] 
    mon = monitors.Monitor(
        'stimulus_screen', width=Monitor_data[2], distance=Monitor_data[3]) # width (cm) is to calcurate the visual angle, distance (cm) is viewing distance.
    mon.setSizePix((Monitor_data[0], Monitor_data[1]))
    mon.save()

    _size = (Monitor_data[0], Monitor_data[1]) # size (resolution) of the window in pixels, e.g., [1920, 1080] for Full HD.
    _fullscr = True # whether the window is displayed in fullscreen mode
    _screen = 0 # which monitor to display
    _winType = 'pyglet' # Specifies the underlying graphics library to use for window management.
    _allowGUI=False # stencil buffer is allowed. The stencil buffer is used for more advanced graphical effects, such as masking.
    _monitor='stimulus_screen' # Refers to a monitor profile defined in PsychoPy
    _color=[0, 0, 0] # RGB color code
    _colorSpace='rgb' # Specifies the color space
    _backgroundImage='' # Allows for setting a background image to be displayed behind any stimuli
    _backgroundFit='none' # Specifies how the background image should fit to the window if provided
    _blendMode='avg' # Controls how overlapping stimuli are blended together on the screen. 'avg' means averaging the pixel values of overlapping stimuli.
    _useFBO=True # Setting useFBO=True allows PsychoPy to render stimuli off-screen (in a buffer) before displaying them. This can improve rendering flexibility and performance, especially when dealing with complex stimuli or advanced visual effects.
    _units='norm' # Defines the units used for positioning and sizing stimuli in the window. units='height' sets the unit of measurement relative to the height of the window. For example, setting a stimulus' size to 0.5 means it will occupy half of the window’s height. Other possible units include norm, pix (pixels), and cm (centimeters).

    win = visual.Window(
        size=_size, 
        fullscr=_fullscr, 
        screen=_screen, 
        winType=_winType, 
        allowGUI=_allowGUI,
        monitor=_monitor,
        color=_color, 
        colorSpace=_colorSpace,
        backgroundImage=_backgroundImage, 
        backgroundFit=_backgroundFit,
        blendMode=_blendMode, 
        useFBO=_useFBO, 
        units=_units
        )
    win.mouseVisible = False # whether mouse cursor is visible or not.

    # store frame rate of monitor if we can measure it
    expInfo['frameRate'] = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...') # determining how frequently the window refreshes (measured in frames per second (FPS))
    if expInfo['frameRate'] != None: # if it is not None
        frameDur = 1.0 / round(expInfo['frameRate']) # For example, if the frame rate is 60 Hz (60 frames per second), then the duration of one frame is 1.0 / 60 = 0.01667 seconds (about 16.67 milliseconds per frame).
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess



    # ~~~~~~~~~~~~~~~ Setup input devices
    ioConfig = {}
    defaultKeyboard = keyboard.Keyboard()

    # add device to the dictionary
    ioConfig['keyboard'] = defaultKeyboard

    # ~~~~~~~~~~~~~~ Parallel port
    if EEG_trigger == True:
        p_port = parallel.ParallelPort(address='0x0378') # or '0x03BC' # This is typically the standard address for LPT1, the first parallel port on many older PCs
        p_port.setData(0) # 0 in this case means all 8 data pins are set to low. In binary, 0 is represented as 00000000, so each pin gets set to 0 (off).
        # Function to send a trigger
        def send_trigger(trigger_value):
            p_port.setData(trigger_value) # Send the trigger value
            time.sleep(0.001) # Wait for a short period to ensure the trigger is registered (e.g., 1 ms) 
            p_port.setData(0) # Reset the parallel port to 0


    # ~~~~~~~~~~~~~~~ Components for Routine "welcomescreen" 
    text_welcome = visual.TextStim(win=win, 
                                name='text_welcome', # The name of this text stimulus
                                text='When you are ready, press "space".', # The actual text to be displayed on the screen
                                font='Open Sans',
                                pos=(0, 0), # (0, 0) indicates that the text will be centered at the origin of the window,
                                height=1, # The height of the text, which determines how large the characters will appear. The unit of height is defined relative to the window size (e.g., 0.05 would be 5% of the window height if you're using the 'height' unit for the window
                                wrapWidth=None, # This parameter controls text wrapping
                                ori=0.0, # The orientation of the text
                                color=[1.0, 1.0, 1.0], 
                                colorSpace='rgb', 
                                opacity=None, # The opacity of the text
                                languageStyle='LTR', # Defines the direction of the text. 'LTR' stands for Left-to-Right,
                                depth=0.0,
                                units='cm'
                                )



    # ~~~~~~~~~~~~~~~ components for Routine "trial" ---
    # Red diagnoal line
    line_diagonal_1 = visual.Line(
        win=win,
        name='line_diagonal_1',
        start=(-1, 1),  # Start coordinate (top-left corner relative to window size)
        end=(1, -1),    # End coordinate (bottom-right corner relative to window size)
        ori=0,
        pos=(0,0),
        lineColor=[250, 0, 0],
        colorSpace='rgb',
        fillColor=[1, 1, 1],
        opacity=1,
        depth=0.0,
        interpolate=True, 
        autoDraw=False,
        lineWidth=1.5,        # Width of the line
        units='norm'
        )
    line_diagonal_2 = visual.Line(
        win=win,
        name='line_diagonal_2',
        start=(-1, -1),  # Start coordinate (top-left corner relative to window size)
        end=(1, 1),    # End coordinate (bottom-right corner relative to window size)
        ori=0,
        pos=(0,0),
        lineColor=[250, 0, 0],
        colorSpace='rgb',
        fillColor=[1, 1, 1],
        opacity=1,
        depth=0.0,
        interpolate=True, 
        autoDraw=False,
        lineWidth=1.5,        # Width of the line
        units='norm'
        )
    
    # Create a text stimulus for the trial number
    trial_text = visual.TextStim(win=win, text='', pos=(-0.97, -0.97), height=0.04, color='black', alignText='center')

    # Audio beep
    wav_file_path = _thisDir + os.sep + 'stimuli/audio/beep.wav' 
    beep = sound.Sound(value=wav_file_path, 
                    stereo=True, #  This indicates that the sound should be played in stereo mode, which involves two channels (left and right) to provide a richer sound experience
                    hamming=True, # A Hamming window is used to reduce spectral leakage in signal processing. It is generally applied to improve the quality of generated sounds by smoothing the edges.
                    name='beep'
                    )
    beep.setVolume(1.0)

    # specify the trial just before the catch trial
    catch_points = list(range(11, 421, 12))
    ii = 0 # catch_points_idx

    # Audio for catch trials
    all_audios = ["1","2","3","4"]
    audios = {}
    for audioNum in all_audios:
        audio_file_path = _thisDir + os.sep + f'stimuli/catch/audio/{audioNum}.wav'
        audio = sound.Sound(value=audio_file_path, 
                        stereo=True, #  This indicates that the sound should be played in stereo mode, which involves two channels (left and right) to provide a richer sound experience
                        hamming=True, # A Hamming window is used to reduce spectral leakage in signal processing. It is generally applied to improve the quality of generated sounds by smoothing the edges.
                        name='audio'
                        )
        audio.setVolume(1.0)
        audio_duration = audio.duration
        audios[f'audio_{audioNum}'] = {'audio': audio, 'duration': audio_duration}
    # sampled_audios = random.choices(all_audios, k=len(catch_points))
    while len(all_audios) < len(catch_points):
        all_audios += all_audios  # Duplicate the list until it's long enough
    # Trim the extended list to exactly 35 items
    extended_audios = all_audios[:len(catch_points)]
    random.shuffle(extended_audios)
    sampled_audios = extended_audios

    # Videos for catch trials
    all_videos = ["1","2","3","4","5","6","7","11","12","13","14","15","16"]
    movies = {} # Initialize the 'movies' dictionary
    for videoNum in all_videos: # loop through all the numbers
        video_file_path = _thisDir + os.sep + f'stimuli/catch/video/{videoNum}.mp4'
        movie = visual.MovieStim(win, 
                                video_file_path, 
                                size=(960, 540),
                                flipVert=False, # flips the video vertically
                                flipHoriz=False, # flips the viode horizontally
                                loop=False, # plays the video once and stops
                                noAudio=True,
                                )
        movie_duration = movie.duration # get the duration
        movies[f'movie_{videoNum}'] = {'movie': movie, 'duration': movie_duration} # create the dictionary for the list of videos
        # Extend the list manually if needed
    while len(all_videos) < len(catch_points):
        all_videos += all_videos  # Duplicate the list until it's long enough

    # Trim the extended list to exactly 35 items
    extended_videos = all_videos[:len(catch_points)]
    random.shuffle(extended_videos)
    sampled_videos = extended_videos
    # sampled_videos = random.choices(all_videos, k=len(catch_points)) # based on the number of catch trial, chose the videos randoemely


    # Image for pause
    image_path = _thisDir + os.sep + f'stimuli/catch/pause.png'  # Replace with the path to your image
    image_pause = visual.ImageStim(win, image=image_path)

    # ~~~~~~~~~~~~~~~ Create some handy timers
    globalClock = core.Clock()  # to track the time since experiment started, Start Timing: Automatically starts from the moment it is created.
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
                                # Start Timing: Use routineTimer.reset() to start or reset the timer. Set Duration: Use routineTimer.add(time) to set the remaining time for a routine. Check Time: Use routineTimer.getTime() to check the remaining time.


    # ~~~~~~~~~~~~~~~ Prepare to start Routine "welcomescreen" 
    continueRoutine = True # A flag to control whether the routine should continue running.
    welcomescreenComponents = [text_welcome] # Defines a list of components that will be used in the "welcomescreen" routine
    for thisComponent in welcomescreenComponents:
        thisComponent.tStart = None # Initializes the start time of the component to None. This will be updated when the component actually starts.
        thisComponent.tStop = None # Initializes the stop time of the component to None. This will be updated when the component stops.
        thisComponent.tStartRefresh = None # This is used to track when the component is drawn on the screen.
        thisComponent.tStopRefresh = None # This is used to track when the component stops being drawn on the screen.
        if hasattr(thisComponent, 'status'): # Checks if the component has a status attribute. If it does, it initializes the status of the component to NOT_STARTED. This status will be updated as the routine progresses.
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0 # Initializes a local timer variable t to 0. This is used to keep track of the time within the routine.
    _timeToFirstFrame = win.getFutureFlipTime(clock="now") # Gets the time to the next flip of the window, which helps in synchronizing the routine with the display refresh rate. This value is used to determine the precise timing of when the routine starts.
    frameN = -1 # Initializes the frame number variable frameN to -1. This variable will be incremented with each frame of the routine and helps track the frame number.



    # ~~~~~~~~~~~~~~~ Run Routine "welcomescreen"
    routineForceEnded = not continueRoutine # if the continueRoutine is False, which indicate the routine ended, so routineForceEnded = True
    while continueRoutine: # and routineTimer.getTime() < 1.0:
        t = routineTimer.getTime() # get the time sicne the routine started
        tThisFlip = win.getFutureFlipTime(clock=routineTimer) # Gets the time until the next screen refresh relative to the routine timer.
        tThisFlipGlobal = win.getFutureFlipTime(clock=None) # Gets the time until the next screen refresh relative to the global clock.
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        
        # if text_welcome is starting this frame...
        if text_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_welcome.frameNStart = frameN  # exact frame index
            text_welcome.tStart = t  # local t and not account for scr refresh
            text_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_welcome, 'tStartRefresh')  # time at next scr refresh
            thisExp.timestampOnFlip(win, 'text_welcome.started') # add timestamp
            text_welcome.status = STARTED # update status
            text_welcome.setAutoDraw(True) # draw the text on the screen
        
        # if text_welcome is active this frame...
        if text_welcome.status == STARTED:
            pass # pass statement is a placeholder that does nothing
        
        # if text_welcome is stopping this frame...
        if text_welcome.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            # Check for "space" key to end the routine
            keys = defaultKeyboard.getKeys()
            if 'space' in keys:  # If the space key is pressed
                continueRoutine = False  # End the routine
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break # Exits the while loop that is running the routine
        continueRoutine = False  
        for thisComponent in welcomescreenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED: # if its status is not FINISHED
                continueRoutine = True
                break 
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()


    # ~~~~~~~~~~~~~~~ Ending Routine "welcomescreen"
    for thisComponent in welcomescreenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False) # the stimuli are no longer visible
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset() # Resets the routine timer to zero 



    # ~~~~~~~~~~~~~~~ Prepare to start Routine "trial" 
    # This is an object in PsychoPy that manages the presentation of trial stimuli and conditions. It allows for looping over multiple trials, randomizing order, and repeating trials.
    isi = 0.1
    _trialList =  [   
        {'condition': '1_con_duration', 'single_duration': 0.62, 'numerosity': 1},
        {'condition': '2_con_duration', 'single_duration': 0.26, 'numerosity': 2},
        {'condition': '3_con_duration', 'single_duration': 0.14, 'numerosity': 3},
        {'condition': '4_con_duration', 'single_duration': 0.08, 'numerosity': 4},
        {'condition': '5_con_duration', 'single_duration': 0.044, 'numerosity': 5},
        {'condition': '6_con_duration', 'single_duration': 0.02, 'numerosity': 6},
        {'condition': '1_con_rate', 'single_duration': 0.02, 'numerosity': 1},
        {'condition': '2_con_rate', 'single_duration': 0.02, 'numerosity': 2},
        {'condition': '3_con_rate', 'single_duration': 0.02, 'numerosity': 3},
        {'condition': '4_con_rate', 'single_duration': 0.02, 'numerosity': 4},
        {'condition': '5_con_rate', 'single_duration': 0.02, 'numerosity': 5},
        {'condition': '6_con_rate', 'single_duration': 0.02, 'numerosity': 6},
        ]
    trials = data.TrialHandler(nReps=35, # 35 for one day = 10min
                            method='random', # randmise trials within 1 repetition
                            extraInfo=expInfo, # such as participant ID can be stored with the data 
                            originPath=-1,# This is used internally by PsychoPy to store the original path of the script. Setting it to -1 allows PsychoPy to track where the script originated.
                            trialList=_trialList, # creates a list of dictionaries where each dictionary corresponds to a row in the Excel file, representing a unique trial.
                            seed=None, # None means that the randomization will vary each time the script is run.
                            name='trials')
    n_trials = len(_trialList) * trials.nReps

    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None: # If a trial exists, it proceeds to the loop
        for paramName in thisTrial: # loop through all the keys
            exec('{} = thisTrial[paramName]'.format(paramName)) # it constructs a string like condition = thisTrial['condition'], single_duration = thisTrial['single_duration']

    for trial_index, thisTrial in enumerate(trials): # thisTrial refers to each individual trial
        currentLoop = trials # want to track which loop is currently active
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                exec('{} = thisTrial[paramName]'.format(paramName)) # this creats local variables that refers to each trial parameter directly.
        
        # Extract 'condition' from thisTrial if it exists
        condition = thisTrial.get('condition', None)  # Use thisTrial.get() to safely access 'condition'
        numerosity = thisTrial.get('numerosity', None) 
        single_duration = thisTrial.get('single_duration', None) 

        # ~~~~~~~~~~~~~~ make the trigger index       
        if 'duration' in condition: 
            idx_trigger = 0 + numerosity # duration is controlled
        else:
            idx_trigger = 100 + numerosity # rate is controlled 
        

        # ~~~~~~~~~~~~~~~ Update trial number text
        trial_text.text = f'T{trial_index}'

        # ~~~~~~~~~~~~~~~ Prepare to start Routine "trial"
        continueRoutine = True

        # ~~~~~~~~~~~~~~~　modify the single item duration depending on the condition
        beep.setSound(value=beep, secs=single_duration)
        beep.setVolume(1.0, log=False)
        beep_loop = numerosity
    

        #　~~~~~~~~~~~~~~~ Numerosity lOOP start, but if it's 1, no loop
        for i in range(beep_loop):
            trialComponents = [beep] # list all invlved in trials

            for thisComponent in trialComponents:
                thisComponent.tStart = None # The time the component started relative to the trial start.
                thisComponent.tStop = None # The time the component stopped relative to the trial start.
                thisComponent.tStartRefresh = None # The exact time (global time) when the component started (when the screen refreshed).
                thisComponent.tStopRefresh = None # The exact time (global time) when the component stopped (when the screen refreshed).
                if hasattr(thisComponent, 'status'): #The hasattr() method returns true if an object has the 'status' attribute and false if it does not.
                    thisComponent.status = NOT_STARTED

            t = 0 # reset the local time
            _timeToFirstFrame = win.getFutureFlipTime(clock="now") # Prepares to track when the first frame of the trial will be drawn
            frameN = -1 # This variable keeps track of the number of frames
            
            # ~~~~~~~~~~~~~~~　Run Routine "trial"
            continueRoutine = True
            while continueRoutine: # and routineTimer.getTime() < answer_time:
                t = routineTimer.getTime() # captures the current time
                tThisFlip = win.getFutureFlipTime(clock=routineTimer) # It ensures that drawing stimuli or updating the screen happens right on the next refresh for precise timing.
                tThisFlipGlobal = win.getFutureFlipTime(clock=None) # It's often used for timestamping events that need to be logged with respect to the overall experiment timeline.
                frameN = frameN + 1  # It starts from -1 (initialized earlier) and increments by 1 with each iteration of the loop.

                # show the diagonal lines on the screen
                line_diagonal_1.draw()
                line_diagonal_2.draw()


                # if beep is starting this fram...
                if beep.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance: # If both conditions are met, it means the beep should start this frame.
                    beep.framNstart = frameN # record the frame number when beep starts
                    beep.tStart = t # record the local trial time when beep starts
                    beep.tStartRefresh = tThisFlipGlobal # get the global time when the beep is presented
                    thisExp.addData(f'beep_{i+1}.started', tThisFlipGlobal)  # write in the datafile
                    if EEG_trigger == True:
                        win.callOnFlip(p_port.setData, idx_trigger)
                    # update status
                    beep.status = STARTED
                    beep.play(when=win)  # sync with win flip
                    

                    trial_text.setAutoDraw(True)


                # if beep is stopping this frame...
                if beep.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > beep.tStartRefresh + (single_duration - frameTolerance): # if the current global time exceeds (tStartRefresh + single_duration) tStartRefresh is the globa time when beep started playing
                        beep.tStop = t  # records the local trial time
                        beep.frameNStop = frameN  # Records the frame number when beep ends
                        beep.tStopRefresh = tThisFlipGlobal
                        thisExp.addData(f'beep_{i+1}.stopped', beep.tStopRefresh)  # add timestamp to datafile
                        # update status
                        beep.status = FINISHED
                        beep.stop()
                        trial_text.setAutoDraw(False)

                        if EEG_trigger == True:
                            p_port.setData(0) # refresh the trigger

                        if i == beep_loop - 1:
                            # Add 800ms silent period after the beep stops
                            silent_duration = 0.800  # 800ms silent period
                            core.wait(silent_duration)  # wait for 800ms before continuing
                        else:
                            core.wait(isi)

                # Pause logic - check if 'p' key is pressed
                keys = defaultKeyboard.getKeys(keyList=['p', 'r']) # they key press is detected immediately. 
                if 'p' in [key.name for key in keys]: # list the keys pressed
                    # Pause the routine
                    while True: # continuously checks for 'r' key
                        image_pause.draw()
                        win.flip()
                        # Check for 'r' to resume
                        keys_resume = defaultKeyboard.getKeys(keyList=['r'])
                        if 'r' in [key.name for key in keys_resume]:
                            break # once 'r' is pressed, break out the loop
                        core.wait(0.1) 
                            
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit() #  terminates the program and saves any data collected up to that point.
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # if continueRoutine is False, the routine ends, and the screen won't be flipped.
                    win.flip()
        #　~~~~~~~~~~~~~~~ Numerosity lOOP end, but if it's 1, no loop  
        
        # insert catch trials
        if trial_index in catch_points:  
            start_time = core.getTime()
            # prepare an audio
            audio_to_show = audios[f'audio_{sampled_audios[ii]}']['audio']
            audio_name = f'{sampled_audios[ii]}.wav'
            thisExp.addData('audio_name', audio_name)  # Store the movie name in the CSV
            thisExp.addData('audio_started', start_time) # store the time started
            audio_to_show.seek(1)
            
            
            # prepare a movie
            movie_duration = movies[f'movie_{sampled_videos[ii]}']['duration']
            movie_to_show = movies[f'movie_{sampled_videos[ii]}']['movie']
            movie_name = f'{sampled_videos[ii]}.mp4'
            movie_to_show.stop()
            # if trial_index < n_trials/2:
                # movie_to_show.seek((movie_duration/2) - 1)
            # else:
                # movie_to_show.seek((movie_duration/2) + 1)
            movie_to_show.seek(movie_duration/2)

            movie_to_show.play()
            audio_to_show.play(when=win)
            
            thisExp.addData('movie_name', movie_name)  # Store the movie name in the CSV
            thisExp.addData('movie_started', start_time) # store the time started

            # Play the video for only 1 second
            while core.getTime() - start_time < 4.0:  # 1 second duration
                line_diagonal_1.draw() # show the diagonal lines on the screen
                line_diagonal_2.draw()
                movie_to_show.draw()  # Draw the current frame of the video
                if EEG_trigger == True:
                    win.callOnFlip(p_port.setData, 50 + int(sampled_videos[ii])) # set the trigger based on the video
                win.flip()
                
            audio_to_show.stop()
            movie_to_show.pause()
            
            if EEG_trigger == True:
                p_port.setData(0) # refresh the trigger
            ii += 1 # updates the index
            # Show the whole background with lines after movie pauses
            line_diagonal_1.draw()
            line_diagonal_2.draw()
            win.flip()
            # Add 800ms silence after the video finishes
            core.wait(0.8)

        # ~~~~~~~~~~~~~~~ Ending Routine "trial" 
        beep.stop()  # ensure sound has stopped at end of routine

        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()

        thisExp.nextEntry() # Advances to the next trial in the experiment. This saves the current trial's data and prepares for the next one.

        # ~~~~~~~~~~~~~~~ completed 1 run (12 conditions) (20~620ms + 800ms)

    # completed whole trials (10min)

    # ~~~~~~~~~~~~~~~ End experiment
    win.flip() # ensues that the final display is shown on the screen.
    thisExp.saveAsWideText(filename+'.csv', delim='auto') # save as csv.file, delimiter used in the CSV file is appropriate
    thisExp.saveAsPickle(filename) # Saves the experiment data as a binary file (a "pickle"). Pickle files are useful because they preserve the Python objects and structure of the experiment data.
    logging.flush() # Ensures that any logs generated during the experiment (e.g., timestamps, warnings, errors) are written to the log file. This prevents loss of log data if something crashes or closes abruptly.
    thisExp.abort() # Signals that the experiment has ended
    win.close() # Closes the PsychoPy window
    core.quit() # Exits the entire PsychoPy script



# audio end
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visual 




def run_visual(monitor_data = [1920, 1080, 50, 90], EEG_trigger = True):

    # ~~~~~~~~~~~~~~~ Import packages
    from psychopy import locale_setup
    from psychopy import prefs
    from psychopy import plugins
    plugins.activatePlugins()
    prefs.hardware['audioLib'] = 'ptb'
    prefs.hardware['audioLatencyMode'] = '3'
    from psychopy import sound, gui, parallel, visual, core, data, event, logging, clock, colors, layout, monitors
    from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                    STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

    import numpy as np  # whole numpy lib is available, prepend 'np.'
    from numpy import (sin, cos, tan, log, log10, pi, average,
                    sqrt, std, deg2rad, rad2deg, linspace, asarray)
    from numpy.random import random, randint, normal, shuffle, choice as randchoice
    import os  # handy system and path functions
    import sys  # to get file system encoding
    import random
    import time
    import re
    import psychopy.iohub as io
    from psychopy.hardware import keyboard



    # ~~~~~~~~~~~~~~~ Ensure that relative paths start from the same directory as this script
    _thisDir = os.path.dirname(os.path.abspath(__file__)) # '__file__' indicate the current directory of this script
    os.chdir(_thisDir) # change the current working directory to the directory specified by '_thisdir'
    # Store info about the experiment session
    psychopyVersion = '2024.1.5'
    expName = 'visual'  # from the Builder filename that created this script
    expInfo = {
        'participant': '',
        'session': '1',
        'date|hid': data.getDateStr(), # "|hid" indicate this info is hidden in the GUI
        'expName|hid': expName,
        'psychopyVersion|hid': psychopyVersion,
        }

    # ~~~~~~~~~~~~~~~ Show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel

    # define csv file name 
    filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date|hid']) # 'os.sep' is a system independent separator for both window and mac 

    # An ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, 
        version='',
        extraInfo=expInfo, 
        runtimeInfo=None,
        originPath='',
        savePickle=True, # saved in a binary pickle file (.psydat)
        saveWideText=True, # saved in a wide-format text file, usually a CSV file
        dataFileName=filename
        )

    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP) # This creates a log file to store detailed information about the experiment’s execution, such as events, errors, and system messages.
    logging.console.setLevel(logging.WARNING)  # only warning messages and errors will be printed to the screen (console)

    endExpNow = False  # whether the experiment should be terminated immediately, You can later set this flag to True in response to certain conditions
    frameTolerance = 0.001  # how close to onset before 'same' frame



    # ~~~~~~~~~~~~~~~ Setup the Window
    Monitor_data = monitor_data # my M1 window = [1440, 900, 50, 50] # [1920, 1080, 50, 50]  The thrid value indicate the width of screen, the fourth value indicates the distance between eyes and screen
    mon = monitors.Monitor(
        'stimulus_screen', width=Monitor_data[2], distance=Monitor_data[3]) # width (cm) is to calcurate the visual angle, distance (cm) is viewing distance.
    mon.setSizePix((Monitor_data[0], Monitor_data[1]))
    mon.save()

    _size = (Monitor_data[0], Monitor_data[1]) # size (resolution) of the window in pixels, e.g., [1920, 1080] for Full HD.
    _fullscr = True # whether the window is displayed in fullscreen mode
    _screen = 0 # which monitor to display
    _winType = 'pyglet' # Specifies the underlying graphics library to use for window management.
    _allowGUI=False # stencil buffer is allowed. The stencil buffer is used for more advanced graphical effects, such as masking.
    _monitor='stimulus_screen' # Refers to a monitor profile defined in PsychoPy
    _color=[0, 0, 0] # RGB color code
    _colorSpace='rgb' # Specifies the color space
    _backgroundImage='' # Allows for setting a background image to be displayed behind any stimuli
    _backgroundFit='none' # Specifies how the background image should fit to the window if provided
    _blendMode='avg' # Controls how overlapping stimuli are blended together on the screen. 'avg' means averaging the pixel values of overlapping stimuli.
    _useFBO=True # Setting useFBO=True allows PsychoPy to render stimuli off-screen (in a buffer) before displaying them. This can improve rendering flexibility and performance, especially when dealing with complex stimuli or advanced visual effects.
    _units='norm' # Defines the units used for positioning and sizing stimuli in the window. units='height' sets the unit of measurement relative to the height of the window. For example, setting a stimulus' size to 0.5 means it will occupy half of the window’s height. Other possible units include norm, pix (pixels), and cm (centimeters).

    win = visual.Window(
        size=_size, 
        fullscr=_fullscr, 
        screen=_screen, 
        winType=_winType, 
        allowGUI=_allowGUI,
        monitor=_monitor,
        color=_color, 
        colorSpace=_colorSpace,
        backgroundImage=_backgroundImage, 
        backgroundFit=_backgroundFit,
        blendMode=_blendMode, 
        useFBO=_useFBO, 
        units=_units
        )
    win.mouseVisible = False # whether mouse cursor is visible or not.

    # store frame rate of monitor if we can measure it
    expInfo['frameRate'] = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...') # determining how frequently the window refreshes (measured in frames per second (FPS))
    if expInfo['frameRate'] != None: # if it is not None
        frameDur = 1.0 / round(expInfo['frameRate']) # For example, if the frame rate is 60 Hz (60 frames per second), then the duration of one frame is 1.0 / 60 = 0.01667 seconds (about 16.67 milliseconds per frame).
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess



    # ~~~~~~~~~~~~~~~ Setup input devices
    ioConfig = {}
    defaultKeyboard = keyboard.Keyboard()

    # add device to the dictionary
    ioConfig['keyboard'] = defaultKeyboard

    # ~~~~~~~~~~~~~~ Parallel port
    if EEG_trigger == True:
        p_port = parallel.ParallelPort(address='0x0378') # or '0x03BC' # This is typically the standard address for LPT1, the first parallel port on many older PCs
        p_port.setData(0) # 0 in this case means all 8 data pins are set to low. In binary, 0 is represented as 00000000, so each pin gets set to 0 (off).
        # Function to send a trigger
        def send_trigger(trigger_value):
            p_port.setData(trigger_value) # Send the trigger value
            time.sleep(0.001) # Wait for a short period to ensure the trigger is registered (e.g., 1 ms) 
            p_port.setData(0) # Reset the parallel port to 0


    # ~~~~~~~~~~~~~~~ Components for Routine "welcomescreen" 
    text_welcome = visual.TextStim(win=win, 
                                name='text_welcome', # The name of this text stimulus
                                text='When you are ready, press "space".', # The actual text to be displayed on the screen
                                font='Open Sans',
                                pos=(0, 0), # (0, 0) indicates that the text will be centered at the origin of the window,
                                height=1, # The height of the text, which determines how large the characters will appear. The unit of height is defined relative to the window size (e.g., 0.05 would be 5% of the window height if you're using the 'height' unit for the window
                                wrapWidth=None, # This parameter controls text wrapping
                                ori=0.0, # The orientation of the text
                                color=[1.0, 1.0, 1.0], 
                                colorSpace='rgb', 
                                opacity=None, # The opacity of the text
                                languageStyle='LTR', # Defines the direction of the text. 'LTR' stands for Left-to-Right,
                                depth=0.0,
                                units='cm'
                                )



    # ~~~~~~~~~~~~~~~ components for Routine "trial" ---

    # Images for trials
    image_file_path = os.path.join(_thisDir,'stimuli', 'visual')
    all_condition_folders = [f for f in os.listdir(image_file_path) if os.path.isdir(os.path.join(image_file_path, f))] # Get list of folders (directories) only,  # number of condition, e.g., total size controlled, single size controlled
    images = {}
    # condition loop (dot, total_dot, circumference)
    for condition_name in all_condition_folders:
        stimuli_folder = os.path.join(_thisDir, 'stimuli', 'visual', condition_name)
        all_numerosity_folders = [f for f in os.listdir(stimuli_folder) if os.path.isdir(os.path.join(stimuli_folder, f))]
        images[condition_name] = []

        # numerosity loop (1 ~ 6)
        for numerosity in all_numerosity_folders:
            each_numerosity_folder = os.path.join(stimuli_folder, numerosity)
            each_image_name = [f for f in os.listdir(each_numerosity_folder) if os.path.isfile(os.path.join(each_numerosity_folder, f)) and not f.startswith("._") ]

            # image loop (1 ~ 70)
            for image_name in each_image_name:
                # image = visual.ImageStim(win, image= os.path.join(each_numerosity_folder, image_name))
                images[condition_name].append({
                                                'image': os.path.join(each_numerosity_folder, image_name), 
                                                'numerosity': numerosity, 
                                                'name': image_name
                                                })



    # specify the trial just before the catch trial
    catch_points = list(range(17, 1260, 18)) # insert the catch video after 18 trials and till 1260, stepsize is 18.
    i = 0 # catch_points_idx

    # Audio for catch trials
    all_audios = ["1","2","3","4"]
    audios = {}
    for audioNum in all_audios:
        audio_file_path = _thisDir + os.sep + f'stimuli/catch/audio/{audioNum}.wav'
        audio = sound.Sound(value=audio_file_path, 
                        stereo=True, #  This indicates that the sound should be played in stereo mode, which involves two channels (left and right) to provide a richer sound experience
                        hamming=True, # A Hamming window is used to reduce spectral leakage in signal processing. It is generally applied to improve the quality of generated sounds by smoothing the edges.
                        name='audio'
                        )
        audio.setVolume(1.0)
        audio_duration = audio.duration
        audios[f'audio_{audioNum}'] = {'audio': audio, 'duration': audio_duration}
    # sampled_audios = random.choices(all_audios, k=len(catch_points))
    while len(all_audios) < len(catch_points):
        all_audios += all_audios  # Duplicate the list until it's long enough
    # Trim the extended list to exactly 35 items
    extended_audios = all_audios[:len(catch_points)]
    random.shuffle(extended_audios)
    sampled_audios = extended_audios

    # Videos for catch trials
    all_videos = ["1","2","3","4","5","6","7","11","12","13","14","15","16"]
    movies = {} # Initialize the 'movies' dictionary
    for videoNum in all_videos: # loop through all the numbers
        video_file_path = _thisDir + os.sep + f'stimuli/catch/video/{videoNum}.mp4'
        movie = visual.MovieStim(win, 
                                video_file_path, 
                                size=(960, 540),
                                flipVert=False, # flips the video vertically
                                flipHoriz=False, # flips the viode horizontally
                                loop=False, # plays the video once and stops
                                noAudio=True
                                )
        movie_duration = movie.duration # get the duration
        movies[f'movie_{videoNum}'] = {'movie': movie, 'duration': movie_duration} # create the dictionary for the list of videos
    while len(all_videos) < len(catch_points):
        all_videos += all_videos  # Duplicate the list until it's long enough
    # Trim the extended list to exactly 35 items
    extended_videos = all_videos[:len(catch_points)]
    random.shuffle(extended_videos)
    sampled_videos = extended_videos

    # Image for pause
    image_path = _thisDir + os.sep + f'stimuli/catch/pause.png'  # Replace with the path to your image
    image_pause = visual.ImageStim(win, image=image_path)

    # Create a text stimulus for the trial number
    trial_text = visual.TextStim(win=win, text='', pos=(-0.97, -0.97), height=0.04, color='black', alignText='center')


    # Image for no dot time. 
    background_path = _thisDir + os.sep + f'stimuli/visual/background.png'  # Replace with the path to your image
    background_image = visual.ImageStim(win, image=background_path)


    # ~~~~~~~~~~~~~~~ Create some handy timers
    globalClock = core.Clock()  # to track the time since experiment started, Start Timing: Automatically starts from the moment it is created.
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
                                # Start Timing: Use routineTimer.reset() to start or reset the timer. Set Duration: Use routineTimer.add(time) to set the remaining time for a routine. Check Time: Use routineTimer.getTime() to check the remaining time.


    # ~~~~~~~~~~~~~~~ Prepare to start Routine "welcomescreen" 
    continueRoutine = True # A flag to control whether the routine should continue running.
    welcomescreenComponents = [text_welcome] # Defines a list of components that will be used in the "welcomescreen" routine
    for thisComponent in welcomescreenComponents:
        thisComponent.tStart = None # Initializes the start time of the component to None. This will be updated when the component actually starts.
        thisComponent.tStop = None # Initializes the stop time of the component to None. This will be updated when the component stops.
        thisComponent.tStartRefresh = None # This is used to track when the component is drawn on the screen.
        thisComponent.tStopRefresh = None # This is used to track when the component stops being drawn on the screen.
        if hasattr(thisComponent, 'status'): # Checks if the component has a status attribute. If it does, it initializes the status of the component to NOT_STARTED. This status will be updated as the routine progresses.
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0 # Initializes a local timer variable t to 0. This is used to keep track of the time within the routine.
    _timeToFirstFrame = win.getFutureFlipTime(clock="now") # Gets the time to the next flip of the window, which helps in synchronizing the routine with the display refresh rate. This value is used to determine the precise timing of when the routine starts.
    frameN = -1 # Initializes the frame number variable frameN to -1. This variable will be incremented with each frame of the routine and helps track the frame number.



    # ~~~~~~~~~~~~~~~ Run Routine "welcomescreen"
    routineForceEnded = not continueRoutine # if the continueRoutine is False, which indicate the routine ended, so routineForceEnded = True
    while continueRoutine: # and routineTimer.getTime() < 1.0:
        t = routineTimer.getTime() # get the time sicne the routine started
        tThisFlip = win.getFutureFlipTime(clock=routineTimer) # Gets the time until the next screen refresh relative to the routine timer.
        tThisFlipGlobal = win.getFutureFlipTime(clock=None) # Gets the time until the next screen refresh relative to the global clock.
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        
        # if text_welcome is starting this frame...
        if text_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_welcome.frameNStart = frameN  # exact frame index
            text_welcome.tStart = t  # local t and not account for scr refresh
            text_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_welcome, 'tStartRefresh')  # time at next scr refresh
            thisExp.timestampOnFlip(win, 'text_welcome.started') # add timestamp
            text_welcome.status = STARTED # update status
            text_welcome.setAutoDraw(True) # draw the text on the screen
        
        # if text_welcome is active this frame...
        if text_welcome.status == STARTED:
            pass # pass statement is a placeholder that does nothing
        
        # if text_welcome is stopping this frame...
        if text_welcome.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            # Check for "space" key to end the routine
            keys = defaultKeyboard.getKeys()
            if 'space' in keys:  # If the space key is pressed
                continueRoutine = False  # End the routine
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break # Exits the while loop that is running the routine
        continueRoutine = False  
        for thisComponent in welcomescreenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED: # if its status is not FINISHED
                continueRoutine = True
                break 
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()


    # ~~~~~~~~~~~~~~~ Ending Routine "welcomescreen"
    for thisComponent in welcomescreenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False) # the stimuli are no longer visible
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset() # Resets the routine timer to zero 



    # ~~~~~~~~~~~~~~~ Prepare to start Routine "trial" 
    # This is an object in PsychoPy that manages the presentation of trial stimuli and conditions. It allows for looping over multiple trials, randomizing order, and repeating trials.
    isi = 0.1
    duration = 0.25
    _trialList =  [   
        {'condition': '1_con_singledot', 'duration': duration, 'numerosity': 1},
        {'condition': '2_con_singledot', 'duration': duration, 'numerosity': 2},
        {'condition': '3_con_singledot', 'duration': duration, 'numerosity': 3},
        {'condition': '4_con_singledot', 'duration': duration, 'numerosity': 4},
        {'condition': '5_con_singledot', 'duration': duration, 'numerosity': 5},
        {'condition': '6_con_singledot', 'duration': duration, 'numerosity': 6},
        {'condition': '1_con_totaldot', 'duration': duration, 'numerosity': 1},
        {'condition': '2_con_totaldot', 'duration': duration, 'numerosity': 2},
        {'condition': '3_con_totaldot', 'duration': duration, 'numerosity': 3},
        {'condition': '4_con_totaldot', 'duration': duration, 'numerosity': 4},
        {'condition': '5_con_totaldot', 'duration': duration, 'numerosity': 5},
        {'condition': '6_con_totaldot', 'duration': duration, 'numerosity': 6},
        {'condition': '1_con_circum', 'duration': duration, 'numerosity': 1},
        {'condition': '2_con_circum', 'duration': duration, 'numerosity': 2},
        {'condition': '3_con_circum', 'duration': duration, 'numerosity': 3},
        {'condition': '4_con_circum', 'duration': duration, 'numerosity': 4},
        {'condition': '5_con_circum', 'duration': duration, 'numerosity': 5},
        {'condition': '6_con_circum', 'duration': duration, 'numerosity': 6},
        ]
    trials = data.TrialHandler(nReps=70, # 70  = 10 min
                            method='random', # randmise trials within 1 repetition
                            extraInfo=expInfo, # such as participant ID can be stored with the data 
                            originPath=-1,# This is used internally by PsychoPy to store the original path of the script. Setting it to -1 allows PsychoPy to track where the script originated.
                            trialList=_trialList, # creates a list of dictionaries where each dictionary corresponds to a row in the Excel file, representing a unique trial.
                            seed=None, # None means that the randomization will vary each time the script is run.
                            name='trials')
    n_trials = len(_trialList) * trials.nReps 

    # trial count for each condition
    singledot_counter_1 = 0
    singledot_counter_2 = 0
    singledot_counter_3 = 0
    singledot_counter_4 = 0
    singledot_counter_5 = 0
    singledot_counter_6 = 0

    totatldot_counter_1 = 0
    totatldot_counter_2 = 0
    totatldot_counter_3 = 0
    totatldot_counter_4 = 0
    totatldot_counter_5 = 0
    totatldot_counter_6 = 0

    cicumference_counter_1 = 0
    cicumference_counter_2 = 0
    cicumference_counter_3 = 0
    cicumference_counter_4 = 0
    cicumference_counter_5 = 0
    cicumference_counter_6 = 0



    # ~~~~~~~~~~~~ Make the list of trials for each condition
    target_condition = 'singledotsize_cont'
    singledot_images_by_numerosity = {}
    for numerosity in range(1, 7):  # Range from 1 to 6 (inclusive)
        numerosity_key = str(numerosity)  # Convert the numerosity to string if stored as string
        target_images = []
        # Iterate through all images in the target condition
        for image_data in images[target_condition]:
            # If the numerosity matches, add the image data to the target list
            if image_data['numerosity'] == f'numerosity_{numerosity_key}':
                target_images.append(image_data)
        # randomize the order
        random.shuffle(target_images)
        # Store the images for the current numerosity in the dictionary
        singledot_images_by_numerosity[numerosity_key] = target_images

    target_condition = 'totaldotsize_cont'
    totaldot_images_by_numerosity = {}
    for numerosity in range(1, 7):  # Range from 1 to 6 (inclusive)
        numerosity_key = str(numerosity)  # Convert the numerosity to string if stored as string
        target_images = []
        # Iterate through all images in the target condition
        for image_data in images[target_condition]:
            # If the numerosity matches, add the image data to the target list
            if image_data['numerosity'] == f'numerosity_{numerosity_key}':
                target_images.append(image_data)
        # randomize the order
        random.shuffle(target_images)
        # Store the images for the current numerosity in the dictionary
        totaldot_images_by_numerosity[numerosity_key] = target_images

    target_condition = 'circumference_cont'
    circumference_images_by_numerosity = {}
    for numerosity in range(1, 7):  # Range from 1 to 6 (inclusive)
        numerosity_key = str(numerosity)  # Convert the numerosity to string if stored as string
        target_images = []
        # Iterate through all images in the target condition
        for image_data in images[target_condition]:
            # If the numerosity matches, add the image data to the target list
            if image_data['numerosity'] == f'numerosity_{numerosity_key}':
                target_images.append(image_data)
        # randomize the order
        random.shuffle(target_images)
        # Store the images for the current numerosity in the dictionary
        circumference_images_by_numerosity[numerosity_key] = target_images



    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None: # If a trial exists, it proceeds to the loop
        for paramName in thisTrial: # loop through all the keys
            exec('{} = thisTrial[paramName]'.format(paramName)) # it constructs a string like condition = thisTrial['condition'], duration = thisTrial['duration']

    for trial_index, thisTrial in enumerate(trials): # thisTrial refers to each individual trial
        currentLoop = trials # want to track which loop is currently active
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                exec('{} = thisTrial[paramName]'.format(paramName)) # this creats local variables that refers to each trial parameter directly.
            # Extract 'condition' from thisTrial if it exists
            condition = thisTrial.get('condition', None)  # Use thisTrial.get() to safely access 'condition'
            numerosity = thisTrial.get('numerosity', None) 
            duration = thisTrial.get('duration', None)

        # ~~~~~~~~~~~~~~ make the trigger index and chose 1 stimuli from the specific condition 
        # ~~~~~~~ single dot size is controlled
        if 'singledot' in condition: 
            idx_trigger = 0 + numerosity # singledot is controlled

            if numerosity == 1:
                image_to_show = visual.ImageStim(win, image=singledot_images_by_numerosity[str(numerosity)][singledot_counter_1]['image'])
                image_to_show_name = singledot_images_by_numerosity[str(numerosity)][singledot_counter_1]['name']
                singledot_counter_1 += 1 # add 1 to the counter
            elif numerosity == 2:
                image_to_show = visual.ImageStim(win, image=singledot_images_by_numerosity[str(numerosity)][singledot_counter_2]['image'])
                image_to_show_name = singledot_images_by_numerosity[str(numerosity)][singledot_counter_2]['name']
                singledot_counter_2 += 1
            elif numerosity == 3:
                image_to_show = visual.ImageStim(win, image=singledot_images_by_numerosity[str(numerosity)][singledot_counter_3]['image'])
                image_to_show_name = singledot_images_by_numerosity[str(numerosity)][singledot_counter_3]['name']
                singledot_counter_3 += 1
            elif numerosity == 4:
                image_to_show = visual.ImageStim(win, image=singledot_images_by_numerosity[str(numerosity)][singledot_counter_4]['image'])
                image_to_show_name = singledot_images_by_numerosity[str(numerosity)][singledot_counter_4]['name']
                singledot_counter_4 += 1
            elif numerosity == 5:
                image_to_show = visual.ImageStim(win, image=singledot_images_by_numerosity[str(numerosity)][singledot_counter_5]['image'])
                image_to_show_name = singledot_images_by_numerosity[str(numerosity)][singledot_counter_5]['name']
                singledot_counter_5 += 1
            else:
                image_to_show = visual.ImageStim(win, image=singledot_images_by_numerosity[str(numerosity)][singledot_counter_6]['image'])
                image_to_show_name = singledot_images_by_numerosity[str(numerosity)][singledot_counter_6]['name']
                singledot_counter_6 += 1


        # ~~~~~~~ totaldot size is controlled
        elif 'totaldot' in condition:
            idx_trigger = 100 + numerosity # totaldot is controlled

            if numerosity == 1:
                image_to_show = visual.ImageStim(win, image=totaldot_images_by_numerosity[str(numerosity)][totatldot_counter_1]['image'])
                image_to_show_name = totaldot_images_by_numerosity[str(numerosity)][totatldot_counter_1]['name']
                totatldot_counter_1 += 1 # add 1 to the counter
            elif numerosity == 2:
                image_to_show = visual.ImageStim(win, image=totaldot_images_by_numerosity[str(numerosity)][totatldot_counter_2]['image'])
                image_to_show_name = totaldot_images_by_numerosity[str(numerosity)][totatldot_counter_2]['name']
                totatldot_counter_2 += 1
            elif numerosity == 3:
                image_to_show = visual.ImageStim(win, image=totaldot_images_by_numerosity[str(numerosity)][totatldot_counter_3]['image'])
                image_to_show_name = totaldot_images_by_numerosity[str(numerosity)][totatldot_counter_3]['name']
                totatldot_counter_3 += 1
            elif numerosity == 4:
                image_to_show = visual.ImageStim(win, image=totaldot_images_by_numerosity[str(numerosity)][totatldot_counter_4]['image'])
                image_to_show_name = totaldot_images_by_numerosity[str(numerosity)][totatldot_counter_4]['name']
                totatldot_counter_4 += 1
            elif numerosity == 5:
                image_to_show = visual.ImageStim(win, image=totaldot_images_by_numerosity[str(numerosity)][totatldot_counter_5]['image'])
                image_to_show_name = totaldot_images_by_numerosity[str(numerosity)][totatldot_counter_5]['name']
                totatldot_counter_5 += 1
            else:
                image_to_show = visual.ImageStim(win, image=totaldot_images_by_numerosity[str(numerosity)][totatldot_counter_6]['image'])
                image_to_show_name = totaldot_images_by_numerosity[str(numerosity)][totatldot_counter_6]['name']
                totatldot_counter_6 += 1


        # ~~~~~~~ circumference is controlled
        else: 
            idx_trigger = 200 + numerosity # circumference is controlled

            if numerosity == 1:
                image_to_show = visual.ImageStim(win, image=circumference_images_by_numerosity[str(numerosity)][cicumference_counter_1]['image'])
                image_to_show_name = circumference_images_by_numerosity[str(numerosity)][cicumference_counter_1]['name']
                cicumference_counter_1 += 1 # add 1 to the counter
            elif numerosity == 2:
                image_to_show = visual.ImageStim(win, image=circumference_images_by_numerosity[str(numerosity)][cicumference_counter_2]['image'])
                image_to_show_name = circumference_images_by_numerosity[str(numerosity)][cicumference_counter_2]['name']
                cicumference_counter_2 += 1
            elif numerosity == 3:
                image_to_show = visual.ImageStim(win, image=circumference_images_by_numerosity[str(numerosity)][cicumference_counter_3]['image'])
                image_to_show_name = circumference_images_by_numerosity[str(numerosity)][cicumference_counter_3]['name']
                cicumference_counter_3 += 1
            elif numerosity == 4:
                image_to_show = visual.ImageStim(win, image=circumference_images_by_numerosity[str(numerosity)][cicumference_counter_4]['image'])
                image_to_show_name = circumference_images_by_numerosity[str(numerosity)][cicumference_counter_4]['name']
                cicumference_counter_4 += 1
            elif numerosity == 5:
                image_to_show = visual.ImageStim(win, image=circumference_images_by_numerosity[str(numerosity)][cicumference_counter_5]['image'])
                image_to_show_name = circumference_images_by_numerosity[str(numerosity)][cicumference_counter_5]['name']
                cicumference_counter_5 += 1
            else:
                image_to_show = visual.ImageStim(win, image=circumference_images_by_numerosity[str(numerosity)][cicumference_counter_6]['image'])
                image_to_show_name = circumference_images_by_numerosity[str(numerosity)][cicumference_counter_6]['name']
                cicumference_counter_6 += 1


        image_name = f"{condition}_{image_to_show_name}"
        thisExp.addData('image_name', image_name)  # Store the movie name in the CSV
  
        image_number = int(re.findall(r'\d+', image_to_show_name)[0]) # Use a regular expression to extract the number (e.g., image_to_show_name = image_5)
        thisExp.addData(f'image_number',  image_number)

        # ~~~~~~~~~~~~~~~ Update trial number text
        trial_text.text = f'T{trial_index}'

        # ~~~~~~~~~~~~~~~ Prepare to start Routine "trial"
        continueRoutine = True   

        trialComponents = [image_to_show] # list all invlved in trials

        for thisComponent in trialComponents:
            thisComponent.tStart = None # The time the component started relative to the trial start.
            thisComponent.tStop = None # The time the component stopped relative to the trial start.
            thisComponent.tStartRefresh = None # The exact time (global time) when the component started (when the screen refreshed).
            thisComponent.tStopRefresh = None # The exact time (global time) when the component stopped (when the screen refreshed).
            if hasattr(thisComponent, 'status'): #The hasattr() method returns true if an object has the 'status' attribute and false if it does not.
                thisComponent.status = NOT_STARTED

        t = 0 # reset the local time
        _timeToFirstFrame = win.getFutureFlipTime(clock="now") # Prepares to track when the first frame of the trial will be drawn
        frameN = -1 # This variable keeps track of the number of frames
        
        # ~~~~~~~~~~~~~~~　Run Routine "trial"
        continueRoutine = True
        while continueRoutine: # and routineTimer.getTime() < answer_time:
            t = routineTimer.getTime() # captures the current time
            tThisFlip = win.getFutureFlipTime(clock=routineTimer) # It ensures that drawing stimuli or updating the screen happens right on the next refresh for precise timing.
            tThisFlipGlobal = win.getFutureFlipTime(clock=None) # It's often used for timestamping events that need to be logged with respect to the overall experiment timeline.
            frameN = frameN + 1  # It starts from -1 (initialized earlier) and increments by 1 with each iteration of the loop.

            # if image is starting this fram...
            if image_to_show.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance: # If both conditions are met, it means the beep should start this frame.
                image_to_show.framNstart = frameN # record the frame number when beep starts
                image_to_show.tStart = t # record the local trial time when beep starts
                image_to_show.tStartRefresh = tThisFlipGlobal # get the global time when the beep is presented
                if EEG_trigger == True:
                    win.callOnFlip(p_port.setData, idx_trigger) # send trigger
                image_to_show.setAutoDraw(True)
                thisExp.addData(f'image.started', tThisFlipGlobal)  # write in the datafile
                image_to_show.status = STARTED
                
                trial_text.setAutoDraw(True)


            # if image is stopping this frame...
            if image_to_show.status == STARTED:

                # is it time to stop? (based on global clock, using actual start)
                if image_to_show.status == STARTED and tThisFlipGlobal > image_to_show.tStartRefresh + (duration - frameTolerance): # if the current global time exceeds (tStartRefresh + duration) tStartRefresh is the globa time when beep started playing
                    image_to_show.tStop = t  # records the local trial time
                    image_to_show.frameNStop = frameN  # Records the frame number when beep ends
                    image_to_show.tStopRefresh = tThisFlipGlobal
                    thisExp.addData(f'image.stopped', image_to_show.tStopRefresh)  # add timestamp to datafile
                    # update status
                    image_to_show.setAutoDraw(False)
                    image_to_show.status = FINISHED
                    image_to_show.image = None
                    if EEG_trigger == True:
                        p_port.setData(0) # refresh the trigger

                    background_image.setAutoDraw(True)
                    win.flip()

                    core.wait(isi) # 100ms wait for the next trial

                    background_image.setAutoDraw(False)
                    trial_text.setAutoDraw(False)
            

            # Pause logic - check if 'p' key is pressed
            keys = defaultKeyboard.getKeys(keyList=['p', 'r']) # they key press is detected immediately. 
            if 'p' in [key.name for key in keys]: # list the keys pressed
                # Pause the routine
                while True: # continuously checks for 'r' key
                    image_pause.draw()
                    win.flip()
                    # Check for 'r' to resume
                    keys_resume = defaultKeyboard.getKeys(keyList=['r'])
                    if 'r' in [key.name for key in keys_resume]:
                        break # once 'r' is pressed, break out the loop
                    core.wait(0.1) 

            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit() #  terminates the program and saves any data collected up to that point.
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # if continueRoutine is False, the routine ends, and the screen won't be flipped.
                win.flip()
        #　~~~~~~~~~~~~~~~ 1 trial end 

        # if the 18 trials are done, a catch video should be presented

        # ~~~~~~~~~~~~~~~~ insert catch trials
        if trial_index in catch_points:  
            start_time = core.getTime()
            # prepare an audio
            audio_to_show = audios[f'audio_{sampled_audios[i]}']['audio']
            audio_name = f'{sampled_audios[i]}.wav'
            thisExp.addData('audio_name', audio_name)  # Store the movie name in the CSV
            thisExp.addData('audio_started', start_time) # store the time started
            audio_to_show.seek(1)
            
            
            # prepare a movie
            movie_duration = movies[f'movie_{sampled_videos[i]}']['duration']
            movie_to_show = movies[f'movie_{sampled_videos[i]}']['movie']
            movie_name = f'{sampled_videos[i]}.mp4'
            movie_to_show.stop()
            movie_to_show.seek(movie_duration/2)

            movie_to_show.play()
            audio_to_show.play(when=win)
            
            thisExp.addData('movie_name', movie_name)  # Store the movie name in the CSV
            thisExp.addData('movie_started', start_time) # store the time started

            # Play the video for only 1 second
            while core.getTime() - start_time < 4.0:  # 1 second duration
                background_image.draw() # show the diagonal lines on the screen
                background_image.draw()
                movie_to_show.draw()  # Draw the current frame of the video
                if EEG_trigger == True:
                    win.callOnFlip(p_port.setData, 50 + int(sampled_videos[i])) # set the trigger based on the video
                win.flip()
                
            audio_to_show.stop()
            movie_to_show.pause()
            if EEG_trigger == True:
                p_port.setData(0) # refresh the trigger
            i += 1 # updates the index
            # Show the whole background with lines after movie pauses
            background_image.draw()
            background_image.draw()
            win.flip()
            # Add 800ms silence after the video finishes
            core.wait(0.8)

        # ~~~~~~~~~~~~~~~ Ending Routine "trial" 

        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()

        thisExp.nextEntry() # Advances to the next trial in the experiment. This saves the current trial's data and prepares for the next one.

        # ~~~~~~~~~~~~~~~ completed 1 run (18 trials)

    # completed whole trials (1260 trials = 10min)



    # ~~~~~~~~~~~~~~~ End experiment
    win.flip() # ensues that the final display is shown on the screen.
    thisExp.saveAsWideText(filename+'.csv', delim='auto') # save as csv.file, delimiter used in the CSV file is appropriate
    thisExp.saveAsPickle(filename) # Saves the experiment data as a binary file (a "pickle"). Pickle files are useful because they preserve the Python objects and structure of the experiment data.
    logging.flush() # Ensures that any logs generated during the experiment (e.g., timestamps, warnings, errors) are written to the log file. This prevents loss of log data if something crashes or closes abruptly.
    thisExp.abort() # Signals that the experiment has ended
    win.close() # Closes the PsychoPy window
    core.quit() # Exits the entire PsychoPy script
