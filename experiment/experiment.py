
import functions as functions

# chose 'audio' or 'visual'
modality = 'visual'
# specify the monitor size (px), width (cm), and distance (cm) from the screen. 
monitor_data = [1440, 900, 30, 90]   # EEG lab: [1920, 1080, 50, 90], my mac: [1440, 900, 30, 90] 
# specify whether you need trigger or not.
EEG_Trigger = False


if modality == 'audio':
    functions.run_audio(monitor_data, EEG_Trigger)
elif modality == 'visual':
    functions.run_visual(monitor_data, EEG_Trigger)


