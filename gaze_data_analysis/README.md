# First Order Gaze Metric Analysis

This is the sub-repo for extracting gaze metrics, including fixations, saccades and smooth pursuits for the XR data we extracted from the Sudoku Helper app we developed. 

## Data
### Collected Raw Data
Due to privacy concerns we cannot open-source our data. However, sample data are provided at [AR samples](../dataset/AR_samples) and [VR samples](../dataset/VR_samples), with the same column names and placeholders showing the format of the data we collected. While most columns have the same meanings, serveral differences between the AR and VR data are listed as follows:

1. AR data was collected at 60Hz, while VR data was collected at 120Hz;
2. Besides than gaze directions, AR data futher include eyes open amount and 3D eye center positions, while VR data have pupil dilations
3. Gaze targets were recorded in different ways in AR and VR: while in both cases "useful", "puzzle" and "mascot" refers to the hints, the Sudoku board and the mascot respectively, in VR "normal" is exclusive of "timer" and "progress bar" while in AR it's inclusive. 
4. In AR an additional column of "FalseMistake" was included, in case the digit recognition module gave incorrect outputs.

Note that the files ending with "base" are data collected from the baseline trial where users solved the easy-level Sudoku without any other virtual content than the puzzle provided (meaning in AR nothing was rendered), and the files ending with "final" are those collected in the official trial when the users solved the hard-level Sudoku with guidance and distractors. The "final" trial might be broken into pieces due to hardware malfunctioning, and we merge those data in our data-processing pipeline.

### Data Processing
Please refer to [the defined data classes](offline/data/gaze_data.py) for details of processing. Prior to gaze metric extraction, two steps are perfromed, namely:
1. Gaze conversion from both eyes' gaze to one direction vector (`compute_combined_gaze()`);
2. Compile the gaze targets (`convert_label_columns()` and `convert_target_columns()`) as specified in each time step.


## Method
The algorithm we used is the I-VT algorithm, and we set the velocity threshold for detecting fixations at 30 deg/s accordingly to Tobii's setup. We define smooth pursuits as gaze events with velocity at 30-100 deg/s, fixating at either (1) only on-puzzle objects (hints and the puzzle), or (2) only off-puzzle objects (the timer, the progress bar and the mascot). 

1. Apply a median filter on gaze direction;
2. Compute the velocity
3. Apply the Savgol filter to velocity
4. Compute fixations and smooth pursuits
5. compute saccades (negative of fixations + smooth pursuits)
6. Compute fixations on each ROI (timer, progress bar, mascot, puzzle, hint)

Please see [metrics](offline/modules/metric.py) for all metrics being computeed. 
You can also define your own metrics!
