import numpy as np
import pandas as pd
import tracktor as tr
import cv2
import sys
import scipy.signal
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist



# colours is a vector of BGR values which are used to identify individuals in the video
# since we only have one individual, the program will only use the first element from this array i.e. (0,0,255) - red
# number of elements in colours should be greater than n_inds (THIS IS NECESSARY FOR VISUALISATION ONLY)
n_inds = 1
colours = [(0,0,255),(0,255,255),(255,0,255),(255,255,255),(255,255,0),(255,0,0),(0,255,0),(0,0,0)]

# this is the block_size and offset used for adaptive thresholding (block_size should always be odd)
# these values are critical for tracking performance
block_size = 81
offset = 38

# the scaling parameter can be used to speed up tracking if video resolution is too high (use value 0-1)
scaling = 1

# minimum area and maximum area occupied by the animal in number of pixels
# this parameter is used to get rid of other objects in view that might be hard to threshold out but are differently sized
min_area = 1000
max_area = 10000

# mot determines whether the tracker is being used in noisy conditions to track a single object or for multi-object
# using this will enable k-means clustering to force n_inds number of animals
mot = False

# name of source video and paths
video = 'fish_video'
input_vidpath = 'C:/Users/aryan/Downloads/tracktor-master/videos/' + video + '.mp4'
output_vidpath = 'C:/Users/aryan/Downloads/tracktor-master/output/' + video + '_tracked.mp4'
output_filepath = 'C:/Users/aryan/Downloads/tracktor-master/output/' + video + '_tracked.csv'
codec = 'DIVX' # try other codecs if the default doesn't work ('DIVX', 'avc1', 'XVID') note: this list is non-exhaustive



## Open video
cap = cv2.VideoCapture(input_vidpath)
if cap.isOpened() == False:
    sys.exit('Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file')

## Video writer class to output video with contour and centroid of tracked object(s)
# make sure the frame size matches size of array 'final'
fourcc = cv2.VideoWriter_fourcc(*codec)
output_framesize = (int(cap.read()[1].shape[1]*scaling),int(cap.read()[1].shape[0]*scaling))
out = cv2.VideoWriter(filename = output_vidpath, fourcc = fourcc, fps = 30.0, frameSize = output_framesize, isColor = True)

## Individual location(s) measured in the last and current step
meas_last = list(np.zeros((n_inds,2)))
meas_now = list(np.zeros((n_inds,2)))

last = 0
df = []

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    this = cap.get(1)
    if ret == True:
        frame = cv2.resize(frame, None, fx = scaling, fy = scaling, interpolation = cv2.INTER_LINEAR)
        
        # Apply mask to aarea of interest
        #mask = np.zeros(frame.shape)
        #mask = cv2.rectangle(mask, (1921, 1), (3840,2160), (255,255,255), -1)
        #frame[mask ==  0] = 0
        
        thresh = tr.colour_to_thresh(frame, block_size, offset)
        final, contours, meas_last, meas_now = tr.detect_and_draw_contours(frame, thresh, meas_last, meas_now, min_area, max_area)
        row_ind, col_ind = tr.hungarian_algorithm(meas_last, meas_now)
        final, meas_now, df = tr.reorder_and_draw(final, colours, n_inds, col_ind, meas_now, df, mot, this)
        
        # Create output dataframe
        #for i in range(n_inds):
            #df.append([this, meas_now[i][0], meas_now[i][1]])
        
        # Display the resulting frame
        out.write(final)
        cv2.imshow('frame', final)
        if cv2.waitKey(1) == 27:
            break
            
    if last >= this:
        break
    
    last = this

## Write positions to file
df = pd.DataFrame(np.matrix(df), columns = ['frame','pos_x','pos_y'])
df.to_csv(output_filepath, sep=',')

## When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)



df = pd.read_csv(output_filepath)
df.head()


import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
plt.scatter(df['pos_x'], df['pos_y'], c=df['frame'], alpha=0.5)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.tight_layout()
plt.savefig('imgs/ex1_fig1a.eps', format='eps', dpi=300)
plt.show()

plt.figure(figsize=(5,5))
plt.hist2d(df['pos_x'], df['pos_y'], bins=20)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.tight_layout()
plt.savefig('imgs/ex1_fig1b.eps', format='eps', dpi=300)
plt.show()



## Parameters like speed and acceleration can be very noisy. Small noise in positional data is amplified as we take the
## derivative to get speed and acceleration. We therefore smooth this data to obtain reliable values and eliminate noise.

# the smoothing window parameter determines the extent of smoothing (this parameter must be odd)
smoothing_window = 11

## Fill in the parameters below if you'd like movement measures to be converted from pixels and frames to 
## real-world measures (cms and secs)

# Frame-rate (fps or frames per second) of recorded video to calculate time
fps = 1000

# Pixels per cm to in the recorded video to calculate distances
pxpercm = 78 * scaling




dx = df['pos_x'] - df['pos_x'].shift(n_inds)
dy = df['pos_y'] - df['pos_y'].shift(n_inds)
d2x = dx - dx.shift(1)
d2y = dy - dy.shift(1)
df['speed'] = np.sqrt(dx**2 + dy**2)
df['smoothed_speed'] = scipy.signal.savgol_filter(df['speed'], smoothing_window, 1)
df['accn'] = np.sqrt(d2x**2 + d2y**2)
df['smoothed_accn'] = scipy.signal.savgol_filter(df['accn'], smoothing_window, 1)
df['cum_dist'] = df['smoothed_speed'].cumsum()
df.head()




def cumul_dist(start_fr, end_fr):
    if start_fr != 1:
        cumul_dist = df['cum_dist'][df['frame'] == end_fr].values[0] - df['cum_dist'][df['frame'] == start_fr].values[0]
    else:
        cumul_dist = df['cum_dist'][df['frame'] == end_fr].values[0]
    return cumul_dist

cumul_dist(150,200)

df['time'] = df['frame'] / fps
df['speed'] = df['speed'] * fps / pxpercm
df['smoothed_speed'] = df['smoothed_speed'] * fps / pxpercm
df['accn'] = df['accn'] * fps * fps / pxpercm
df['smoothed_accn'] = df['smoothed_accn'] * fps * fps / pxpercm
df['cum_dist'] = df['cum_dist'] / pxpercm
df.head()

cumul_dist(140,170) / pxpercm


np.nanmax(df['smoothed_speed']), np.nanmax(df['smoothed_accn'])


## We now remove any outliers that remain post smoothing
## Here we want to conservative and not eliminate any relavant points as outliers. We therefore choose a high 'm' value
## in the reject_outliers functions. The best approach is to visually compare smoothed data with the original data
index = tr.reject_outliers(df['smoothed_speed'], m = 6)
index = np.array(index[0])


plt.scatter(df['time'][index], df['cum_dist'][index], c='#FF7F50', s=8, alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Cumulative distance (cm)')
plt.tight_layout()
plt.savefig('imgs/ex1_fig2a.eps', format='eps', dpi=300)
plt.show()

plt.scatter(df['time'][index], df['speed'][index], s=5, alpha=0.5)
plt.plot(df['time'][index], df['smoothed_speed'][index], c='#FF7F50', lw=3)
plt.ylim(0,200)
plt.xlabel('Time')
plt.ylabel('Speed (cm/s)')
plt.tight_layout()
plt.savefig('imgs/ex1_fig2b.eps', format='eps', dpi=300)
plt.show()

plt.scatter(df['time'][index], df['accn'][index], s=5, alpha=0.5)
plt.plot(df['time'][index], df['smoothed_accn'][index], c='#FF7F50', lw=3)
plt.ylim(0,200000)
plt.xlabel('Time')
plt.ylabel('Acceleration (cm/sq.s)')
plt.tight_layout()
plt.savefig('imgs/ex1_fig2c.eps', format='eps', dpi=300)
plt.show()







