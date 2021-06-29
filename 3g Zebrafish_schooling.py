import numpy as np
import pandas as pd
import tracktor as tr
import cv2
import sys
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


df_params = pd.read_csv("params.csv")




##Temporary Cell

# colours is a vector of BGR values which are used to identify individuals in the video
# t_id is termite id and is also used for individual identification
# number of elements in colours should be greater than n_inds (THIS IS NECESSARY FOR VISUALISATION ONLY)
# number of elements in t_id should be greater than n_inds (THIS IS NECESSARY TO GET INDIVIDUAL-SPECIFIC DATA)
n_inds = 1
t_id = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
colours = [(0,0,255),(0,255,255),(255,0,255),(255,255,255),(255,255,0),(255,0,0),(0,255,0),(0,0,0)]

# this is the block_size and offset used for adaptive thresholding (block_size should always be odd)
# these values are critical for tracking performance
block_size = 51
offset = 25

# the scaling parameter can be used to speed up tracking if video resolution is too high (use value 0-1)
scaling = 1.0

# minimum area and maximum area occupied by the animal in number of pixels
# this parameter is used to get rid of other objects in view that might be hard to threshold out but are differently sized
min_area = 50
max_area = 700

# mot determines whether the tracker is being used in noisyconditions to track a single object or for multi-object
# using this will enable k-means clustering to force n_inds number of animals
mot = False

# name of source video and paths
video = 'resize2'
input_vidpath = 'C:/Users/aryan/Downloads/tracktor-master/videos/' + video + '.mp4'
output_vidpath = 'C:/Users/aryan/Downloads/tracktor-master/output/' + video + '_tracked.mp4'
output_filepath = 'C:/Users/aryan/Downloads/tracktor-master/output/' + video + '_tracked.csv'
codec = 'DIVX' # try other codecs if the default doesn't work ('DIVX', 'avc1', 'XVID') note: this list is non-exhaustive



def get_countours_bounds(contours):
    bounds_list = []
    contours_list = [[list(c[0]) for c in cc] for cc in contours]
    for i in range(len(contours_list)):
        min_xy = tuple(np.min(np.asarray(contours_list[i]), axis=0))
        max_xy = tuple(np.max(np.asarray(contours_list[i]), axis=0))
        bounds_list.append([min_xy, max_xy])
        
    return bounds_list


## Open video
cap = cv2.VideoCapture(input_vidpath)
if cap.isOpened() == False:
    sys.exit('Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file')

## Video writer class to output video with contour and centroid of tracked object(s)
# make sure the frame size matches size of array 'final'
fourcc = cv2.VideoWriter_fourcc(*codec)
output_framesize = (int(cap.read()[1].shape[1]*scaling),int(cap.read()[1].shape[0]*scaling))
out = cv2.VideoWriter(filename = output_vidpath, fourcc = fourcc, fps = 60.0, frameSize = output_framesize, isColor = True)

## Individual location(s) measured in the last and current step
meas_last = list(np.zeros((n_inds,2)))
meas_now = list(np.zeros((n_inds,2)))

df = []
last = 0

frame_bounds_list = []

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    this = cap.get(1)
    if ret == True:
        # Preprocess the image for background subtraction
        frame = cv2.resize(frame, None, fx = scaling, fy = scaling, interpolation = cv2.INTER_LINEAR)
        thresh = tr.colour_to_thresh(frame, block_size, offset)
        final, contours, meas_last, meas_now = tr.detect_and_draw_contours(frame, thresh, meas_last, meas_now, min_area, max_area)
        if len(meas_now) != n_inds:
            contours, meas_now = tr.apply_k_means(contours, n_inds, meas_now)
            
            frame_bounds_list.append(get_countours_bounds(contours))
        
        row_ind, col_ind = tr.hungarian_algorithm(meas_last, meas_now)
        final, meas_now, df = tr.reorder_and_draw(final, colours, n_inds, col_ind, meas_now, df, mot, this)
        
        # Create output dataframe
        for i in range(n_inds):
            df.append([this, meas_now[i][0], meas_now[i][1]])
        
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



import matplotlib.pyplot as plt

tmp = df[df['frame'] < 500]
plt.figure(figsize=(10,10))
plt.scatter(tmp['pos_x'], tmp['pos_y'], c=tmp['id'])
plt.xlabel('pos_x')
plt.ylabel('pos_y')
plt.show()


