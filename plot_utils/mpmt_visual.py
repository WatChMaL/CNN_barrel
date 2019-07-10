"""
Set of tools for transforming 19-layer mPMT data into a pixel grid array for matplotlib

Author: Julian Ding
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, ceil

# 10x10 square represents one mPMT
# List of top-left pixel positions (row,col) for 2x2 grids representing PMTs 0 to 18
POS_MAP = [(8,4), #0
           (7,2), #1
           (6,0), #2
           (4,0), #3
           (2,0), #4
           (1,1), #5
           (0,4), #6
           (1,6), #7
           (2,8), #8
           (4,8), #9
           (6,8), #10
           (7,6), #11
           # Inner ring
           (6,4), #12
           (5,2), #13
           (3,2), #14
           (2,4), #15
           (3,6), #16
           (5,6), #17
           (4,4)] #18

# Function to plot a figure of subplots containing visuals of events in sequential
# order when provided with a list of events
def plot_events(data_list, save_path):
    dlist = process(data_list)
    # Error checking
    if dlist is None:
        return
    # Plot
    width = ceil(sqrt(dlist.shape[0]))
    fig, axes = plt.subplots(width, width, sharex='col', sharey='row')
    event = 0
    # Tile the figure with subplots
    for row in axes:
        for f in row:
            curr = dlist[event]
            img = plot_single_image(curr)
            f.imshow(img)
            event += 1
            f.set_title(str(event)+'/'+str(dlist.shape[0]))
    # Save plot
    plt.savefig(save_path)
    
    plt.clf() # Clear the plot frame
    plt.close() # Close the opened window if any

# Function to get a 2D list (i.e. a list of lists, NOT a 2D numpy array)
# of mPMT subplots (numpy arrays) representing a single event
def get_mpmt_grid(data):
    rows = data.shape[0]
    cols = data.shape[1]
    grid = []
    for row in range(rows):
        subgrid = []
        for col in range(cols):
            pmts = data[row, col]
            mpmt = make_mpmt(pmts)
            subgrid.append(mpmt)
        grid.append(subgrid)
    return grid

# Function to get a 2D array of pixels representing a single event
def plot_single_image(data, padding=1):
    rows = data.shape[0]
    cols = data.shape[1]
    # Make empty output pixel grid
    output = np.zeroes(((10+padding)*rows-padding, (10+padding*cols)-padding))
    i, j = 0, 0
    for row in range(rows):
        for col in range(cols):
            pmts = data[row, col]
            tile(output, (i, j), pmts)
            j += 10+padding
        i += 10+padding
        j = 0
    return output

# Helper function to generate a 10x10 array representing an mPMT module
def make_mpmt(pmt_array):
    mpmt = np.zeros((10, 10))
    for i, val in enumerate(pmt_array):
        mpmt[POS_MAP[i][0]][POS_MAP[i][1]] = val
    return mpmt
            
# Helper function to tile a canvas with mpmt subplots (in-place)
def tile(canvas, ul, pmts):
    # First, create 10x10 grid representing single mPMT
    mpmt = make_mpmt(pmts)
        
    # Then, place grid on appropriate position on canvas
    for row in range(10):
        for col in range(10):
            canvas[row+ul[0]][col+ul[1]] = mpmt[row][col]
            
# Helper function to process input data and extract only the 19 layers
# associated with charge (chrg=True) or timing (chrg=False) information for plotting
def process(data_list, chrg=True):
    # Convert data into numpy array if necessary
    data_list = np.asarray(data_list)
    # Check data shape matches (n, 16, 40, 19 or 38)
    if len(data_list.shape) == 4 and data_list.shape[1:3] == (16, 40) and (data_list.shape[-1] == 19 or data_list.shape[-1] == 38):
        if data_list.shape[-1] == 38:
            if chrg:
                data_list = data_list[:,:,:,:19]
            else:
                data_list = data_list[:,:,:,19:]
                
        return data_list
    else:
        print("Invalid data shape (required: n, 16, 40, 19 or 38), aborting")
        return None