"""
Set of tools for transforming 19-layer mPMT data into a pixel grid array for matplotlib

Author: Julian Ding
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from math import sqrt, ceil

# 10x10 square represents one mPMT
# List of top-left pixel positions (row,col) for 2x2 grids representing PMTs 0 to 18
# PMT mapping below:
#
#0 -------|06|-------
#1 ----|05|06|07|----
#2 -|04|05|15|07|08|-
#3 -|04|14|15|16|08|-
#4 -|03|14|18|16|09|-
#5 -|03|13|18|17|09|-
#6 -|02|13|12|17|10|-
#7 -|02|01|12|11|10|-
#8 ----|01|00|11|----
#9 -------|00|-------
#
POS_MAP = [(8,4), #0
           (7,2), #1
           (6,0), #2
           (4,0), #3
           (2,0), #4
           (1,2), #5
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
    maximum = np.amax(dlist)
    # Error checking
    if dlist is None:
        return
    # Plot
    nevents = dlist.shape[0]
    width = ceil(sqrt(nevents))
    fig, axes = plt.subplots(width, width, sharex='col', sharey='row')
    event = 0
    # Tile the figure with subplots
    for row in axes:
        for f in row:
            curr = dlist[event]
            img = plot_single_image(curr)
            
            f.imshow(img, origin="upper", cmap="inferno", norm=LogNorm(vmax=maximum, clip=True))
            # Turn off axis labels and ticks
            f.axis('off')
    
            event += 1
            f.set_title(str(event)+'/'+str(nevents), fontsize=5)
            
            if event >= nevents:
                # Clean up plot
                plt.tight_layout(pad=0.1)
                # Save plot
                plt.savefig(save_path, dpi=600)
                
                plt.clf() # Clear the plot frame
                plt.close() # Close the opened window if any
                return

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
    output = np.zeros(((10+padding)*rows-padding, (10+padding)*cols-padding))
    i, j = 0, 0
    for row in range(rows):
        for col in range(cols):
            pmts = data[row][col]
            tile(output, (i, j), pmts)
            j += 10+padding
        i += 10+padding
        j = 0
    return output
            
# Helper function to tile a canvas with mpmt subplots (in-place)
def tile(canvas, ul, pmts):
    # First, create 10x10 grid representing single mPMT
    mpmt = make_mpmt(pmts)
        
    # Then, place grid on appropriate position on canvas
    for row in range(10):
        for col in range(10):
            canvas[row+ul[0]][col+ul[1]] = mpmt[row][col]

# Helper function to generate a 10x10 array representing an mPMT module
def make_mpmt(pmt_array):
    mpmt = np.zeros((10, 10))
    for i, val in enumerate(pmt_array):
        mpmt[POS_MAP[i][0]][POS_MAP[i][1]] = val
    return mpmt
            
# Helper function to process input data and extract only the 19 layers
# associated with charge (chrg=True) or timing (chrg=False) information for plotting
def process(events, chrg=True):
    # Check data shape matches (n, 16, 40, 19 or 38)
    if len(events.shape) == 4 and events.shape[1:3] == (16, 40) and (events.shape[-1] == 19 or events.shape[-1] == 38):
        if events.shape[-1] == 38:
            if chrg:
                events = events[:,:,:,:19]
            else:
                events = events[:,:,:,19:]
                
        return events
    else:
        print("Invalid data shape {actual:", events.shape, "| required: (n, 16, 40, 19 or 38)}, aborting")
        return None