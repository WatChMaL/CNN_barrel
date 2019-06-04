"""
Script to normalize mPMT hit information in an HDF5 dataset.
Transforms input HDF5 dataset into output HDF5 dataset (not in place).

Author: Julian Ding
"""
import os, sys
import argparse
import h5py
import numpy as np
from math import ceil

# Key for data dimension that requires normalization
NORM_CAT = 'event_data'

def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalizes data from an input HDF5 file and outputs to HDF5 file.")
    parser.add_argument("--input_file", '-in', dest="input_file", type=str, nargs=1,
                        help="path to dataset to normalize")
    parser.add_argument('--output_file', '-out', dest="output_file", type=str, nargs=1,
                        help="desired output path")
    parser.add_argument('--block_size', '-blk', dest="block_size", type=int, default=3500,
                        required=False)
    args = parser.parse_args()
    return args

# c_func and t_func must transform data in-place
def normalize_dataset(config, c_func=None, t_func=None):
    config.input_file = config.input_file[0]
    config.output_file = config.output_file[0]
    # Ensure specified input file exists, then open file
    assert os.path.isfile(config.input_file), "Invalid input file path provided: "+config.input_file
    print("Reading data from", config.input_file)
    infile = h5py.File(config.input_file, 'r')
    # Separate charge and time domain hits
    event_data = infile['event_data']
    chrg_data = event_data[:,:,:,:19]
    time_data = event_data[:,:,:,19:]
    
    # Open output file for saving
    if not os.path.isdir(os.path.pardir(config.output_file)):
        os.mkdir(os.path.pardir(config.output_file))
    print("Saving normalized dataset to", config.output_file)
    outfile = h5py.File(config.output_file, 'x')
    # Generate data categories
    dsets = {}
    for key in infile.keys():
        c_dset=outfile.create_dataset(key, shape=infile[key].shape, dtype=infile[key].dtype)
        dsets[key]=c_dset
    # Write data to outfile
    block_size = int(config.block_size)
    for key in infile.keys():
        offset=0
        # If the category is the data we wish to normalize, apply normalization function
        if key == NORM_CAT and c_func is not None and t_func is not None:
            c_func(chrg_data)
            t_func(time_data)
            data = np.concatenate(chrg_data, time_data, axis=-1)
        else:
            data = infile[key]
        chunk_length = data.shape[0]
        num_blocks_in_file = int(ceil(chunk_length / block_size))
        # Write to outfile in chunks to prevent memory overflow
        for iblock in range(num_blocks_in_file):
            block_begin=iblock*block_size
            block_end=(iblock+1)*block_size
            if block_end>chunk_length:
                block_end=chunk_length
            dsets[key][offset+block_begin:offset+block_end] = data[block_begin:block_end]
        offset+=block_end
        
    # Close files
    infile.close()
    outfile.close()

# =================== NORMALIZATION FUNCTION CANDIDATES ====================

# Current candidate for c_func
# Function that divides every (non-zero) entry in data array by the mean of the data (in-place)
def divide_by_mean(data):
    check_data(data)
    # Calculate mean of all non-zero hits
    total_sum, count = 0, 0
    for event in data:
        for row in event:
            for mpmt in row:
                for hit in mpmt:
                    if hit != 0:
                        total_sum += hit
                        count += 1
    mean = total_sum / count
    
    # Divide all non-zero hits by mean
    for a, event in enumerate(data):
        for b, row in enumerate(event):
            for c, mpmt in enumerate(row):
                for d, hit in enumerate(mpmt):
                    if hit != 0:
                       data[a][b][c][d] = hit / mean

# Function that divides every (non-zero) entry in data array by the mean of the individual events (in-place)
def divide_by_mean_event(data):
    check_data(data)
    for i, event in enumerate(data):
        divide_by_mean(event)
        data[i] = event
        
# Function that removes offsets in data by setting the lowest non-zero hit value as the new zero (in-place)
def remove_offset_lowest(data):
    check_data(data)
    # Find minimum non-zero hit value
    minimum = sys.maxsize
    for event in data:
        for row in event:
            for mpmt in row:
                for hit in mpmt:
                    if hit != 0 and hit < minimum:
                        minimum = hit
    
    # Subtract minimum value from every value
    for a, event in enumerate(data):
        for b, row in enumerate(event):
            for c, mpmt in enumerate(row):
                for d, hit in enumerate(mpmt):
                    if hit != 0:
                       data[a][b][c][d] = hit - minimum

# Current candidate for t_func
def remove_offset_and_divide_by_mean_event(data):
    check_data(data)
    remove_offset_lowest(data)
    divide_by_mean_event(data)

def check_data(data):
    assert len(data.shape) == 4, "Invalid data shape (required: n, 16, 40, 19), aborting"
    assert data.shape[1:] == (16, 40, 19), "Invalid data shape (required: n, 16, 40, 19), aborting"
    
# Main
if __name__ == "__main__":
    config = parse_args()
    normalize_dataset(config)