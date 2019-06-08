"""
Script to normalize mPMT hit information in an HDF5 dataset.
Transforms input HDF5 dataset into output HDF5 dataset (not in place).

Author: Julian Ding
"""
import os
import argparse
import h5py
import numpy as np
from math import ceil, floor

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
                        help="number of events to load into memory at once", required=False)
    parser.add_argument('--chrg_norm', '-cf', dest="chrg_norm_func", type=str, nargs=1, default=['identity'],
                        help="normalization function to apply to charge data", required=False)
    parser.add_argument('--time_norm', '-tf', dest="time_norm_func", type=str, nargs=1, default=['identity'],
                        help="normalization function to apply to time data", required=False)
    args = parser.parse_args()
    return args

def normalize_dataset(config):
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
    if not os.path.isdir(os.path.dirname(config.output_file)):
        os.mkdir(os.path.dirname(config.output_file))
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
        print('Saving key', key, '| Block size:', block_size)
        offset=0
        # If the category is the data we wish to normalize, apply normalization function
        if key == NORM_CAT:
            c_func = globals()[config.chrg_norm_func[0]]
            t_func = globals()[config.time_norm_func[0]]
            print(key, "normalization scheme: charge =", str(c_func), "| timing =", str(t_func))
            chrg_data = c_func(chrg_data)
            time_data = t_func(time_data)
            data = np.concatenate((chrg_data, time_data), axis=-1)
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
            print(iblock, 'of', num_blocks_in_file, 'blocks written')
        offset+=block_end
        
    # Close files
    infile.close()
    outfile.close()
    
    print("Normalization complete.")

# =================== NORMALIZATION FUNCTION CANDIDATES ====================
    
# Identity function that returns the input dataset
def identity(data):
    return data

# Function that divides every entry in data array by the (non-zero) mean of the data
def divide_by_mean(data):
    check_data(data)
    # Calculate mean of all non-zero hits
    nonzero = np.asarray([hit for hit in data.reshape(-1,1) if hit != 0])
    mean = np.mean(nonzero)
    # Divide data by mean
    return data / mean

# Function that divides every entry in data array by the (non-zero) median of the data
def divide_by_median(data):
    check_data(data)
    # Find median of all non-zero hits
    nonzero = np.asarray([hit for hit in data.reshape(-1,1) if hit != 0])
    median = np.median(nonzero)
    # Divide data by median
    return data / median

# Function that divides every entry in a data array by the max of the data
def divide_by_max(data):
    return data / np.amax(data)

## Function that divides every entry in data array by the (non-zero) mean of the individual events
#def divide_by_mean_event(data):
#    check_data(data)
#    out = []
#    for i, event in enumerate(data):
#        out.append(divide_by_mean(np.asarray([event])))
#    return np.asarray(out)
#
## Function that divides every (non-zero) entry in data array by the median of the individual events
#def divide_by_median_event(data):
#    check_data(data)
#    out = []
#    for i, event in enumerate(data):
#        out.append(divide_by_median(np.asarray([event])))
#    return np.asarray(out)

# Function that scales a dataset logarithmically: x = log(x+1)
def scale_log(data):
    check_data(data)
    return np.log(data+1)

# Function that removes offsets in data by setting the lowest non-zero hit value as the new zero
def remove_offset_min(data):
    check_data(data)
    # Find minimum non-zero hit value
    nonzero = np.asarray([hit for hit in data.reshape(-1,1) if hit != 0])
    minimum = np.amin(nonzero)
    # Subtract minimum value from every value
    return (data-minimum).clip(0)

# Function that removes offsets in data by setting the mode (peak) non-zero hit value to 1
def remove_offset_mode(data, bins=10000):
    check_data(data)
    # Find mode non-zero hit value by binning data and selecting highest-frequency bin
    nonzero = np.asarray([hit for hit in data.reshape(-1,1) if hit != 0])
    data_max = np.amax(nonzero)
    interval = data_max/bins
    flat_data = nonzero.reshape(-1,1)
    binned_data = np.zeros((bins+1, 1))
    for item in flat_data:
        binned_data[floor((item/data_max)*bins)] += 1
    mode_idx = np.where(binned_data == np.amax(binned_data))[0]
    mode = mode_idx*interval
    # Subtract minimum value from every nonzero value
    out = data - mode + 1
    return out.clip(0)

# Temporary function for shifting data by an arbitrary amount
def offset_arbitrary(data, offset=750):
    return (data-offset).clip(0)
    
# =============== Function compositions =================
    
def offset_divide_by_mean(data):
    check_data(data)
    return divide_by_mean(offset_arbitrary(data))

def offset_divide_by_max(data):
    check_data(data)
    return divide_by_max(offset_arbitrary(data))

def offset_scale_log(data):
    check_data(data)
    return scale_log(offset_arbitrary(data))

# Helper function to check input data shape
def check_data(data):
    assert len(data.shape) == 4, "Invalid data shape (required: n, 16, 40, 19), aborting"
    assert data.shape[1:] == (16, 40, 19), "Invalid data shape (required: n, 16, 40, 19), aborting"
    
# Main
if __name__ == "__main__":
    config = parse_args()
    
    normalize_dataset(config)