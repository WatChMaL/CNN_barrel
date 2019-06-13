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
# Name for temporary file used in normalize_dset
TEMP = 'temp.h5'
# Exception for when functions are called with null accumulator
ACC_EXCEPTION = Exception("Attempted to apply operation with null accumulator.")
# Large number to initialize min accumulators to
LARGE = 1e10

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
    # Read and parse data to be normed in chunks to prevent memory overflow
    c_func = globals()[config.chrg_norm_func[0]]
    t_func = globals()[config.time_norm_func[0]]
    print("Event data normalization scheme: charge =", str(c_func), "| timing =", str(t_func))
    event_data = infile[NORM_CAT]
    chunk_length = event_data.shape[0]
    num_blocks_in_file = int(ceil(chunk_length / block_size))
    # Accumulators for functions that need them
    c_acc, t_acc = None, None
    # Use a temporary h5 file to facilitate reading and writing to disk (instead of loading all data in memory)
    temp = h5py.File(TEMP, 'w')
    temp_dset = temp.create_dataset(NORM_CAT, shape=event_data.shape, dtype=event_data.dtype)
    print("Performing normalization | Block size:", block_size)
    # Read and process in chunks (apply=False)
    for iblock in range(num_blocks_in_file):
        block_begin=iblock*block_size
        block_end=(iblock+1)*block_size
        if block_end>chunk_length:
            block_end=chunk_length
        # Do necessary calculations while updating accumulators
        chrg_data = event_data[block_begin:block_end,:,:,:19]
        time_data = event_data[block_begin:block_end,:,:,19:]
        c_acc = c_func(chrg_data, acc=c_acc, apply=False)
        t_acc = t_func(time_data, acc=t_acc, apply=False)
        if iblock != 0:
            print('\r', end='')
        print('[', iblock+1, 'of', num_blocks_in_file, 'blocks parsed ]', end='')
    print('')
    # Write to temp file in chunks (apply=True)
    for iblock in range(num_blocks_in_file):
        block_begin=iblock*block_size
        block_end=(iblock+1)*block_size
        if block_end>chunk_length:
            block_end=chunk_length
        # Apply normalization schemes to data
        chrg_data = event_data[block_begin:block_end,:,:,:19]
        time_data = event_data[block_begin:block_end,:,:,19:]
        chrg_data = c_func(chrg_data, acc=c_acc, apply=True)
        time_data = t_func(time_data, acc=t_acc, apply=True)
        out_data = np.concatenate((chrg_data, time_data), axis=-1)
        # Save state to temporary h5
        temp_dset[block_begin:block_end] = out_data
        if iblock != 0:
            print('\r', end='')
        print('[', iblock+1, 'of', num_blocks_in_file, 'blocks normalized ]', end='')
    print('')
    
    # Write to outfile in chunks to prevent memory overflow
    for key in infile.keys():
        print('Saving key', key)
        if key == NORM_CAT:
            # Use new event data in tempfile
            data = temp[NORM_CAT]
        else:
            data = infile[key]
        chunk_length = data.shape[0]
        num_blocks_in_file = int(ceil(chunk_length / block_size))
        offset=0
        for iblock in range(num_blocks_in_file):
            block_begin=iblock*block_size
            block_end=(iblock+1)*block_size
            if block_end>chunk_length:
                block_end=chunk_length
            # Write to file
            dsets[key][offset+block_begin:offset+block_end] = data[block_begin:block_end]
            if iblock != 0:
                print('\r', end='')
            print('[', iblock+1, 'of', num_blocks_in_file, 'blocks written ]', end='')
        offset+=block_end
        print('')
        
    # Delete tempfile
    temp.close()
    os.remove(TEMP)
        
    # Close files
    infile.close()
    outfile.close()
    
    print("Normalization complete.")

# =================== NORMALIZATION FUNCTION CANDIDATES ====================
    
# Identity function that returns the input dataset
def identity(data, acc=None, apply=False):
    return data

# Function that divides every entry in data array by the (non-zero) mean of the data
# acc = (mean of events seen, number of events seen)
def divide_by_mean(data, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            return data / (acc['sum']/acc['number'])
    else:
        # Calculate mean of all non-zero hits in chunk
        nonzero = np.asarray([hit for hit in data.reshape(-1,1) if hit != 0])
        if acc is None:
            acc = {'sum':0, 'number':0}
        acc['sum'] += np.sum(nonzero)
        acc['number'] += nonzero.size
        return acc

# Function that divides every entry in a data array by the max of the data
def divide_by_max(data, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            return data / acc
    else:
        if acc is None:
            acc = 0
        return max(np.amax(data), acc)

# Function that scales a dataset logarithmically: x = log(x+1)
def scale_log(data, acc=None, apply=False):
    check_data(data)
    if apply:
        return np.log(data+1)
    else:
        return identity(data)

# Function that removes offsets in data by setting the lowest non-zero hit value as the new zero
def remove_offset_min(data, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            offset = data - acc
            return offset.clip(0)
    else:
        # Initialize accumulator if necessary
        if acc is None:
            acc = LARGE
        # Find minimum nonzero value and save lowest so far
        nonzero = np.asarray([hit for hit in data.reshape(-1,1) if hit != 0])
        return min(np.amin(nonzero), acc)

# Function that removes offsets in data by setting the mode (peak) non-zero hit value to 1
def remove_offset_mode(data, bins=10000, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            mode_idx = np.where(acc['hist'] == np.amax(acc['hist']))[0]
            mode = mode_idx*acc['max']/bins
            # Subtract minimum value from every nonzero value
            out = data - mode + 1
            return out.clip(0)
    else:
        # Find mode non-zero hit value by binning data and selecting highest-frequency bin
        nonzero = np.asarray([hit for hit in data.reshape(-1,1) if hit != 0])
        data_max = np.amax(nonzero)
        flat_data = nonzero.reshape(-1,1)
        if acc is None:
            acc = {'hist': np.zeros((bins+1, 1)), 'max': 0}
        for item in flat_data:
            acc['hist'][floor((item/data_max)*bins)] += 1
        acc['max'] = max(acc['max'], data_max)
        return acc
    

# Temporary function for shifting data by an arbitrary amount
def offset_arbitrary(data, offset=750, acc=None, apply=False):
    check_data(data)
    if apply:
        return (data-offset).clip(0)
    else:
        return identity(data)
    
# =============== Function compositions =================
    
def offset_divide_by_mean(data, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            return divide_by_mean(offset_arbitrary(data, apply=True), acc=acc, apply=True)
    else:
        return divide_by_mean(data, acc=acc, apply=False)

def offset_divide_by_max(data, acc=None, apply=False):
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            return divide_by_max(offset_arbitrary(data, apply=True), acc=acc, apply=True)
    else:
        return divide_by_max(data, acc=acc, apply=False)

def offset_scale_log(data, acc=None, apply=False):
    check_data(data)
    if apply:
        return scale_log(offset_arbitrary(data, apply=True), apply=True)
    else:
        return identity(data)

# Helper function to check input data shape
def check_data(data):
    assert len(data.shape) == 4, "Invalid data shape (required: n, 16, 40, 19), aborting"
    assert data.shape[1:] == (16, 40, 19), "Invalid data shape (required: n, 16, 40, 19), aborting"
    
# Main
if __name__ == "__main__":
    config = parse_args()
    normalize_dataset(config)