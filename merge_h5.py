import numpy as np
import glob
import sys
import os
from pathlib import Path
import argparse
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sn

import h5py

'''
Merges numpy arrays into an hdf5 file
'''

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merges h5 files into one uncompressed h5 file")
    parser.add_argument("input_file_list",
                        type=str, nargs=1,
                        help="txt file with a list of files to merge")
    parser.add_argument('output_file', type=str, nargs=1,
                        help="where do we put the output")
    parser.add_argument('keys', type=str, nargs='*',
                        help="keys to store")
    parser.add_argument('--block_size', type=int, default=3500)
    
    
    
    args = parser.parse_args()
    return args


def merge_h5(config):
    
    #read in the input file list
    with open(config.input_file_list[0]) as f:
        files = f.readlines()

    #remove whitespace 
    file_list = [x.strip() for x in files]

    # -- Check that files were provided
    if len(file_list) == 0:
        raise ValueError("No files provided!!")
    print("Merging "+str(len(file_list))+" files")
    
    print("Files are:")
    print(file_list)

    keys=config.keys
    print("keys are:")
    print(keys)

    
    start_indices=np.zeros((len(file_list)),dtype=np.int)
    chunk_lengths=np.zeros((len(file_list)),dtype=np.int)
    total_rows=0
    for file_index, file_name in enumerate(file_list):
        infile=h5py.File(file_name,"r")

        current_keys=list(infile.keys())
        
        if file_index==0:
            if len(keys)==0:
                keys=list(infile.keys())
            shapes={}
            dtypes={}
            for key in keys:
                data=infile[key]
                shapes[key]=data.shape[1:] #only want the tensor shape not the number of examples
                dtypes[key]=data.dtype
                chunk_lengths[0]=data.shape[0]

        if not set(keys).issubset(set(current_keys)):
            raise(ValueError("keys in file {} are missing"\
                             "or different from first file {}".
                             format(file_name,
                                    file_list[0])))

        prev_data_length=None
        for key in keys:
            data=infile[key]
            current_data_length=data.shape[0]
            if prev_data_length is not None:
                if prev_data_length!=current_data_length:
                    rise(ValueError("keys don't have same length in {}".format(
                        file_name)))
            else:
                prev_data_length=current_data_length
                total_rows+=current_data_length
                chunk_lengths[file_index]=current_data_length
                if file_index>0:
                    start_indices[file_index]=start_indices[file_index-1] \
                                               + chunk_lengths[file_index-1]

            if data.dtype!=dtypes[key]:
                raise(ValueError("dtype changed in file {} key {}".format(
                    file_name, key)))

            if data.shape[1:]!=shapes[key]:
                raise(ValueError("shapes changed for key {}"\
                                  "in file {}".format(key,file_name)))
                    

    
                    
                                    
    print("keys: {}".format(keys))
    print("shapes: {}".format(shapes))
    print("start indices: {}".format(start_indices))
    print("chunk lengths: {}".format(chunk_lengths))
    print("dtypes: {}".format(dtypes))
    print("total_rows: {}".format(total_rows))
                        
                                  
    print("opening the hdf5 file\n")
    f=h5py.File(config.output_file[0],'x')
    
    dsets={}
    for key in keys:
        c_dset=f.create_dataset(key,
                                shape=(total_rows,)+shapes[key],
                                dtype=dtypes[key])
        dsets[key]=c_dset


    for key in keys:

        offset=0
        for file_index, file_name in enumerate(file_list):
            infile=h5py.File(file_name,"r")

            data=f[key]

            #ceiling division trick
            assert offset==start_indices[file_index]
            num_blocks_in_file=-(-chunk_lengths[file_index] // config.block_size)
            for iblock in range(num_blocks_in_file):
                block_begin=iblock*config.block_size
                block_end=(iblock+1)*config.block_size
                if block_end>chunk_lengths[file_index]:
                    block_end=chunk_lengths[file_index]

                dsets[key][offset+block_begin:offset+block_end]=data[block_begin:block_end]

            offset+=block_end
        

    f.close()
        
                             
                
            
            
        
    

if __name__ == '__main__':

    config=parse_args()
    merge_h5(config)
