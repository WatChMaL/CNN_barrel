# !/bin/bash
# npz_to_h5 -- a script to convert a single npz file to an h5 file

infile_list=$1
outfile=$2
merge_hdf5=/project/rpp-tanaka-ab/jzding/CNN/preprocessing/merge_numpy_arrays_hdf5.py

python $merge_hdf5 $infile $outfile