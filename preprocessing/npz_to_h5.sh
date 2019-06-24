# !/bin/bash
# npz_to_h5 -- a script to convert a single npz file to an h5 file

infile=$1
outfile=$2
list="temp.txt"
merge_numpy_arrays_hdf5="merge_numpy_arrays_hdf5.py"

echo $infile >> $list
python $merge_numpy_arrays_hdf5 $list $outfile

rm -rf $list