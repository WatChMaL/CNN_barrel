# !/bin/bash
# event_dump -- A shell script to replicate the functionality of event_dump.py
#               by iteratively calling event_dump_one.py while circumventing the
#               memory leak in event_dump.py

# Usage: ./event_dump.sh INPUT_DIR OUTPUT_DIR
# Put entire script in subshell to kill with single Ctrl-C

source /project/rpp-tanaka-ab/hk_software/nuPRISM/sourceme.sh
module load python/2.7.14
module load scipy-stack
export PYTHONPATH=$ROOTSYS/../bindings/pyroot:$PYTHONPATH
# Comment out the above lines if not on Cedar (ComputeCanada)

(
input_dir=$1
output_dir=$2
valids=()

for file in `ls $input_dir`; do
    if [[ $file = *"_R0cm_"* ]]&&[[ $file = *".root" ]]&&[[ ! $file = *"_flat"* ]]; then
        valids+=($file)
    fi
done

total=${#valids[@]}
echo "Found $total valid files in $input_dir."
idx=0

for file in ${valids[@]}; do
    file_dir="$input_dir$file"
    echo "Attempting to process $file... (Progress: $idx/$total)" || break
    python event_dump_one.py $file_dir $output_dir || break
    ((idx++))
done

if [ $idx == $total ]; then
    echo "==== Finished processing $total files! ===="
fi
)