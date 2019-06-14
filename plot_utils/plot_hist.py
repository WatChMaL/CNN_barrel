"""
Script for plotting charge and time histograms of HDF5 event dataset

Author: Julian Ding
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
import h5py

# File extension to save histograms as
EXT = '.pdf'

# Constants to set limits in plotting
SCALE_X = 0.5
Y_CHRG = 3e6
Y_TIME = 1e6

# Large initialization constant for minimizers
LARGE = 1e10

# Font size for titles
FONTSIZE = 12

# Argument parsing for commandline functionality
def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalizes data from an input HDF5 file and outputs to HDF5 file.")
    parser.add_argument("--input_file", '-in', dest="input_file", type=str, nargs=1,
                        help="path to dataset to visualize", required=True)
    parser.add_argument('--output_path', '-out', dest="output_path", type=str, nargs=1,
                        help="desired output path", required=True)
    parser.add_argument('--sample_size', '-num', dest="sample_size", type=int, default=10000,
                        help="number of events to sample from dataset", required=False)
    parser.add_argument('--num_bins', '-bin', dest='num_bins', type=int, default=1000,
                        help="number of bins to separate data into in histograms", required=False)
    parser.add_argument('--show_plts', '-show', dest="show_plts", type=str, default=None,
                        help="use this flag to show plots", required=False)
    args = parser.parse_args()
    return args

# Returns an array of event samples given the path to an HDF5 dataset
# Note: sample is loaded into memory, thus may cause a memory overflow for large sample_size
def sample(infile, sample_size):
    assert os.path.isfile(infile), "Provided input file ("+infile+") is not a valid file, aborting."
    file = h5py.File(infile)
    print("Successfully loaded dataset from", infile)
    data = file['event_data']
    labels = file['labels']
    
    event_size = labels.size
    if sample_size > event_size:
        sample_size = event_size
        
    sample_idx = np.random.randint(low=0, high=event_size-1, size=sample_size)
    sample_data = np.asarray([data[i] for i in sample_idx])
    
    file.close()
    
    return (sample_data, event_size)

# Dump a set of histograms from a single dataset
def plot_all(infile, outpath, sample_size, bins, show=False):
    outpath += '' if outpath.endswith('/') else '/'
    if not os.path.isdir(outpath):
        print("Making output directory for plots as", outpath)
        os.mkdir(outpath)
    sample_data, event_size = sample(infile, sample_size)
    sample_size = min(sample_size, event_size)
    print("Successfully prepared sampling subset of", sample_size, "events.")
    
    sample_chrg = sample_data[:,:,:,:19].reshape(-1,1)
    sample_chrg_nonzero = sample_chrg[sample_chrg > 0]
    sample_time = sample_data[:,:,:,19:].reshape(-1,1)
    sample_time_nonzero = sample_time[sample_time > 0]
    
    SAMPLE_RATIO = str(sample_size)+" of "+str(event_size)
    
    # Full histogram of PMT hit charges over all sampled events
    plot_single_hist(sample_chrg, bins, outpath, 0, title="Charge Distribution Over All Event Types (Sampling "+SAMPLE_RATIO+" events)",
                     xlabel="Charge", right=SCALE_X*np.amax(sample_chrg_nonzero), ylabel="Hits", top=Y_CHRG, show=show)
    # Log-scaled version of above
    plot_single_hist(sample_chrg, bins, outpath, 1, title="Log-Scaled Charge Distribution Over All Event Types (Sampling "+SAMPLE_RATIO+" events)",
                     xlabel="Charge", right=SCALE_X*np.amax(sample_chrg_nonzero), ylabel="log(Hits)", yscale="log", top=Y_CHRG, show=show)
    
    # Full histogram of PMT hit timing over all sampled events
    plot_single_hist(sample_time, bins, outpath, 2, title="Timing Distribution Over All Event Types (Sampling "+SAMPLE_RATIO+" events)",
                     xlabel="Time", left=max(np.amin(sample_time_nonzero)-10, 0), right=np.amax(sample_time_nonzero), ylabel="Hits",
                     top=Y_TIME, show=show)
    # Log-scaled version of above
    plot_single_hist(sample_time, bins, outpath, 3, title="Log-Scaled Timing Distribution Over All Event Types (Sampling "+SAMPLE_RATIO+" events)",
                     xlabel="Time", left=max(np.amin(sample_time_nonzero)-10, 0), right=np.amax(sample_time_nonzero), ylabel="log(Hits)", yscale="log",
                     top=Y_TIME, show=show)

# Dump a set of histograms of multiple datasets overlaid
def plot_overlaid(files, outpath, sample_size, bins, show=False):
    outpath += '' if outpath.endswith('/') else '/'
    if not os.path.isdir(outpath):
        print("Making output directory for plots as", outpath)
        os.mkdir(outpath)
    print("Overlaying datasets:", files)
    
    chrg_dsets = []
    min_chrg = LARGE
    max_chrg = 0
    time_dsets = []
    min_time = LARGE
    max_time = 0
    for file in files:
        filename = os.path.basename(file)
        sample_data, event_size = sample(file, sample_size)
        chrg_data = sample_data[:,:,:,:19].reshape(-1,1)
        time_data = sample_data[:,:,:,19:].reshape(-1,1)
        chrg_dsets.append((filename, chrg_data))
        time_dsets.append((filename, time_data))
        
        # Update min and max counters
        min_chrg = min(np.amin(chrg_data[chrg_data>0]), min_chrg)
        max_chrg = max(np.amax(chrg_data), max_chrg)
        min_time = min(np.amin(time_data[time_data>0]), min_time)
        max_time = max(np.amax(time_data), max_time)
        
        print("Successfully prepared sampling subset of", sample_size, "events from", filename)
    # Overlaid charge histograms of normalization schemes
    plot_overlaid_hist(chrg_dsets, bins, outpath, 0,
                       title="Charge distributions of all normalization schemes (sample size "+str(sample_size)+" for all datasets)",
                       xlabel="Charge", right=SCALE_X*max_chrg, ylabel="Hits", top=Y_CHRG, show=show)
    # Log-scaled version of above
    plot_overlaid_hist(chrg_dsets, bins, outpath, 1,
                       title="Log-scaled charge distributions of all normalization schemes (sample size "+str(sample_size)+" for all datasets)",
                       xlabel="Charge", right=SCALE_X*max_chrg, ylabel="log(Hits)", top=Y_CHRG, yscale='log', show=show)
    
    # Overlaid timing histograms of normalization schemes
    plot_overlaid_hist(time_dsets, bins, outpath, 2,
                       title="Timing distributions of all normalization schemes (sample size "+str(sample_size)+" for all datasets)",
                       xlabel="Timing", left=max(min_time-10, 0), right=max_time, ylabel="Hits", top=Y_TIME, show=show)
    # Log-scaled version of above
    plot_overlaid_hist(time_dsets, bins, outpath, 3,
                       title="Log-scaled timing distributions of all normalization schemes (sample size "+str(sample_size)+" for all datasets)",
                       xlabel="Timing", left=max(min_time-10, 0), right=max_time, ylabel="log(Hits)", top=Y_TIME, yscale='log', show=show)

# Helper function to create a single histogram figure
def plot_single_hist(sampled_data, bins, outpath, figure_id=0,
                     title=None, xlabel=None, ylabel=None, left=None, right=None, top=None, yscale=None, show=False):
    plt.figure(figure_id)
    plt.hist(sampled_data, bins, linewidth=0)
    if title is not None: plt.title('\n'.join(wrap(title,60)), fontsize=FONTSIZE)
    if left is not None: plt.xlim(left=left)
    if right is not None: plt.xlim(right=right)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if yscale is not None: plt.yscale(yscale)
    else: plt.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    if top is not None: plt.ylim(top=top)
    plt.savefig(outpath+title+EXT)
    print("Saved", '"'+title+'"', "to", outpath)
    if show: plt.show()

# Helper function to create a single histogram figure overlaid with multiple samples
def plot_overlaid_hist(dsets, bins, outpath, figure_id=0,
                  title=None, xlabel=None, ylabel=None, left=None, right=None, top=None, yscale=None, show=False):
    plt.figure(figure_id)
    for name, dset in dsets:
        plt.hist(dset, bins, alpha=0.3, linewidth=0, label=name)
    plt.legend(loc="upper right")
    if title is not None: plt.title('\n'.join(wrap(title,60)), fontsize=FONTSIZE)
    if left is not None: plt.xlim(left=left)
    if right is not None: plt.xlim(right=right)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if yscale is not None: plt.yscale(yscale)
    else: plt.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    if top is not None: plt.ylim(top=top)
    plt.savefig(outpath+title+EXT)
    print("Saved", '"'+title+'"', "to", outpath)
    if show: plt.show()

# Executable behaviour
if __name__ == "__main__":
    config = parse_args()
    infile = config.input_file[0]
    outpath = config.output_path[0]
    if os.path.isfile(infile):
        plot_all(infile, outpath, config.sample_size, config.num_bins,
                 show=(config.show_plts is not None))
    if os.path.isdir(infile):
        dset_files = [os.path.join(infile, file) for file in os.listdir(infile) if file.endswith('.h5')]
        plot_overlaid(dset_files, outpath, config.sample_size, config.num_bins,
                      show=(config.show_plts is not None))