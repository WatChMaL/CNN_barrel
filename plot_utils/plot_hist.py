"""
Script for plotting charge and time histograms of HDF5 event dataset

Axes scales are set as such:
    - y-axis: 0 to SCALE_Y*maximum frequency in dataset
    - x-axis: lowest value to reach frequency of SCALE_X to highest value to reach frequency of SCALE_X

Author: Julian Ding
"""

import os
import argparse
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from textwrap import wrap
import h5py

# Available plotting tasks
TASKS = ['c', 't', 'p', 'h', 'e']

# File extension to save histograms as
EXT = '.pdf'

# Constants to set axes scales
SCALE_X = 5
SCALE_Y = 1

# Large initialization constant for minimizers
LARGE = 1e10

# Font sizes
FONT_TITLE = 10
FONT_LEGEND = 5

# Names corresponding to labels
CLASSES = ['gamma', 'e', 'mu']

# Scientific notation formatting
EPS = np.finfo(np.double).eps

# Argument parsing for commandline functionality
def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalizes data from an input HDF5 file and outputs to HDF5 file.")
    parser.add_argument("--input_file", '-in', dest="input_file", type=str, nargs='+',
                        help="path to dataset(s) to visualize", required=True)
    parser.add_argument('--output_path', '-out', dest="output_path", type=str, nargs=1,
                        help="desired output path", required=True)
    parser.add_argument('--sample_size', '-sub', dest="sample_size", type=int, default=10000,
                        help="number of events to sample from dataset", required=False)
    parser.add_argument('--num_bins', '-bin', dest='num_bins', type=int, default=1000,
                        help="number of bins to separate data into in histograms", required=False)
    parser.add_argument('--to_plot', '-plt', dest='to_plot', type=str, nargs='+', default=TASKS,
                        help="specify data to plot: c=charge, t=time, p=overlaid particle types, h=hit frequency, e=overlay single events", required=False)
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
    
    tol = ceil(0.01*sample_size)
    
    _, counts = np.unique(labels, return_counts=True)
    if counts.size == 3 and all(c >= (sample_size//3) for c in counts):
        while True:
            sample_idx = np.random.choice(event_size, size=sample_size, replace=False)
            sample_labels = np.asarray([labels[i] for i in sample_idx])
            d_01 = abs(sample_labels[sample_labels == 0].size - sample_labels[sample_labels == 1].size)
            d_12 = abs(sample_labels[sample_labels == 0].size - sample_labels[sample_labels == 1].size)
            d_02 = abs(sample_labels[sample_labels == 0].size - sample_labels[sample_labels == 2].size)
            if d_01 <= tol and d_12 <= tol and d_02 <= tol:
                break
            else: print("Resampling:", d_01, d_12, d_02, ">", tol)
    else:
        print("Warning: uneven class distribution detected in dataset", {CLASSES[i] : counts[i] for i in range(counts.size)},
                                                                   "\n...Performing completely random sampling")
        sample_idx = np.random.choice(event_size, size=sample_size, replace=False)
        sample_labels = np.asarray([labels[i] for i in sample_idx])
    
    sample_data = np.asarray([data[i] for i in sample_idx])
    file.close()
    
    return (sample_data, sample_labels, event_size)

# Dump a set of histograms from a single dataset
def plot_all(infile, outpath, sample_size, bins, to_plot, show=False):
    outpath += '' if outpath.endswith('/') else '/'
    if not os.path.isdir(outpath):
        print("Making output directory for plots as", outpath)
        os.mkdir(outpath)
    sample_data, sample_labels, event_size = sample(infile, sample_size)
    sample_size = min(sample_size, event_size)
    print("Successfully prepared sampling subset of", sample_size, "events.")
    
    sample_chrg = sample_data[:,:,:,:19].reshape(-1,1)
    sample_time = sample_data[:,:,:,19:].reshape(-1,1)
    
    fig_id = 0
    
    if 'p' in to_plot or 'e' in to_plot:
        class_data = [sample_data[sample_labels == i] for i in range(len(CLASSES))]
    
    sample_ratio_str = str(sample_size).format(EPS)+" of "+str(event_size).format(EPS)
    
    if 'c' in to_plot:
        title = "Charge: "
        # Full histogram of PMT hit charges over all sampled events
        plot_single_hist(sample_chrg, bins, outpath, fig_id, title=title+"Distribution Over All Event Types (Sampling "+sample_ratio_str+" events)",
                         xlabel="Charge", ylabel="Hits", show=show)
        fig_id += 1
        # Log-scaled version of above
        plot_single_hist(sample_chrg, bins, outpath, fig_id, title=title+"Log-Scaled Distribution Over All Event Types (Sampling "+sample_ratio_str+" events)",
                         xlabel="Charge", ylabel="Hits (log-scaled)", yscale="log", show=show)
        fig_id += 1

        if 'p' in to_plot:
            charge_dsets = [(CLASSES[i], dset[:,:,:,:19].reshape(-1,1)) for i, dset in enumerate(class_data)]
            # Overlaid histogram of PMT hit over particle classes
            plot_overlaid_hist(charge_dsets, bins, outpath, fig_id,
                               title=title+"Overlaid Distribution of All Event Types (Sampling "+sample_ratio_str+" events)",
                               xlabel="Charge", ylabel="Hits", show=show)
            fig_id += 1
            # Log-scaled version of above
            plot_overlaid_hist(charge_dsets, bins, outpath, fig_id,
                               title=title+"Log-scaled Overlaid Distribution of All Event Types (Sampling "+sample_ratio_str+" events)",
                               xlabel="Charge", ylabel="Hits (log-scaled)", yscale='log', show=show)
            fig_id += 1
            
        if 'e' in to_plot:
            # Reshape data to (events, pmts per event)
            flat_gamma = class_data[0][:,:,:,:19].reshape(class_data[0].shape[0],-1)
            flat_electron = class_data[1][:,:,:,:19].reshape(class_data[1].shape[0],-1)
            flat_muon = class_data[2][:,:,:,:19].reshape(class_data[2].shape[0],-1)
            
            # Plot individual overlaid events of each class
            # Gamma
            plot_overlaid_events(flat_gamma, bins, outpath, figure_id=fig_id, title=title+"Overlaid Individual Gamma Events (Sampling "+sample_ratio_str+" events)",
                                 xlabel="Charge", ylabel="Hits", show=show)
            fig_id += 1
            # Log-scaled Gamma
            plot_overlaid_events(flat_gamma, bins, outpath, figure_id=fig_id, title=title+"Log-scaled Overlaid Individual Gamma Events (Sampling "+sample_ratio_str+" events)",
                                 xlabel="Charge", ylabel="Hits", yscale='log', show=show)
            fig_id += 1
            
            # Electron
            plot_overlaid_events(flat_electron, bins, outpath, figure_id=fig_id, title=title+"Overlaid Individual Electron Events (Sampling "+sample_ratio_str+" events)",
                                 xlabel="Charge", ylabel="Hits", show=show)
            fig_id += 1
            # Log-scaled Electron
            plot_overlaid_events(flat_electron, bins, outpath, figure_id=fig_id, title=title+"Log-scaled Overlaid Individual Electron Events (Sampling "+sample_ratio_str+" events)",
                                 xlabel="Charge", ylabel="Hits", yscale='log', show=show)
            fig_id += 1
            
            # Muon
            plot_overlaid_events(flat_muon, bins, outpath, figure_id=fig_id, title=title+"Overlaid Individual Muon Events (Sampling "+sample_ratio_str+" events)",
                                 xlabel="Charge", ylabel="Hits", show=show)
            fig_id += 1
            # Log-scaled Muon
            plot_overlaid_events(flat_muon, bins, outpath, figure_id=fig_id, title=title+"Log-scaled Overlaid Individual Muon Events (Sampling "+sample_ratio_str+" events)",
                                 xlabel="Charge", ylabel="Hits", yscale='log', show=show)
            fig_id += 1
            
    
    if 't' in to_plot:
        title = "Timing: "
        # Full histogram of PMT hit timing over all sampled events
        plot_single_hist(sample_time, bins, outpath, fig_id, title=title+"Distribution Over All Event Types (Sampling "+sample_ratio_str+" events)",
                         xlabel="Time", ylabel="Hits", show=show)
        fig_id += 1
        # Log-scaled version of above
        plot_single_hist(sample_time, bins, outpath, fig_id, title=title+"Log-Scaled Distribution Over All Event Types (Sampling "+sample_ratio_str+" events)",
                         xlabel="Time", ylabel="Hits (log-scaled)", yscale="log", show=show)
        fig_id += 1
        
        if 'p' in to_plot:
            time_dsets = [(CLASSES[i], dset[:,:,:,19:].reshape(-1,1)) for i, dset in enumerate(class_data)]
            # Overlaid histogram of PMT hit over particle classes
            plot_overlaid_hist(time_dsets, bins, outpath, fig_id,
                               title=title+"Overlaid Distribution of All Event Types (Sampling "+sample_ratio_str+" events)",
                               xlabel=title+"Time", ylabel="Hits", show=show)
            fig_id += 1
            # Log-scaled version of above
            plot_overlaid_hist(time_dsets, bins, outpath, fig_id,
                               title=title+"Log-scaled Overlaid Distribution of All Event Types (Sampling "+sample_ratio_str+" events)",
                               xlabel="Time", ylabel="Hits (log-scaled)", yscale='log', show=show)
            fig_id += 1
            
        if 'e' in to_plot:
            # Reshape data to (events, pmts per event)
            flat_gamma = class_data[0][:,:,:,19:].reshape(class_data[0].shape[0],-1)
            flat_electron = class_data[1][:,:,:,19:].reshape(class_data[1].shape[0],-1)
            flat_muon = class_data[2][:,:,:,:19:].reshape(class_data[2].shape[0],-1)
            
            # Plot individual overlaid events of each class
            # Gamma
            plot_overlaid_events(flat_gamma, bins, outpath, figure_id=fig_id, title=title+"Overlaid Individual Gamma Events (Sampling "+sample_ratio_str+" events)",
                                 xlabel="Timing", ylabel="Hits", show=show)
            fig_id += 1
            # Log-scaled Gamma
            plot_overlaid_events(flat_gamma, bins, outpath, figure_id=fig_id, title=title+"Log-scaled Overlaid Individual Gamma Events (Sampling "+sample_ratio_str+" events)",
                                 xlabel="Timing", ylabel="Hits", yscale='log', show=show)
            fig_id += 1
            
            # Electron
            plot_overlaid_events(flat_electron, bins, outpath, figure_id=fig_id, title=title+"Overlaid Individual Electron Events (Sampling "+sample_ratio_str+" events)",
                                 xlabel="Timing", ylabel="Hits", show=show)
            fig_id += 1
            # Log-scaled Electron
            plot_overlaid_events(flat_electron, bins, outpath, figure_id=fig_id, title=title+"Log-scaled Overlaid Individual Electron Events (Sampling "+sample_ratio_str+" events)",
                                 xlabel="Timing", ylabel="Hits", yscale='log', show=show)
            fig_id += 1
            
            # Muon
            plot_overlaid_events(flat_muon, bins, outpath, figure_id=fig_id, title=title+"Overlaid Individual Muon Events (Sampling "+sample_ratio_str+" events)",
                                 xlabel="Timing", ylabel="Hits", show=show)
            fig_id += 1
            # Log-scaled Muon
            plot_overlaid_events(flat_muon, bins, outpath, figure_id=fig_id, title=title+"Log-scaled Overlaid Individual Muon Events (Sampling "+sample_ratio_str+" events)",
                                 xlabel="Timing", ylabel="Hits", yscale='log', show=show)
            fig_id += 1
    
    if 'h' in to_plot:
        # Extract vector representing the number of hit PMTs for each event in sample_data
        hits_per_event = sample_data[:,:,:,:19].reshape(sample_data.shape[0], -1)
        hits_data = np.asarray([np.count_nonzero(event) for event in hits_per_event])
        hit_dsets = [(CLASSES[i], hits_data[sample_labels == i]) for i in range(len(CLASSES))]
        # Plot overlaid histogram of different event classes
        plot_overlaid_hist(hit_dsets, bins, outpath, fig_id, title="Overlaid Hit PMTs per Event Histogram (Sampling "+sample_ratio_str+" events)",
                           xlabel="Hits", ylabel="Events", show=show)
        fig_id += 1
        # Log-scaled version of above
        plot_overlaid_hist(hit_dsets, bins, outpath, fig_id, title="Log-scaled Overlaid Hit PMTs per Event Histogram (Sampling "+sample_ratio_str+" events)",
                           xlabel="Hits", ylabel="Events", yscale='log', show=show)
        fig_id += 1
        

# Dump a set of histograms of multiple datasets overlaid
def plot_overlaid_dsets(files, outpath, sample_size, bins, to_plot, show=False):
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
        sample_data, _, event_size = sample(file, sample_size)
        chrg_data = sample_data[:,:,:,:19].reshape(-1,1)
        time_data = sample_data[:,:,:,19:].reshape(-1,1)
        chrg_dsets.append((filename, chrg_data))
        time_dsets.append((filename, time_data))
        
        # Update min and max counters
        min_chrg = min(np.amin(chrg_data[chrg_data>0]), min_chrg)
        max_chrg = max(np.amax(chrg_data), max_chrg)
        min_time = min(np.amin(time_data[time_data>0]), min_time)
        max_time = max(np.amax(time_data), max_time)
        
        sample_size = min(sample_size, event_size)
        size_str = str(sample_size).format(EPS)
        print("Successfully prepared sampling subset of", sample_size, "events from", filename)
        
    if 'c' in to_plot:
        # Overlaid charge histograms of normalization schemes
        plot_overlaid_hist(chrg_dsets, bins, outpath, 0,
                           title="Charge distributions of multiple normalization schemes (sample size "+size_str+" for all datasets)",
                           xlabel="Charge", ylabel="Hits", show=show)
        # Log-scaled version of above
        plot_overlaid_hist(chrg_dsets, bins, outpath, 1,
                           title="Log-scaled charge distributions of multiple normalization schemes (sample size "+size_str+" for all datasets)",
                           xlabel="Charge", ylabel="Hits (log-scaled)", yscale='log', show=show)
    
    if 't' in to_plot:
        # Overlaid timing histograms of normalization schemes
        plot_overlaid_hist(time_dsets, bins, outpath, 2,
                           title="Timing distributions of multiple normalization schemes (sample size "+size_str+" for all datasets)",
                           xlabel="Timing", ylabel="Hits", show=show)
        # Log-scaled version of above
        plot_overlaid_hist(time_dsets, bins, outpath, 3,
                           title="Log-scaled timing distributions of multiple normalization schemes (sample size "+size_str+" for all datasets)",
                           xlabel="Timing", ylabel="Hits (log-scaled)", yscale='log', show=show)

# ========================= Plotting Functions ============================

# Helper function to create a single histogram figure
def plot_single_hist(sampled_data, bins, outpath, figure_id=0,
                     title=None, xlabel=None, ylabel=None, yscale=None, show=False):
    plt.figure(figure_id)
    histogram, edges, _ = plt.hist(sampled_data, bins, histtype='step', fill=False)
    # Disregard zero hits
    histogram[0] = 0
    if title is not None: plt.title('\n'.join(wrap(title,60)), fontsize=FONT_TITLE)
    
    # Calculate and set left and right limits of dataset
    valids = np.arange(len(edges)-1)[histogram > SCALE_X]
    left = edges[valids[0]]
    right = edges[valids[-1]]
    plt.xlim(left=left)
    plt.xlim(right=right)
    
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if yscale is not None: plt.yscale(yscale)
    else: plt.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    plt.ylim(top=SCALE_Y*np.amax(histogram))
    plt.savefig(outpath+title+EXT)
    print("Saved", '"'+title+'"', "to", outpath)
    if show: plt.show()
    
    plt.close(fig=figure_id)

# Helper function to create a single histogram figure overlaid with multiple samples
# Requires: dsets is a list of tuples (name, 1D flat data array)
def plot_overlaid_hist(dsets, bins, outpath, figure_id=0,
                  title=None, xlabel=None, ylabel=None, yscale=None, show=False):
    plt.figure(figure_id)
    lefts, rights, tops = [], [], []
    for name, dset in dsets:
        histogram, edges, _ = plt.hist(dset, bins, histtype='step', fill=False, label=name)
        # Disregard zero hits
        histogram[0] = 0
        valids = np.arange(len(edges)-1)[histogram > SCALE_X]
        lefts.append(edges[valids[0]])
        rights.append(edges[valids[-1]])
        tops.append(SCALE_Y*np.amax(histogram))

    if title is not None: plt.title('\n'.join(wrap(title,60)), fontsize=FONT_TITLE)
    plt.legend(loc="upper right", fontsize=FONT_LEGEND)
    plt.xlim(left=min(lefts))
    plt.xlim(right=max(rights))
    
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if yscale is not None: plt.yscale(yscale)
    else: plt.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    plt.ylim(top=max(tops))
    plt.savefig(outpath+title+EXT)
    print("Saved", '"'+title+'"', "to", outpath)
    if show: plt.show()
    
    plt.close(fig=figure_id)
    
# Helper function to overlay histograms of individual events onto one big histogram
def plot_overlaid_events(flat_data, bins, outpath, figure_id=0,
                         title=None, xlabel=None, ylabel=None, yscale=None, show=False):
    plt.figure(figure_id)
    lefts, rights, tops = [], [], []
    for event in flat_data:
        histogram, edges, _ = plt.hist(event, bins, histtype='step', linewidth=1, fill=False)
        # Disregard zero hits
        histogram[0] = 0
        valids = np.arange(len(edges)-1)[histogram > 0]
        lefts.append(edges[valids[0]])
        rights.append(edges[valids[-1]])
        tops.append(SCALE_Y*np.amax(histogram))
        
    if title is not None: plt.title('\n'.join(wrap(title,60)), fontsize=FONT_TITLE)
    plt.xlim(left=min(lefts))
    plt.xlim(right=max(rights))

    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if yscale is not None: plt.yscale(yscale)
    plt.ylim(top=max(tops))
    plt.savefig(outpath+title+EXT)
    print("Saved", '"'+title+'"', "to", outpath)
    if show: plt.show()
    
    plt.close(fig=figure_id)

# Executable behaviour
if __name__ == "__main__":
    config = parse_args()
    infile = config.input_file[0]
    outpath = config.output_path[0]
    if len(config.input_file) > 1:
        plot_overlaid_dsets(config.input_file, outpath, config.sample_size, config.num_bins, config.to_plot,
                      show=(config.show_plts is not None))
    elif os.path.isdir(infile):
        dset_files = [os.path.join(infile, file) for file in os.listdir(infile) if file.endswith('.h5')]
        plot_overlaid_dsets(dset_files, outpath, config.sample_size, config.num_bins, config.to_plot,
                      show=(config.show_plts is not None))
    else:
        plot_all(infile, outpath, config.sample_size, config.num_bins, config.to_plot,
                 show=(config.show_plts is not None))