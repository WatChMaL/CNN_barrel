"""
Source code borrowed from https://github.com/WatChMaL/UVicWorkshopPlayground/blob/master/B/notebooks/utils/utils.py
Edited by : Abhishek .
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from math import sqrt

import plot_utils.mpmt_visual as mpmt_visual

# Set the style
plt.style.use("classic")

# Fix the colour scheme for each particle type
color_dict = {"gamma":"red", "e":"blue", "mu":"black"}

# Function to convert from the true particle energies to visible energies
# E_vis = gamma*mc^2 - gamma(at Cherenkov speed)*mc^2
#       = total energy - energy at Cherenkov threshold
def convert_to_visible_energy(energies, labels):
    
    """
    convert_to_visible_energy(energies, labels)
    
    Purpose : Convert the true event energies to visible energy collected by the PMTs
    
    Args: energies ... 1D array of event energies, the length = sample size
          labels   ... 1D array of true label value, the length = sample size
    """
    
    # Convert true particle energies to visible energies
    m_mu = 105.7
    m_e = 0.511
    m_p = 0.511

    # Constant for the inverse refractive index of water
    beta = 0.75

    # Denominator for the scaling factor to be used for the cherenkov threshold
    dem = sqrt(1 - beta**2)
    
    # Perform the conversion from true particle energy to visible energy
    energies[labels == 0] -= (m_e + m_p)/dem
    energies[labels == 1] -= m_e/dem
    energies[labels == 2] -= m_mu/dem
        
    return energies.clip(0)

# Function to plot the energy distribution over a given dataset
def plot_event_energy_distribution(energies, labels, label_dict, dset_type="full", show_plot=False, save_path=None):
    
    """
    plot_confusion_matrix(labels, predictions, energies, class_names, min_energy, max_energy, save_path=None)
    
    Purpose : Plot the confusion matrix for a given energy interval
    
    Args: energies            ... 1D array of event energies, the length = sample size
          labels              ... 1D array of true label value, the length = sample size
          labels_dict         ... Dictionary with the keys as event types and values as labels, default=None
          dset_type           ... String describing the type of dataset (full, train, validation, train), default="full"
          show_plot[optional] ... Boolean to determine whether to display the plot, default=False
          save_path[optional] ... Path to save the plot as an image, default=None
    """
    # Assertions
    assert label_dict is not None
    
    # Extract the event energies corresponding to given event types
    energies_dict = {}
    for key in label_dict.keys():
        energies_dict[key] = energies[labels==label_dict[key]]
        
    fig, axes = plt.subplots(3,1,figsize=(16,12))
    plt.subplots_adjust(hspace=0.6)
    
    for label in energies_dict.keys():
        label_to_use = r"$\{0}$".format(label) if label is not "e" else r"${0}$".format(label)
        
        axes[label_dict[label]].hist(energies_dict[label], bins=50, histtype='step', linewidth=3, fill=False,
            label=label_to_use, color=color_dict[label])
        axes[label_dict[label]].tick_params(labelsize=20)
        axes[label_dict[label]].legend(prop={"size":20})
        axes[label_dict[label]].grid(True, which="both", axis="both")
        axes[label_dict[label]].set_ylabel("Frequency", fontsize=20)
        axes[label_dict[label]].set_xlabel("Event Visible Energy (MeV)", fontsize=20)
        axes[label_dict[label]].set_xlim(0, max(energies)+20)
        axes[label_dict[label]].set_title("Energy distribution for " + label_to_use + " over the " + dset_type + " dataset",
                             fontsize=20)
        
    if save_path is not None:
        plt.savefig(save_path)
    
    if show_plot:
        plt.show()

    plt.clf() # Clear the plot frame
    plt.close() # Close the opened window if any


# Function to plot a confusion matrix
def plot_confusion_matrix(labels, predictions, energies, class_names, min_energy=0, max_energy=1500, 
                          show_plot=False, save_path=None):
    
    """
    plot_confusion_matrix(labels, predictions, energies, class_names, min_energy, max_energy, save_path=None)
    
    Purpose : Plot the confusion matrix for a given energy interval
    
    Args: labels              ... 1D array of true label value, the length = sample size
          predictions         ... 1D array of predictions, the length = sample size
          energies            ... 1D array of event energies, the length = sample size
          class_names         ... 1D array of string label for classification targets, the length = number of categories
          min_energy          ... Minimum energy for the events to consider
          max_energy          ... Maximum energy for the events to consider
          show_plot[optional] ... Boolean to determine whether to display the plot
          save_path[optional] ... Path to save the plot as an image
    """
    
    # Create a mapping to extract the energies in
    energy_slice_map = [False for i in range(len(energies))]
    for i in range(len(energies)):
        if(energies[i] >= min_energy and energies[i] < max_energy):
                energy_slice_map[i] = True
                
    # Filter the CNN outputs based on the energy intervals
    labels = labels[energy_slice_map]
    predictions = predictions[energy_slice_map]
    
    if(show_plot or save_path is not None):
        fig, ax = plt.subplots(figsize=(12,10),facecolor='w')
        num_labels = len(class_names)
        max_value = np.max([np.max(np.unique(labels)),np.max(np.unique(labels))])
        assert max_value < num_labels
        mat,_,_,im = ax.hist2d(predictions, labels,
                               bins=(num_labels,num_labels),
                               range=((-0.5,num_labels-0.5),(-0.5,num_labels-0.5)),cmap=plt.cm.Blues)

        # Normalize the confusion matrix
        mat = mat.astype("float") / mat.sum(axis=0)[:, np.newaxis]

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=20) 
        
        ax.set_xticks(np.arange(num_labels))
        ax.set_yticks(np.arange(num_labels))
        ax.set_xticklabels(class_names,fontsize=20)
        ax.set_yticklabels(class_names,fontsize=20)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_xlabel('Prediction',fontsize=20)
        ax.set_ylabel('True Label',fontsize=20)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(i,j, r"${0:0.3f}$".format(mat[i,j]),
                        ha="center", va="center", fontsize=20,
                        color="white" if mat[i,j] > (0.5*mat.max()) else "black")
        fig.tight_layout()
        plt.title("Confusion matrix, " + r"${0} \leq E < {1}$".format(round(min_energy, 2), round(max_energy, 2)), fontsize=20)        
        plt.subplots_adjust(top=0.9)
   
    if save_path is not None:
        plt.savefig(save_path)
        
    if show_plot:
        plt.show()
    else:
        plt.clf() # Clear the plot frame
        plt.close() # Close the opened window if any

# Plot the classifier for a given event type for several true event types
def plot_classifier_response(softmaxes, labels, energies, softmax_index_dict, event_dict, min_energy=0,
                             max_energy=1500, num_bins=100, show_plot=False, save_path=None):
    
    """
    plot_classifier_response(softmaxes, labels, energies, softmax_index_dict, event, min_energy=0,
                             max_energy=1000, num_bins=100, show_plot=False, save_path=None)
                             
    Purpose : Plot the classifier softmax response for a given event type for several true event types
    
    Args: softmaxes             ... 2D array of softmax output, length = sample size,
                                    dimensions = (n_samples, n_classes)
          labels                ... 1D array of true labels
          energies              ... 1D array of visible event energies
          softmax_index_dict    ... Dictionary with the keys as event types and values as column 
                                    indices in the softmax array, default=None
          event_dict            ... Dictionary with the softmax class as the key and column indices
                                    in the softmax array as the values
          min_energy            ... Minimum energy for the events to consider, default=0
          max_energy            ... Maximum energy for the events to consider, default=1000
          num_bins[optional]    ... Number of bins to use per histogram, default=100
          show_plot[optional]   ... Boolean to determine whether to show the plot, default=False
          save_path[optional]   ... Path to save the plot to, format='eps', default=None
          
    """
    
    assert softmaxes is not None and softmaxes.any() != None
    assert labels is not None and labels.any() != None
    assert energies is not None
    
    # Initialize the plot and corresponding parameters
    fig, ax = plt.subplots(figsize=(12,8),facecolor="w")
    ax.tick_params(axis="both", labelsize=20)
    
    # Get the softmax output class for which to plot the response
    event = list(event_dict.keys())[0]
    
    for event_type in softmax_index_dict.keys():
        
        label_to_use = r"$\{0}$ events".format(event_type) if event_type is not "e" else r"${0}$ events".format(event_type)
        
        # Get the softmax values for the given true event label
        label_map = [False for i in range(len(labels))]
        for i in range(len(labels)):
            if( labels[i] == softmax_index_dict[event_type] ):
                label_map[i] = True
        
        # Get the softmax values for the given true event label
        curr_softmax = softmaxes[label_map]

        # Get the energy values for the given true event label
        curr_energies = energies[label_map]

        # Create a mapping to extract the energies in
        energy_slice_map = [False for i in range(len(curr_energies))]
        for i in range(len(curr_energies)):
            if(curr_energies[i] >= min_energy and curr_energies[i] < max_energy):
                    energy_slice_map[i] = True

        # Filter the CNN outputs based on the energy intervals
        curr_softmax = curr_softmax[energy_slice_map]
        curr_softmax = curr_softmax[:,event_dict[event]]
        
        if(curr_softmax.shape[0] <= 0):
            return None, None, None
        else:
            values, bins, patches = plt.hist(curr_softmax, bins=num_bins, histtype='step', linewidth=3, fill=False,
                                             label= label_to_use, color=color_dict[event_type])
        
    if save_path is not None or show_plot:
        ax.grid(True)
        if event is not "e":
            ax.set_xlabel(r"Classifier softmax output : $P(\{0})$".format(event), fontsize=20)
        else:
            ax.set_xlabel(r"Classifier softmax output : $P(e)$".format(event), fontsize=20)

        ax.set_ylabel("Count (Log scaled)", fontsize=20)
        plt.yscale("log")

        ax.set_xlim(0,1)

        plt.legend(loc="upper left", prop={"size":20})
        
        plt.title(r"${0} \leq E < {1}$".format(round(min_energy,2), round(max_energy,2)), fontsize=20)
        
    if save_path is not None:
        plt.savefig(save_path)
        
    if show_plot:
        plt.show()
        
    plt.clf() # Clear the current figure
    plt.close() # Close the opened window
        
    return values, bins, patches

# Plot the ROC curve for one vs another class
def plot_ROC_curve_one_vs_one(softmaxes, labels, energies, softmax_index_dict, label_0, label_1, min_energy=0,
                              max_energy=1500, show_plot=False, save_path=None, inverse=False):
    """
    plot_ROC_curve_one_vs_one(softmaxes, labels, energies, softmax_index_dict, 
                              min_energy, max_energy, show_plot=False, save_path=None)
                              
    Purpose : Plot the Reciver Operating Characteristic (ROC) curve given the softmax values and true labels
                              
    Args: softmaxes             ... 2D array of softmax output, length = sample size, dimensions = n_samples, n_classes
          labels                ... 1D array of true labels
          energies              ... 1D array of visible event energies
          softmax_index_dict    ... Dictionary with the keys as event type (str) and values as the column indices 
                                    in the np softmax array
          label_0               ... Event type for which to plot the ROC for
          label_1               ... Event type for which to plot the ROC against
          min_energy            ... Minimum energy for the events to consider, default=0
          max_energy            ... Maximum energy for the events to consider, default=1000
          show_plot[optional]   ... Boolean to determine whether to show the plot, default=False
          save_path[optional]   ... Path to save the plot to, format='eps', default=None
    """
    
    assert softmaxes is not None
    assert labels is not None
    assert softmax_index_dict  is not None
    assert softmaxes.shape[0] == labels.shape[0]
    assert label_0 in softmax_index_dict.keys()
    assert label_1 in softmax_index_dict.keys()
    
    # Create a mapping to extract the energies in
    energy_slice_map = [False for i in range(len(energies))]
    for i in range(len(energies)):
        if(energies[i] >= min_energy and energies[i] < max_energy):
                energy_slice_map[i] = True
                
    # Filter the CNN outputs based on the energy intervals
    curr_softmax = softmaxes[energy_slice_map]
    curr_labels = labels[energy_slice_map]
    
    # Extract the useful softmax and labels from the input arrays
    softmax_0 = curr_softmax[curr_labels==softmax_index_dict[label_0]]# or 
    labels_0 = curr_labels[curr_labels==softmax_index_dict[label_0]] #or 
    
    softmax_1 = curr_softmax[curr_labels==softmax_index_dict[label_1]]
    labels_1 = curr_labels[curr_labels==softmax_index_dict[label_1]]
    
    # Add the two arrays
    softmax = np.concatenate((softmax_0, softmax_1), axis=0)
    labels = np.concatenate((labels_0, labels_1), axis=0)
    
    # Binarize the labels
    binary_labels_1 = label_binarize(labels, classes=[softmax_index_dict[label_0], softmax_index_dict[label_1]])
    binary_labels_0 = 1 - binary_labels_1

    # Compute the ROC curve and the AUC for class corresponding to label 0
    fpr_0, tpr_0, threshold_0 = roc_curve(binary_labels_0, softmax[:,softmax_index_dict[label_0]])
    
    inv_fpr_0 = []
    for i in fpr_0:
        inv_fpr_0.append(1/i) if i != 0 else inv_fpr_0.append(1/1e-3)
        
    roc_auc_0 = auc(fpr_0, tpr_0)
    
    # Compute the ROC curve and the AUC for class corresponding to label 1
    fpr_1, tpr_1, threshold_1 = roc_curve(binary_labels_1, softmax[:,softmax_index_dict[label_1]])
    
    inv_fpr_1 = []
    for i in fpr_1:
        inv_fpr_1.append(1/i) if i != 0 else inv_fpr_1.append(1/1e-3)
        
    roc_auc_1 = auc(fpr_1, tpr_1)
    
    if show_plot or save_path is not None:
        # Plot the ROC curves
        if inverse:
            # 1/FPR vs TPR plot
            fig, ax = plt.subplots(figsize=(16,9),facecolor="w")
            ax.tick_params(axis="both", labelsize=20)
    
            ax.plot(tpr_0, inv_fpr_0, color=color_dict[label_0],
                     label=r"$\{0}$, AUC ${1:0.3f}$".format(label_0, roc_auc_0) if label_0 is not "e" else r"${0}$, AUC ${1:0.3f}$".format(label_0, roc_auc_0),
                     linewidth=1.0, marker=".", markersize=4.0, markerfacecolor=color_dict[label_0])
            
            # Show coords of individual points near x = 0.2, 0.5, 0.8
            todo = {0.2: True, 0.5: True, 0.8: True}
            for xy in zip(tpr_0, inv_fpr_0):
                xy = (round(xy[0], 2), round(xy[1], 2))
                for point in todo.keys():
                    if xy[0] >= point and todo[point]:
                        ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data', fontsize=18)
                        todo[point] = False
    
            ax.grid(True, which='both', color='grey')
            xlabel = r"$\{0}$ signal efficiency".format(label_0) if label_0 is not "e" else r"${0}$ signal efficiency".format(label_0)
            ylabel = r"$\{0}$ background rejection".format(label_1) if label_1 is not "e" else r"${0}$ background rejection".format(label_1)
            
            ax.set_xlabel(xlabel, fontsize=20) 
            ax.set_ylabel(ylabel, fontsize=20)
            
            ax.set_yscale("log")
            ax.set_title(r"${0} \leq E < {1}$".format(round(min_energy,2), round(max_energy,2)), fontsize=20)
            ax.legend(loc="center right", prop={"size":20})
            
        else:
            # Classic TPR vs FPR plot
            fig, ax = plt.subplots(figsize=(16,9),facecolor="w")
            ax.tick_params(axis="both", labelsize=20)
            ax.plot(fpr_0, tpr_0, color=color_dict[label_0],
                     label=r"$\{0}$, AUC ${1:0.3f}$".format(label_0, roc_auc_0) if label_0 is not "e" else r"${0}$, AUC ${1:0.3f}$".format(label_0, roc_auc_0),
                     linewidth=1.0, marker=".", markersize=4.0, markerfacecolor=color_dict[label_0])
            ax.plot(fpr_1, tpr_1, color=color_dict[label_1], 
                     label=r"$\{0}$, AUC ${1:0.3f}$".format(label_1, roc_auc_0) if label_1 is not "e" else r"${0}$, AUC ${1:0.3f}$".format(label_1, roc_auc_0),
                     linewidth=1.0, marker=".", markersize=4.0, markerfacecolor=color_dict[label_1])
            
            # Show coords of individual points near x = 0.2, 0.5, 0.8
            points = [0.2, 0.5, 0.8]
            plot_0 = zip(fpr_0, tpr_0)
            plot_1 = zip(fpr_1, tpr_1)
            coords = {point:
                {label_0: next(x for x in plot_0 if x[0] > point),
                 label_1: next(x for x in plot_1 if x[0] > point)}
                for point in points}
            for point in points:
                ax.annotate(label_0+(' (${1:0.3f}$, ${1:0.3f}$)\n'.format(coords[point][label_0][0], coords[point][label_0][1]))+label_1+(' (${1:0.3f}$, ${1:0.3f}$)'.format(coords[point][label_1][0], coords[point][label_1][1])),
                            xy=coords[point][label_0], textcoords='data', fontsize=18, va="top", ha="left", bbox=dict(boxstyle="square", fc="w"))
    
            ax.grid(True)
            ax.set_xlabel("False Positive Rate", fontsize=20)
            ax.set_ylabel("True Positive Rate", fontsize=20)
            ax.set_title(r"${0} \leq E < {1}$".format(min_energy, max_energy), fontsize=20)
            ax.legend(loc="lower right", prop={"size":20})

    if save_path is not None:
        plt.savefig(save_path)
    
    if show_plot:
        plt.show()
        
    plt.clf() # Clear the current figure
    plt.close() # Close the opened window
        
        
    return fpr_0, tpr_0, threshold_0, roc_auc_0, fpr_1, tpr_1, threshold_1, roc_auc_1

# TODO: fix
# Plot signal efficiency for a given event type at different energies
def plot_signal_efficiency(softmaxes, labels, energies, softmax_index_dict, label_0, label_1,
                           avg_efficiencies=[0.2, 0.5, 0.8], avg_efficiency_colors=None,
                           energy_interval=25, min_energy=100, max_energy=1000,
                           num_bins=100, show_plot=False, save_path=None):
    
    """
    plot_signal_efficiency(softmaxes, labels, energies, softmax_index_dict, event,
                           avg_efficiencies=[0.2, 0.5, 0.8], energy_interval=25,
                           avg_efficiency_colors=None, min_energy=100, max_energy=1000,
                           num_bins=100, show_plot=False, save_path=None)
                           
    Purpose : Plot the signal efficiency vs energy for several thresholds
    
    Args: softmaxes             ... 2D array of softmax output, length = sample size, dimensions = n_samples, n_classes
          labels                ... 1D array of true labels
          energies              ... 1D array of visible event energies
          softmax_index_dict    ... Dictionary with the keys as event type (str) and values as the column indices 
                                    in the np softmax array. Should only contain two key-value pairs.
          label_0               ... Event type for which to plot the signal efficiency for
          label_1               ... Event type for which to plot the signal efficiency against
          avg_efficiencies      ... 1D array with the average efficiency values for which to plot the signal efficiency
                                    vs energy plot, default=[0.2, 0.5, 0.8]
          avg_efficiency_colors ... Average efficiencies color dictionary to use. The keys are the iterms in the
                                    avg_efficiencies list and values are the colors to be used.
          energy_interval       ... Energy interval to be used to calculate the response curve and calculating the signal                 
                                    efficiency, default=25
          min_energy            ... Minimum energy for the events to consider, default=0
          max_energy            ... Maximum energy for the events to consider, default=1000
          num_bins              ... Number of bins to use in the classifier response histogram ( 
                                    should be greater than 100 to prevent 0 values )
          show_plot[optional]   ... Boolean to determine whether to show the plot, default=False
          save_path[optional]   ... Path to save the plot to, format='eps', default=None
    """
    
    # Need high number of bins to avoid empty values
    assert num_bins >= 100
    assert label_0 in softmax_index_dict.keys()
    assert label_1 in softmax_index_dict.keys()
    
    # Calculate the threshold here according to the desired average efficiencies
    _, _, _, _, _, tpr_1, threshold_1, _ = plot_ROC_curve_one_vs_one(softmaxes, labels, 
                                                                               energies,
                                                                               softmax_index_dict,
                                                                               label_0,
                                                                               label_1,
                                                                               min_energy,
                                                                               max_energy,
                                                                               show_plot=False)
    
    thresholds = []
    TOLERANCE = 0.25
    
    # If threshold values are specified, find associated indices
    if avg_efficiencies is not None:
        for tpr_value in avg_efficiencies:
            index_list = []
            
            for i in range(len(tpr_1)):
                if(math.fabs(tpr_1[i]-tpr_value) < 0.001):
                    index_list.append(i)
                    
            # If no threshold found near tpr_value, approximate with neighboring points
            if(len(index_list) == 0):
                lower_tpr, lower_index, upper_index, upper_tpr = 0, 0, 0, 1
                for i in range(len(tpr_1)):
                    if(tpr_1[i] < tpr_value and tpr_1[i] > lower_tpr):
                        lower_index = i
                        lower_tpr = tpr_1[i]
                    if(tpr_1[i] > tpr_value):
                        upper_index = i
                        upper_tpr = tpr_1[i]
                        break
                # If no points near enough, data is insufficient to generate meaningful plot
                if(upper_tpr - lower_tpr > TOLERANCE):
                    print("""plot_utils.plot_signal_efficiency() : Unable to calculate threshold for average_efficiency =  
                         {0}""".format(tpr_value))
                    return None
                else:
                    thresholds.append(round((threshold_1[lower_index] + threshold_1[upper_index])/2, 2))
                
            else:
                index = index_list[math.floor(len(index_list)/2)]
                thresholds.append(round(threshold_1[index], 2))
                
    # Otherwise, scan over efficiency space (0,1) for 3 representative efficiencies/thresholds
    # Search pattern: scan forward 0 -> 1 until tpr found, then scan backward 1 -> 0 until second found, repeat.
    else:
        avg_efficiencies = []
        
        rem = 3
        INCR = 0.05
        JUMP = 5
        reverse = False
        bins = int(1/INCR)
        seen_values = [False for i in range(bins)]
        
        idx = 0
        while rem > 0:
            
            index_list = []
            tpr_value = idx*INCR
        
            for i in range(len(tpr_1)):
                if(math.fabs(tpr_1[i]-tpr_value) < 0.01):
                    index_list.append(i)
            
            seen_values[idx] = True
            
            if all(seen_values):
                break
                    
            # If no threshold found near tpr_value, go to next
            if(len(index_list) == 0):
                idx += 1
            else:
                avg_efficiencies.append(tpr_value)
                index = index_list[math.floor(len(index_list)/2)]
                thresholds.append(round(threshold_1[index], 2))
                rem -= 1
                
                if reverse:
                    idx = ((bins-idx) + JUMP) % bins
                else:
                    idx = (bins - idx - JUMP - 1) % bins
                    
                while seen_values[idx]:
                    idx = (idx + 1) % bins

    # Get the energy intervals to plot the signal efficiency against ( replace with max(energies) ) 
    energy_lb = [min_energy+(energy_interval*i) for i in range(math.ceil((max_energy-min_energy)/energy_interval))]
    energy_ub = [energy_low+energy_interval for energy_low in energy_lb]
    
    # Epsilon to ensure the plots are OK for low efficiency thresholds
    epsilon = 0.0001
    
    # Plot the signal efficiency vs energy
    fig = plt.figure(figsize=(32,18), facecolor="w")
        
    for threshold, efficiency in zip(thresholds, avg_efficiencies):
        
        # Values to be plotted at the end
        signal_efficiency = []
        energy_values = []
        
        # Value for the previous non-zero events
        prev_non_zero_efficiency = 0
    
        # Iterate over the energy intervals computing the efficiency
        for energy_lower, energy_upper in zip(energy_lb, energy_ub):
            values, bins, _ = plot_classifier_response(softmaxes, labels, energies,
                                                      {label_0:softmax_index_dict[label_0]},
                                                      {label_0:softmax_index_dict[label_0]},
                                                      energy_lower, energy_upper,
                                                      num_bins=num_bins, show_plot=False)
            if values is None or bins is None:
                print("""plot_utils.plot_signal_efficiency() : No events for the energy interval {0} to {1}.
                      Unable to plot.""".format(energy_lower, energy_upper))
                return None
                
            total_true_events = np.sum(values)
            num_true_events_selected = np.sum(values[bins[:len(bins)-1] > threshold-epsilon])
            
            curr_interval_efficiency = num_true_events_selected/total_true_events if total_true_events > 0 else prev_non_zero_efficiency

            if(curr_interval_efficiency != 0):
                prev_non_zero_efficiency = curr_interval_efficiency

            # Add two times once for the lower energy bound and once for the upper energy bound
            signal_efficiency.append(curr_interval_efficiency)
            signal_efficiency.append(curr_interval_efficiency)

            # Add the lower and upper energy bounds
            energy_values.append(energy_lower)
            energy_values.append(energy_upper)

            label_to_use = r"Average signal efficiency = {0}, Threshold = {1:0.3f}".format(efficiency, threshold)

        if(avg_efficiency_colors != None):
            plt.plot(energy_values, signal_efficiency, color=avg_efficiency_colors[threshold], linewidth=2.0,
                 marker=".", markersize=6.0, markerfacecolor=avg_efficiency_colors[threshold], label=label_to_use)
        else:
            plt.plot(energy_values, signal_efficiency, linewidth=2.0, marker=".", markersize=6.0, label=label_to_use)
            
        

    if(label_0 is not "e"):
             title = r"Signal Efficiency vs Energy for $\{0}$ events.".format(label_0)
    else:
             title = r"Signal Efficiency vs Energy for ${0}$ events.".format(label_0)
             
    plt.title(title, fontsize=20)
    plt.grid(True, color='grey')
             
    plt.xlim([min_energy, max_energy])
    plt.ylim([0, 1.05])
    plt.tick_params(axis="both", labelsize=20)
             
    plt.xlabel("Event Visible Energy (MeV)", fontsize=20)
    plt.ylabel("Signal Efficiency", fontsize=20)
    plt.legend(loc="upper left", prop={"size":20})
        
    if save_path is not None:
        plt.savefig(save_path)
    
    if show_plot:
        plt.show()
        
    plt.clf() # Clear the current figure
    plt.close() # Close the opened window
        
# TODO: something's broken here
# Plot background rejection for a given event
def plot_background_rejection(softmaxes, labels, energies, softmax_index_dict, label_0, label_1,
                              avg_efficiencies=[0.2, 0.5, 0.8], avg_efficiency_colors=None,
                              energy_interval=25, min_energy=100, max_energy=1000, num_bins=100,
                              show_plot=False, save_path=None):
    
    """
    plot_background_rejection(softmaxes, labels, energies, softmax_index_dict, event,
                              avg_efficiencies=[0.2, 0.5, 0.8], avg_efficiency_color=None,
                              energy_interval=25, min_energy=100, max_energy=1000, num_bins=100,
                              show_plot=False, save_path=None)
                           
    Purpose : Plot the background rejection vs energy for several thresholds
    
    Args: softmaxes             ... 2D array of softmaxes output, length = sample size, dimensions = n_samples, n_classes
          labels                ... 1D array of true labels
          energies              ... 1D array of visible event energies
          softmax_index_dict    ... Dictionary with the keys as event type (str) and values as the column indices 
                                    in the np softmaxes array
          label_0               ... Event type for which to plot the background rejection for
          label_1               ... Event type for which to plot the background rejection against
          avg_efficiencies      ... 1D array with the average efficiency values for which to plot the signal efficiency
                                    vs energy plot, default=[0.2, 0.5, 0.8]
          avg_efficiency_colors ... Average efficiencies color dictionary to use. The keys are the iterms in the
                                    avg_efficiencies list and values are the colors to be used.
          energy_interval       ... Energy interval to be used to calculate the response curve and calculating the signal                 
                                    efficiency, default=25
          min_energy            ... Minimum energy for the events to consider, default=0
          max_energy            ... Maximum energy for the events to consider, default=1000
          show_plot[optional]   ... Boolean to determine whether to show the plot, default=False
          save_path[optional]   ... Path to save the plot to, format='eps', default=None
    """
    
    # Assertions to check for valid inputs
    assert softmaxes is not None
    assert labels is not None
    assert energies is not None
    
    # Need high number of bins to avoid empty values
    assert num_bins >= 100
    assert label_0 in softmax_index_dict.keys()
    assert label_1 in softmax_index_dict.keys()
    
    # Calculate the threshold here according to the desired average efficiencies
    _, _, threshold_0, _, _, tpr_1, threshold_1, _ = plot_ROC_curve_one_vs_one(softmaxes, labels, 
                                                                               energies,
                                                                               softmax_index_dict,
                                                                               label_0,
                                                                               label_1,
                                                                               min_energy,
                                                                               max_energy,
                                                                               show_plot=False)
    
    thresholds = []
    threshold_index_dict = {}
    tolerance = 0.25
    
    # Get the index o
    for tpr_value in avg_efficiencies:
        
        index_list = []
        
        for i in range(len(tpr_1)):
            if(math.fabs(tpr_1[i]-tpr_value) < 0.001):
                index_list.append(i)
                
        if(len(index_list) == 0):
            lower_tpr, lower_index, upper_index, upper_tpr = 0.0, 0, 0, 1.0
            for i in range(len(tpr_1)):
                if(tpr_1[i] < tpr_value and tpr_1[i] > lower_tpr):
                    lower_index = i
                    lower_tpr = tpr_1[i]
                if(tpr_1[i] > tpr_value):
                    upper_index = i
                    upper_tpr = tpr_1[i]
                    break
            if(upper_tpr - lower_tpr > tolerance):
                print("""plot_utils.plot_background_rejection() : Unable to calculate threshold for average
                      efficiency = {0}""".format(tpr_value))
                return None
            else:
                thresholds.append(round((threshold_1[lower_index] + threshold_1[upper_index])/2, 2))
                
        else:
            index = index_list[math.floor(len(index_list)/2)]
            thresholds.append(round(threshold_1[index], 2))
    
    # Get the energy intervals to plot the signal efficiency against ( replace with max(energies) ) 
    energy_lb = [min_energy+(energy_interval*i) for i in range(math.ceil((max_energy-min_energy)/energy_interval))]
    energy_ub = [energy_low+energy_interval for energy_low in energy_lb]
    
    # Epsilon to ensure the plots are OK for low efficiency thresholds
    epsilon = 0.0001
    
    # Plot the background rejection vs energy
    fig = plt.figure(figsize=(32,18), facecolor="w")
    
    for threshold, efficiency in zip(thresholds, avg_efficiencies):
    
        # Initialize the dictionary to hold the background rejection values
        background_rejection_dict = {}
        for key in softmax_index_dict.keys():
            if(key != label_0):
                background_rejection_dict[key] = []
    
        energy_values = []
    
        # List of all the keys for background rejection
        background_rejection_keys = list(background_rejection_dict.keys())
    
        # Add an extra color to the color dict for total background rejection
        color_dict["total"] = "black"
    
        # Iterate over the energy intervals to compute the background rejection
        for key in background_rejection_dict.keys():

            # Value for the previous non-zero events
            prev_non_zero_rejection = 0.0

            # Initialize the dict to pass
            if( key == "total" ):
                pass_dict = softmax_index_dict.copy()
                del pass_dict[key]
            else:
                pass_dict = {key:softmax_index_dict[key]}

            for energy_lower, energy_upper in zip(energy_lb, energy_ub):

                values, bins, _ = plot_classifier_response(softmaxes, labels, energies, pass_dict,
                                                          {label_0:softmax_index_dict[label_0]},
                                                          energy_lower, energy_upper, 
                                                          num_bins=num_bins, show_plot=False)
                
                # Find the number of false events rejected
                if values is None or bins is None:
                    print("""plot_utils.plot_background_rejection() : No events for the energy interval {0} to {1}.
                          Unable to plot.""".format(energy_lower, energy_upper))
                    return None
                    
                # Find the number of false events rejected
                total_false_events = np.sum(values)
                num_false_events_rejected = np.sum(values[bins[:len(bins)-1] < threshold])
                
                curr_interval_rejection = num_false_events_rejected/total_false_events if total_false_events > 0 else 0

                if(curr_interval_rejection == 0):
                    curr_interval_rejection = prev_non_zero_rejection
                else:
                    prev_non_zero_rejection = curr_interval_rejection

                # Add two times once for the lower energy bound and once for the upper energy bound
                background_rejection_dict[key].append(curr_interval_rejection)
                background_rejection_dict[key].append(curr_interval_rejection)

                # If the key is the last key in the dict
                if( key == background_rejection_keys[len(background_rejection_keys)-1]):

                    # Add the lower and upper energy bounds
                    energy_values.append(energy_lower)
                    energy_values.append(energy_upper)
                    
        for key in background_rejection_keys:
            
            label_to_use = None
            if( key == "total" ):
                label_to_use = r"Average signal efficiency = {0}, Threshold = {1:0.3f}".format(efficiency, threshold)
            elif( key == "e" ):
                label_to_use = r"Average signal efficiency = {0}, Threshold = {1:0.3f}".format(efficiency, threshold)
            else:
                label_to_use = r"Average signal efficiency = {0}, Threshold = {1:0.3f}".format(efficiency, threshold)

            if(avg_efficiency_colors != None):
                plt.plot(energy_values, background_rejection_dict[key], color=avg_efficiency_colors[threshold], 
                         linewidth=2.0, marker=".", markersize=6.0, markerfacecolor=avg_efficiency_colors[threshold],
                         label=label_to_use)
            else:
                plt.plot(energy_values, background_rejection_dict[key], linewidth=2.0, marker=".", markersize=6.0,
                         label=label_to_use)
            
        
    # Delete the total key from the color dict
    del color_dict["total"]
             
    if label_0 is not "e" and key is not "e":
        title = r"$\{0}$ Background rejection vs Energy for selecting $\{1}$ events.".format(key, label_0)
    elif label_0 is "e":
        title = r"$\{0}$ Background rejection vs Energy for selecting ${1}$ events.".format(key, label_0)
    elif key is "e":
        title = r"${0}$ Background rejection vs Energy for selecting $\{1}$ events.".format(key, label_0)
             
    plt.title(title, fontsize=20)
    plt.grid(True, color='grey')
             
    plt.xlim([min_energy, max_energy])
    plt.ylim([0.0, 1.05])
    plt.tick_params(axis="both", labelsize=20)
             
    plt.xlabel("Event visible energy (MeV)", fontsize=20)
    plt.ylabel("Background rejection", fontsize=20)
    plt.legend(loc="upper left", prop={"size":20})
        
    if save_path is not None:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
        
    plt.clf() # Clear the plot frame
    plt.close() # Close the opened window if any
    
        
# Plot model performance over the training iterations
def plot_training(log_paths, model_names, model_color_dict, state_paths=[], downsample_interval=None, legend_loc=(0,0.8), show_plot=False, save_path=None):
    """
    plot_training_loss(training_directories=None, model_names=None, show_plot=False, save_path=None)
                           
    Purpose : Plot the training loss for various models for visual comparison
    
    Args: log_paths           ... List containing the absolute path to the .csv log files containing training info
                                  Type : str
          state_paths         ... List containing the absolute path to the .csv log files containing best state info
                                  Type : str
          model_names         ... List of the tring model name
                                  Type : str
          model_color_dict    ... Dictionary with the model_names as keys and
                                  the corresponding colors as values
          downsample_interval ... Downsample interval to smoothen the results,
                                  Type : int
          legend_loc          ... Location of where to put the legend on the plot
                                  Type : tuple
                                  Format : (x_pos, y_pos), 0 <= x_pos <= 1, 0 <= y_pos <= 1
          show_plot[optional] ... Boolean to determine whether to show the plot
                                  Type : Boolean
          save_path[optional] ... Absolute path to save the plot to
                                  Type : str
    """
    
    # Assertions
    assert log_paths is not None
    assert model_names is not None
    assert model_color_dict is not None
    assert len(log_paths) == len(model_names)
    assert len(model_names) == len(model_color_dict.keys())
    
    # Extract the values stored in the .csv log files
    loss_values = []
    epoch_values = []
    acc_values = []
    
    # Iterate over the list of log files provided
    for log_path in log_paths:
        if(os.path.exists(log_path)):
            log_df = pd.read_csv(log_path, usecols=["epoch", "loss", "accuracy"])
            
            # Downsample the epoch and training loss values w.r.t. the downsample interval
            curr_epoch_values = log_df["epoch"].values
            curr_loss_values  = log_df["loss"].values
            curr_acc_values  = log_df["accuracy"].values
            
            # Downsample using the downsample interval
            if downsample_interval == None:
                epoch_values.append(curr_epoch_values)
                loss_values.append(curr_loss_values)
                acc_values.append(curr_acc_values)
            else:
                curr_epoch_values_downsampled = []
                curr_loss_values_downsampled  = []
                curr_acc_values_downsampled  = []

                curr_epoch_list = []
                curr_loss_list = []
                curr_acc_list = []

                for i in range(1, len(curr_epoch_values)):

                    if(i%downsample_interval == 0):

                        # Downsample the values using the mean of the values for the current interval
                        curr_epoch_values_downsampled.append(sum(curr_epoch_list)/downsample_interval)
                        curr_loss_values_downsampled.append(sum(curr_loss_list)/downsample_interval)
                        curr_acc_values_downsampled.append(sum(curr_acc_list)/downsample_interval)

                        # Reset the list for the next interval
                        curr_loss_list = []
                        curr_epoch_list = []
                        curr_acc_list = []
                    else:
                        # Add the values in the interval to the list
                        curr_epoch_list.append(curr_epoch_values[i])
                        curr_loss_list.append(curr_loss_values[i]) 
                        curr_acc_list.append(curr_acc_values[i])

                epoch_values.append(curr_epoch_values_downsampled)
                loss_values.append(curr_loss_values_downsampled)
                acc_values.append(curr_acc_values_downsampled)
        else:
            print("Error. log path {0} does not exist".format(log_path))
            
    # Extract values stored in best state files
    state_log_epochs = []
    state_log_losses = []
    state_log_acc = []
    for state_path in state_paths:
        if(os.path.exists(state_path)):
            # No downsampling needed here
            state_df = pd.read_csv(state_path, usecols=["epoch", "loss", "accuracy"])
            state_log_epochs.append(state_df["epoch"].values)
            state_log_losses.append(state_df["loss"].values)
            state_log_acc.append(state_df["accuracy"].values)
        else:
            print("Error. state log path {0} does not exist".format(state_path))
            
    # Initialize the plot
    fig, ax1 = plt.subplots(figsize=(16,11))
    ax2= ax1.twinx()
    
    # Plot the values
    for i, model_name in enumerate(model_names):
        ax1.plot(epoch_values[i], loss_values[i], color=model_color_dict[model_name][0],
                 label= model_name + " loss")
        ax2.plot(epoch_values[i], acc_values[i], color=model_color_dict[model_name][1],
                 label= model_name + " accuracy")
        
        if len(state_log_epochs) > i:
            ax1.plot(state_log_epochs[i], state_log_losses[i], marker='o', markersize=5, linestyle='', color=model_color_dict[model_name][0],
                     label= model_name + " saved best states (loss)")
            xy = (round(state_log_epochs[i][-1],4), round(state_log_losses[i][-1],4))
            ax1.annotate('\t(Epoch: %s, Loss: %s)' % xy, xy=xy, textcoords='data', fontsize=18)
            ax2.plot(state_log_epochs[i], state_log_acc[i], marker='o', markersize=5, linestyle='', color=model_color_dict[model_name][1],
                     label= model_name + " saved best states (accuracy)")
            xy = (round(state_log_epochs[i][-1],4), round(state_log_acc[i][-1],4))
            ax2.annotate('\t(Epoch: %s, Accuracy: %s)' % xy, xy=xy, textcoords='data', fontsize=18)
        
    # Setup plot characteristics
    ax1.tick_params(axis="both", labelsize=20)
    ax2.tick_params(axis="both", labelsize=20)
    
    ax1.set_xlabel("Epoch", fontsize=20)
    ax1.set_ylabel("Loss", fontsize=20)
    ax1.set_ylim(bottom=0)
    ax2.set_ylabel("Accuracy", fontsize=20)
    ax2.set_ylim(bottom=0)
    
    plt.grid(True, color='grey')
    lgd = fig.legend(prop={"size":20}, bbox_to_anchor=legend_loc)
    fig.suptitle("Training vs Epochs", fontsize=25)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_extra_artists=(lgd))
    if show_plot:
        plt.show()
        
    plt.clf() # Clear the plot frame
    plt.close() # Close the opened window if any

def plot_learn_hist_smoothed(train_log, val_log, window=40, save_path=None, show_plot=False):

    train_log_csv = pd.read_csv(train_log)
    val_log_csv  = pd.read_csv(val_log)
    
    # Smooth training, plot validation as scatter
    train_epoch    = moving_average(np.array(train_log_csv.epoch),window)
    train_accuracy = moving_average(np.array(train_log_csv.accuracy),window)
    train_loss     = moving_average(np.array(train_log_csv.loss),window)

    fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    line11 = ax1.plot(train_epoch, train_loss, linewidth=2, label='Train loss', color='b', alpha=0.7)
    line12 = ax1.plot(val_log_csv.epoch, val_log_csv.loss, marker='o', markersize=3, linestyle='', label='Validation loss', color='blue')
    
    
    ax2 = ax1.twinx()
    line21 = ax2.plot(train_epoch, train_accuracy, linewidth=2, label='Train accuracy', color='r', alpha=0.7)
    line22 = ax2.plot(val_log_csv.epoch, val_log_csv.accuracy, marker='o', markersize=3, linestyle='', label='Validation accuracy', color='red')

    ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)
    
    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,1.05)
    

    # added these four lines
    lines  = line11 + line12 + line21 + line22
    labels = [l.get_label() for l in lines]
    leg    = ax2.legend(lines, labels, fontsize=16, loc=5, numpoints=1)
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')

    plt.grid(color='grey')
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show_plot:
        plt.show()
        
    plt.clf() # Clear the plot frame
    plt.close() # Close the opened window if any

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Compound plot for training, validation, and best saved state positions
def plot_train_learn_log(train_log, state_log=None, val_log=None, window=128, save_path=None, show_plot=False):
    train_log_csv = pd.read_csv(train_log)
    # Smooth training
    train_epoch    = moving_average(np.array(train_log_csv.epoch),window)
    train_accuracy = moving_average(np.array(train_log_csv.accuracy),window)
    train_loss     = moving_average(np.array(train_log_csv.loss),window)
    
    lines = []
    
    # Plot training loss
    fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    lines.append(ax1.plot(train_epoch, train_loss, linewidth=2, label='Train loss', color='b', alpha=0.5))
    # Plot training accuracy
    ax2 = ax1.twinx()
    lines.append(ax2.plot(train_epoch, train_accuracy, linewidth=2, label='Train accuracy', color='r', alpha=0.5))
    
    # Load validation if provided
    if val_log is not None and os.path.isfile(val_log):
        val_log_csv  = pd.read_csv(val_log)
        # Smooth validation
        val_epoch    = moving_average(np.array(val_log_csv.epoch), window)
        val_accuracy = moving_average(np.array(val_log_csv.accuracy), window)
        val_loss     = moving_average(np.array(val_log_csv.loss), window)
        
        # Plot validation loss
        lines.append(ax1.plot(val_epoch, val_loss, linewidth=2, label='Validation loss', color='b', alpha=1))
        # Plot validation accuracy
        lines.append(ax2.plot(val_epoch, val_accuracy, linewidth=2, label='Validation accuracy', color='r', alpha=1))
    else:
        print("Provided validation log", val_log, "cannot be located, skipping validation plot...")
        
    # Load best state information if provided
    if state_log is not None and os.path.isfile(state_log):
        state_log_csv = pd.read_csv(state_log)
        
        state_log_epoch    = np.array(state_log_csv.epoch)
        state_log_accuracy = np.array(state_log_csv.accuracy)
        state_log_loss     = np.array(state_log_csv.loss)
        
        # Plot best state scatter on loss
        lines.append(ax1.plot(state_log_epoch, state_log_loss, marker='o', markersize=3, linestyle='', label='Best saved states loss', color='b'))
        xy = (round(state_log_epoch[-1],4), round(state_log_loss[-1],4))
        ax1.annotate('\t(Epoch: %s, Loss: %s)' % xy, xy=xy, textcoords='data', fontsize=14)
        # Plot best state scatter on accuracy
        lines.append(ax2.plot(state_log_epoch, state_log_accuracy, marker='o', markersize=3, linestyle='', label='Best saved states accuracy', color='r'))
        xy = (round(state_log_epoch[-1],4), round(state_log_accuracy[-1],4))
        ax2.annotate('\t(Epoch: %s, Accuracy: %s)' % xy, xy=xy, textcoords='data', fontsize=14)
    else:
        print("Provided state log", state_log, "cannot be located, skipping best state plot...")

    # General axis formatting
    ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)
    
    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,1.05)
    
    # Add lines
    linesum = lines[0]
    for line in lines[1:]: linesum += line
    labels = [l.get_label() for l in linesum]
    leg    = ax2.legend(linesum, labels, fontsize=16, loc=5, numpoints=1)
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')

    plt.grid(color='grey')
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show_plot:
        plt.show()
        
    plt.clf() # Clear the plot frame
    plt.close() # Close the opened window if any