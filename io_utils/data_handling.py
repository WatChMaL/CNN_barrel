"""
Dataset storing image-like data from Water Cherenkov detector memory-maps the
detector data from hdf5 file

Collaborators: Wojciech Fedorko, Julian Ding

Usage/implementation notes:
    - The input h5 dataset must contain data categories corresponding to the
    - contents in the data_keys.ini configuration file in the program root directory
    - Assuming the same h5 dataset, val_split, test_split, and seed, the pseudorandom
      index shuffler will always produce the same training, validation, and testing subsets
      (the pseudorandom number generator is deterministic)
    - The detector data must be uncompressed and unchunked
    - Labels are loaded into memory outright
"""

from torch.utils.data import Dataset
import h5py
from io_utils.ioconfig import get_keys_dict

import numpy as np

class WCH5Dataset(Dataset):

    def __init__(self, path, val_split, test_split, shuffle=True, transform=None, reduced_dataset_size=None, seed=42):
        
        assert val_split+test_split <= 1, "val_split and test_split cannot sum to larger than 1, aborting."

        # Open HDF5 dataset for loading
        f=h5py.File(path,'r')
        # Get data key strings
        keys = get_keys_dict()
        
        # Data and labels are essential
        hdf5_event_data = f[keys['data']]
        event_data_shape = hdf5_event_data.shape
        event_data_offset = hdf5_event_data.id.get_offset()
        event_data_dtype = hdf5_event_data.dtype
        
        hdf5_labels=f[keys['labels']]
        labels_shape = hdf5_labels.shape
        labels_offset = hdf5_labels.id.get_offset()
        labels_dtype = hdf5_labels.dtype
        
        assert hdf5_event_data.shape[0] == hdf5_labels.shape[0], "Number of labels does not match number of events, aborting."
        
        # Creates a memory map - i.e. events are not loaded in memory here, only on get_item
        self.event_data = np.memmap(path, mode='r', shape=event_data_shape, offset=event_data_offset, dtype=event_data_dtype)
        
        # Everything below is loaded outright onto memory
        self.labels = np.array(hdf5_labels)
        
        # Energies and positions are optional
        try:
            hdf5_energies=f[keys['energies']]
            energies_shape = hdf5_energies.shape
            energies_offset = hdf5_energies.id.get_offset()
            energies_dtype = hdf5_energies.dtype
            self.energies = np.array(hdf5_energies)
            self.has_energies = True
        except KeyError:
            print("Warning: No energy labels detected in the dataset.")
            self.has_energies = False
            
        try:
            hdf5_positions=f[keys['positions']]
            self.positions = np.array(hdf5_positions)
            self.has_positions = True
        except KeyError:
            print("Warning: No position labels detected in the dataset.")
            self.has_positions = False
        
        # Root file paths and event ids are required for root traceback
        try:
            hdf5_PATHS=f[keys['root_paths']]
            hdf5_IDX=f['root_idx']
            self.PATHS = np.array(hdf5_PATHS)
            self.IDX = np.array(hdf5_IDX)
            self.has_traceback = True
        except KeyError:
            print("Warning: Root file paths and/or event_ids missing, cannot perform traceback on events.")
            self.has_traceback = False

        self.transform=transform
        self.reduced_size = reduced_dataset_size

        #the section below handles the subset
        #(for reduced dataset training tests)
        #as well as shuffling and train/test/validation splits
        
        #save prng state
        rstate=np.random.get_state()
        if seed is not None:
            np.random.seed(seed)

        indices = np.arange(len(self))

        """
            Commenting for energy segmentation - Maybe not needed
        """
        if self.reduced_size is not None:
            assert len(indices)>=self.reduced_size
            indices = np.random.choice(self.labels.shape[0], reduced_dataset_size)

        #shuffle index array
        if shuffle:
            np.random.shuffle(indices)
        
        #restore the prng state
        if seed is not None:
            np.random.set_state(rstate)
            
        n_val = int(len(indices) * val_split)
        n_test = int(len(indices) * test_split)
        self.test_indices = indices[:n_test]
        self.val_indices = indices[n_test:n_test+n_val]
        self.train_indices = indices[n_test+n_val:]
        print(len(self.train_indices), "examples in training set.")
        print(len(self.val_indices), "examples in validation set.")
        print(len(self.test_indices), "examples in testing set.")

    def __getitem__(self,index):
        if self.transform is None:
            out = [np.array(self.event_data[index,:]), self.labels[index]]
            if self.has_energies:
                out.append(self.energies[index])
            if self.has_positions:
                out.append(self.positions[index])
            if self.has_traceback:
                out.extend([self.PATHS[index], self.IDX[index]])
            return out
        else:
            raise NotImplementedError

    def __len__(self):
        if self.reduced_size is None:
            return self.labels.shape[0]
        else:
            return self.reduced_size