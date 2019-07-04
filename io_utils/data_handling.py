"""
Dataset storing image-like data from Water Cherenkov detector memory-maps the
detector data from hdf5 file

Author: Wojciech Fedorko

Usage/implementation notes:
    - The input h5 dataset must contain data categories titled as follows:
        "event_data": mPMT charge/timing data in the shape
        (nevents, num_mPMTs_x, num_mPMTs_y, 19 charge channels + 19 timing channels = 38).
        "labels": truth labels of the particle class that generated each event, shape
        (nevents, 1). Labels are {gamma: 0, electron: 1, muon: 2}
        "positions" (currently unused in training engine)
        "energies" (currently unused in training engine)
    - The detector data must be uncompressed and unchunked
    - Labels are loaded into memory outright
"""

from torch.utils.data import Dataset
import h5py

import numpy as np

class WCH5Dataset(Dataset):

    def __init__(self, path, val_split, test_split, shuffle=True, transform=None, reduced_dataset_size=None, seed=42):
        
        assert val_split+test_split <= 1, "val_split and test_split cannot sum to larger than 1, aborting."

        f=h5py.File(path,'r')
        hdf5_event_data = f["event_data"]
        hdf5_labels=f["labels"]
        hdf5_energies=f["energies"]
        hdf5_positions=f["positions"]
        
        hdf5_PATHS=f["PATHS"]
        hdf5_IDX=f["IDX"]

        assert hdf5_event_data.shape[0] == hdf5_labels.shape[0]

        event_data_shape = hdf5_event_data.shape
        event_data_offset = hdf5_event_data.id.get_offset()
        event_data_dtype = hdf5_event_data.dtype
        
        labels_shape = hdf5_labels.shape
        labels_offset = hdf5_labels.id.get_offset()
        labels_dtype = hdf5_labels.dtype
        
        energies_shape = hdf5_energies.shape
        energies_offset = hdf5_energies.id.get_offset()
        energies_dtype = hdf5_energies.dtype

        #this creates a memory map - i.e. events are not loaded in memory here
        #only on get_item
        self.event_data = np.memmap(path, mode='r', shape=event_data_shape, offset=event_data_offset, dtype=event_data_dtype)
        

        #this will fit easily in memory even for huge datasets
        self.labels = np.array(hdf5_labels)
        
        # This will also fit easily in memory
        self.energies = np.array(hdf5_energies)
        self.PATHS = np.array(hdf5_PATHS)
        self.IDX = np.array(hdf5_IDX)

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
            return np.array(self.event_data[index,:]),  self.labels[index], self.energies[index], self.PATHS[index], self.IDX[index]
        else:
            raise NotImplementedError

    def __len__(self):
        if self.reduced_size is None:
            return self.labels.shape[0]
        else:
            return self.reduced_size