# Water Cherenkov Machine Learning (WatChMaL) - Convolutional Neural Network (CNN)
Codebase for training and performance evaluation of CNNs using simulated neutrino weak interaction event data formatted as an image.

## User Guide
To start the program, download the repository and navigate to the parent folder, `CNN/`, then enter on terminal/cmd

```python3 watchmal.py <flags and arguments>```

There is an extensive list of flags which can be used to tune the training engine, detailed below. Every flag has valid default behaviour and thus none of the flags need to be specified to run the program.
### Setup
- `-h` prints out the help dialogue for all flags onto the terminal window. There is no config option for this flag.
- `-m <name> <constructor>` specifies an architecture to train on. Make sure the selected architecture exists in `models/`. A list of available architectures is printed on the terminal for convenience. The config option for this flag is `model`.
- `-pms <space-delimited list of named arguments>` specifies a list of arguments to pass to the CNN constructor. Make sure the arguments are valid for the selected constructor. A list of arguments taken by each constructor is printed on the terminal for convenience. The config option for this flag is `params`.
- `-dev <cpu/gpu>` sets the engine to offload work to the CPU or GPU. If GPU is selected, you must also specify a list of GPUs. The config option for this flag is `device`.
- `-gpu <space-delimited list of gpus (ints)>` gives the engine a list of GPUs to train on. If no GPUs are given, the training engine defaults to running on the CPU. The config option for this flag is `gpu_list`.
- `-do <train test val plot>` instructs the engine to run training, testing, validation, and plot visualization tasks. The engine can run any subset of these tasks and runs them all by default. The config option for this flag is `tasks`. Note that if `plot` is the only specified task, the engine will use the `save_path` parameter as the directory where it will look for data to plot. In this case, if `save_path` does not exist or is empty, the program will exit.
- `-wst <integer>` instructs the engine to dump a list of root file paths and indices identifying the *n* worst-identified events _of each class_ in the input dataset during validation. This dumps to a plain-text file in the `save_path` directory. By default this is set to `0`. The config option for this flag is `worst`.
- `-bst <integer>` instructs the engine to dump a list of root file paths and indices identifying the *n* best-identified events _of each class_ in the input dataset during validation. This dumps to the same directory as `worst`. The config option for this flag is `best`.

### Data Handling and Training Behaviour
Note: the keys of various elements of the input HDF5 dataset are set in the `data_keys.ini` configuration file in the `CNN` root directory. If keys are changed in the file conversion pathway for whatever reason, this configuration file must be updated accordingly for data loading to continue to work properly.

See the wiki page on ROOT file conversion for the conversion pathway from ROOT to .npz to HDF5.
- `-pat <path to data>` specifies the path to the labeled dataset which the engine will train, test, and validate on. HDF5 is the only supported data format at the moment. The config option for this flag is `path`. Note that if this flag is not specified, only the `plot` task will be able to execute.
- `-sub <number of events>` specifies a subset of the dataset located at `path` to use, which can be useful for making faster training runs. By default, all of the data is used. The config option for this flag is `subset`.
- `-shf <True/False>` specifies whether or not to shuffle the contents of the input dataset. By default this is set to `True`. The config option for this flag is `shuffle`.
- `-vas <decimal between 0 and 1>` specifies the fraction of the dataset to use for validation. By default this is set to `0.1`. The config option for this flag is `val_split`.
- `-tes <decimal between 0 and 1>` specifies the fraction of the dataset to use for testing. By default this is set to `0.1`. The config option for this flag is `test_split`.
- There is no option to specify the fraction of the dataset to use for training. This fraction is the remainder of the dataset that is outside the validation and test splits (i.e. `train_split = 1 - val_split - test_split`).
- `-epo <decimal>` specifies the number of epochs to train the data over. This number does not have to be a whole number. By default this is set to `1.0`. The config option for this flag is `epochs`.
- `-tnb <integer>` specifies the batch size during training. By default this is set to `16`. The config option for this flag is `batch_size_train`.
- `-vlb <integer>` specifies the batch size during validation (this includes early-stopping validation). By default this is set to `1024`. The config option for this flag is `batch_size_val`.
- `-tsb <integer>` specifies the batch size during testing. By default this is set to `1024`. The config option for this flag is `batch_size_test`.
- Note: the batch size should never exceed the dataset size.
- `-esn <integer>` specifies the number of batches to validate over during early-stopping validation. If this flag is set to `0`, early-stopping is turned off. By default this is set to `8`. The config option for this flag is `es_batches`.
- `-esi <integer>` defines the early-stopping validation frequency by specifying the number of training batches to iterate over before running early-stopping validation. If this flag is set to `0`, early-stopping is turned off. By default this is set to `128`. The config option for this flag is `es_valid_interval`.
- `-sap <path>` specifies the directory into which to save the training engine output data. This directory will be located inside `USER/` and has a default name of `save_path`. The config option for this flag is `save_path`. This directory is used as the target directory within which to search for plotting data.
- `-dsc <description>` specifies a subdirectory under `save_path` to save data from a particular run. By default this is set to `logs`. The config option for this flag is `data_description`.
- `-ret <state file>` specifies the path to a state file from which to restore the weights in the neural net. By default a state file is not loaded. The config option for this flag is `restore_state`.

### Config File Management
- `-l <config file name>` specifies a config file to load settings from. By default no config file is loaded and settings are interpreted from the specified flags. If this flag is specified but other flags conflict with the settings in the conflict file, the flags given on the commandline will override the respective settings in the config file. The config option for this flag is `load`.
- `-s <config file name>` specifies the name of a config file to save settings to (overwrite enabled). By default no config file is saved. The config option for this flag is `cfg`.

Note that you can manually write a config file and load it with the `-l` flag as an alternative to using commandline flags. The syntax for the config file is
```
[config]
option1 = string1
option2 = string2
...
```
(where `option` corresponds to the config options for the flags listed above and `string` corresponds to the desired commandline argument input.)

By default the config file extension is `.ini`. This can only be changed in the source code.
