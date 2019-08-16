"""
Module for config file loading/saving functionality

Author: Julian Ding
"""

import os
import configparser

# Config file type
CFG_EXT = '.ini'

# Delimiters
DELIM = ' '
ARG_DELIM = '='
NAME_DELIM = '_'

# Default keys for datasets
KEYS_LIST = 'data_keys.ini'

# Class to encapsulate the necessary properties of a config object attribute
class ConfigAttr():
    def __init__(self, name, dtype, list_dtype=None, overwrite=True):
        self.name = name
        self.dtype = dtype
        self.list_dtype = list_dtype
        self.overwrite = overwrite
        
# Helper function to add an attribute to a config object
def add_attr(config, name, data_str, dtype, list_dtype=None):
    if data_str == 'None':
        setattr(config, name, None)
    elif dtype != list:
        setattr(config, name, dtype(data_str))
    elif list_dtype is not None:
        attrlist = [list_dtype(x) for x in data_str.split(DELIM)]
        setattr(config, name, attrlist)
    else:
        print('Load error encountered when parsing', data_str, 'as', dtype)

# Loads configuration from a specified file from within a specified directory
def loadConfig(config, pardir, inFile, attr_dict):
    if not inFile.endswith(CFG_EXT):
        inFile += CFG_EXT
    print('Requested load from:', inFile)
    print('Scanning', pardir, 'for configuration file...')
    # Get list of valid config files in current directory
    cFiles = [f for f in os.listdir(pardir) if (os.path.splitext(f)[1] == '.ini')]
    if len(cFiles) > 0:
        print('Config files found:', cFiles)
        if inFile in cFiles:
            parser = configparser.ConfigParser()
            parser.read(os.path.join(pardir, inFile))
            keys = parser.items('config')
            print('Loading from', inFile)
            for (item, data_str) in keys:
                if item in attr_dict:
                    curr = attr_dict[item]
                    if curr.overwrite:
                        # If the item is a requested config attribute, parse string
                        add_attr(config, curr.name, data_str, curr.dtype,
                                 list_dtype=curr.list_dtype)
            print(inFile, 'loaded!')
        else:
            print(inFile, 'not found, aborting.')
    else:
        print('No config files found, aborting.')

# Saves config object to configuration file
def saveConfig(config, outFile):
    if not outFile.endswith(CFG_EXT):
        outFile += CFG_EXT
    print('Saving config file as', outFile)
    conf = configparser.ConfigParser()
    conf.add_section('config')
    # Store all config attributes in ConfigParser
    for x in dir(config):
        if not x.startswith('_'):
            item = vars(config)[x]
            if type(item) == list:
                listStr = ''
                for t in item:
                    listStr += DELIM + str(t)
                item = listStr
            conf.set('config', str(x), str(item))
    # Do not overwrite existing config files
    while os.path.isfile(outFile):
        n_outFile = outFile.split('.')[0]
        name_split = n_outFile.split(NAME_DELIM)
        n_outFile = name_split[0]+NAME_DELIM+('0' if len(name_split) == 1 else str(int(name_split[1])+1))+CFG_EXT
        print('Config file with name', outFile, 'already exists. Saving to:', n_outFile)
        outFile = n_outFile
    with open(outFile, 'w+') as configFile:
        conf.write(configFile)
    
# Function to convert a list of strings into a kwargs-interpretable dict
def to_kwargs(arglist, delim=ARG_DELIM):
    args = [arg.split(delim) for arg in arglist]
    return {x[0] : parse_arg(x[1]) for x in args}

# Converts an argument string to the correct dtype
def parse_arg(arg):
    # If argument string is enclosed with quotes, parse as string
    if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
        return arg[1:-1]
    # Otherwise, check if arg is an integer
    try: return int(arg)
    except:
        ValueError
        # If not, check if arg is a float
        try: return float(arg)
        except:
            ValueError
            # If not, return string
            return arg
        
# Loads data keys from KEYS_LIST file into a dictionary
def get_keys_dict():
    parser = configparser.ConfigParser()
    parser.read(KEYS_LIST)
    keys = {name:key for name, key in parser.items('data_keys')}
    return keys