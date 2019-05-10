"""
START PROGRAM HERE

Script to pass commandline arguments from user to neural net framework.

Author: Julian Ding
"""

# Make sure all custom modules can be seen by the compiler
import os
import sys

par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if par_dir not in sys.path:
    sys.path.append(par_dir)
    
# Let's begin...
import training_utils.engine as net
import models.resnet as resnet
import argparse
import configparser

# Global dictionary of of config arguments (key : dtype)
ARGS = {'device' : 'str',
        'gpu' : 'bool',
        'gpu_list' : 'list str',
        'path' : 'str',
        'val_split' : 'float',
        'test_split' : 'float',
        'batch_size_train' : 'int',
        'batch_size_val' : 'int',
        'batch_size_test': 'int',
        'save_path' : 'str',
        'data_description' : 'str',
        'load' : 'str',
        'cfg' : 'str'}

def loadConfig(config):
    file = config.load
    if file == None:
        return config
    print('Requested load from:', file)
    print('Scanning', os.getcwd(), 'for configuration files...')
    # Get list of valid config files in current directory
    cFiles = [f for f in os.listdir(os.getcwd()) if (os.path.splitext(f)[1] == '.ini')]
    if len(cFiles) > 0:
        print('Config files found:', cFiles)
        if file in cFiles:
            parser = configparser.ConfigParser()
            parser.read(file)
            # Make sure the file has all the requisite config options
            assert(set(parser.items('config')).issubset(set([k for k in ARGS])))            
            print('Loading from', file)
            for item in parser.items('config'):
                argtype = ARGS[item].split()[0]
                option = parser.get('config', item)
                if argtype == 'str':
                    setattr(config, item, option)
                elif argtype == 'int':
                    setattr(config, item, int(option))
                elif argtype == 'float':
                    setattr(config, item, float(option))
                elif argtype == 'bool':
                    setattr(config, item, bool(option))
                elif argtype == 'list':
                    listtype = ARGS[item].split()[1]
                    # Note: str is the only list type at the moment
                    if listtype == 'str':
                        setattr(config, item, parser.get('config', item).split())
                else:
                    print(option, 'is not a valid config option, ignoring...')
        else:
            print(file, 'not found, aborting.')
            return config
    else:
        print('No config files found.')
        return config
    
def saveConfig(config):
    outFile = config.cfg
    print('Saving config file as', outFile)
    conf = configparser.ConfigParser()
    conf.add_section('config')
    # Store all config attributes in ConfigParser
    for x in dir(config):
        if not (x.startswith('_') or x.startswith('__')):
            conf.set('config', str(x), str(vars(config)[x]))
    with open(outFile, 'w+') as configFile:
        conf.write(configFile)
    print('Config file saved in', os.getcwd())

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-device', dest='device', default='cpu',
                        required=False, help='Enter cpu to use CPU resources or gpu to use GPU resources.')
    parser.add_argument('-gpu', nargs='+', dest='gpu_list', default=None,
                        required=False, help='List of available GPUs')
    parser.add_argument('-path', dest='path', default='',
                        required=False, help='Path to training dataset.')
    parser.add_argument('-vs', dest='val_split',
                        required=False, help='Fraction of dataset used in validation. Note: requires vs + ts < 1')
    parser.add_argument('-ts', dest='test_split', default=0.33,
                        required=False, help='Fraction of dataset used in testing. Note: requires vs + ts < 1')
    parser.add_argument('-tnb', dest='batch_size_train', default=20,
                        required=False, help='Batch size for training.')
    parser.add_argument('-vlb', dest='batch_size_val', default=1000,
                        required=False, help='Batch size for validating.')
    parser.add_argument('-tsb', dest='batch_size_test', default=1000,
                        required=False, help='Batch size for testing.')
    parser.add_argument('-save', dest='save_path', default='save_path',
                        required=False, help='Specify path to save data to. Default is save_path')
    parser.add_argument('-desc', dest='data_description', default='DATA DESCRIPTION',
                        required=False, help='Specify description for data.')
    parser.add_argument('-l', dest='load', default=None,
                        required=False, help='Specify config file to load from. No action by default.')
    parser.add_argument('-s', dest='cfg', default=None,
                        required=False, help='Specify name for destination config file. No action by default.')
    
    config = parser.parse_args()
    config.load += '.ini'
    config.cfg += '.ini'
    if config.device != 'GPU' or config.gpu_list is None:
        config.gpu = False
    else:
        config.gpu = True
        
    return config

if __name__ == '__main__':
    print('[HK-Canada] TRIUMF Neutrino Group: Deep learning initiative')
    print('Collaborators: Wojciech Fedorko, Julian Ding, Abhishek Kajal\n')
    config = main()
    loadConfig(config)
    saveConfig(config)
    model = resnet.resnet18()
    nnet = net.Engine(model, config)
    nnet.train()
    