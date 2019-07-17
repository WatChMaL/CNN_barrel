"""
WELCOME TO WatChMaL, USER

START PROGRAM HERE

watchmal.py: Script to pass commandline arguments from user to neural net framework.

Author: Julian Ding
"""

import os

import training_utils.engine as net
import io_utils.arghandler as arghandler
import io_utils.ioconfig as ioconfig
import io_utils.modelhandler as modelhandler

USER_DIR = 'USER/'
TASKS = ['train', 'test', 'valid', 'plot']

# Global list of arguments to request from commandline
ARGS = [arghandler.Argument('model', list, list_dtype=str, flag='-m',
                            default=['resnet', 'resnet18'], help='Specify neural net architecture. Default is resnet18.'),
        arghandler.Argument('params', list, list_dtype=str, flag='-pms',
                            default=['num_input_channels=38', 'num_classes=3'], help='Specify args to pass to neural net constructor.'),
        arghandler.Argument('device', str, '-dev',
                            default='cpu', help='Enter cpu to use CPU resources or gpu to use GPU resources.'),
        arghandler.Argument('gpu_list', list, list_dtype=int, flag='-gpu',
                            help='List of available GPUs.'),
        arghandler.Argument('path', str, '-pat',
                            default=None, help='Path to training dataset.'),
        arghandler.Argument('subset', int, '-sub',
                            default=None, help='Number of data from training set to use.'),
        arghandler.Argument('shuffle', bool, '-shf',
                            default=True, help='Specify whether or not to shuffle training dataset. Default is True.'),
        arghandler.Argument('l2_lambda', float, '-l2l',
                            default=0, help='Specify lambda (weight) for L2 regularization.'),
        arghandler.Argument('val_split', float, '-vas',
                            default=0.1, help='Fraction of dataset used in validation.'),
        arghandler.Argument('test_split', float, '-tes',
                            default=0.1, help='Fraction of dataset used in testing. (Note: remaining fraction is used in training)'),
        arghandler.Argument('epochs', float, '-epo',
                            default=1.0, help='Number of training epochs to run.'),
        arghandler.Argument('batch_size_train', int, '-tnb',
                            default=16, help='Batch size for training.'),
        arghandler.Argument('batch_size_val', int, '-vlb',
                            default=1024, help='Batch size for validation. This also applies to early-stopping validation.'),
        arghandler.Argument('batch_size_test', int, '-tsb',
                            default=1024, help='Batch size for testing.'),
        arghandler.Argument('es_batches', int, '-esn',
                            default=8, help='Number of batches to run for early-stopping. If 0, early-stopping is turned off.'),
        arghandler.Argument('es_valid_interval', int, '-esi',
                            default=128, help='Run early-stopping validation every n training batches. Must be a positive integer.'),
        arghandler.Argument('num_workers', int, '-wkr',
                            default=2, help='Number of subprocesses to load data onto.'),
        arghandler.Argument('tasks',list, list_dtype=str, flag='-do',
                            default=TASKS, help='Specify list of tasks: "train" = run training; "test" = run testing; "valid" = run validation; "plot" = dump validation plots. Default behaviour runs all tasks.'),
        arghandler.Argument('worst', int, flag='-wst',
                            default=0, help='Specify the number of WORST-identified events to dump root file references to at the end of validation.'),
        arghandler.Argument('best', int, flag='-bst',
                            default=0, help='Specify the number of BEST-identified events to dump root file references to at the end of validation.'),
        arghandler.Argument('save_path', str, '-sap',
                            default='save_path', help='Specify path to save data to. Default is save_path.'),
        arghandler.Argument('data_description', str, '-dsc',
                            default='logs', help='Specify description for data/name for data subdirectory.'),
        arghandler.Argument('load', str, '-l',
                            default=None, help='Specify config file to load from. No action by default.'),
        arghandler.Argument('restore_state', str, '-ret',
                            default=None, help='Specify a state file to restore the neural net to the training state from a previous run. No action by default'),
        arghandler.Argument('cfg', str, '-s',
                            default=None, help='Specify name for destination config file. No action by default.')]

ATTR_DICT = {arg.name : ioconfig.ConfigAttr(arg.name, arg.dtype,
                                            list_dtype = arg.list_dtype if hasattr(arg, 'list_dtype') else None) for arg in ARGS}

if __name__ == '__main__':
    # Intro message :D
    print("""[HK-Canada] TRIUMF Neutrino Group: Water Cherenkov Machine Learning (WatChMaL)
\tCollaborators: Wojciech Fedorko, Julian Ding, Abhishek Kajal\n""")
    # Reflect available models
    print('CURRENT AVAILABLE ARCHITECTURES')
    modelhandler.print_models()
    config = arghandler.parse_args(ARGS)
    # Do not overwrite any attributes specified by commandline flags
    for ar in ARGS:
        if getattr(config, ar.name) != ar.default:
            ATTR_DICT[ar.name].overwrite = False
    # Create user directory if necessary
    if not os.path.isdir(USER_DIR):
        os.mkdir(USER_DIR)
        print("Created user directory", USER_DIR)
    # Load from file
    if config.load is not None:
        ioconfig.loadConfig(config, config.load, ATTR_DICT)
    # Check attributes for validity
    for task in config.tasks:
        assert(task in TASKS)
    # Save to file
    if config.cfg is not None:
        ioconfig.saveConfig(config, config.cfg)
    # Set save directory to under USER_DIR
    config.save_path = USER_DIR+config.save_path+('' if config.save_path.endswith('/') else '/')
    # Select requested model
    print('Selected architecture:', config.model)
    # Make sure the specified arguments can be passed to the model
    params = ioconfig.to_kwargs(config.params)
    modelhandler.check_params(config.model, params)
    constructor = modelhandler.select_model(config.model)
    model = constructor(**params)
    # Finally, construct the neural net
    nnet = net.Engine(model, config)
    # Do some work...
    if config.restore_state is not None:
        nnet.restore_state(config.restore_state)
    if 'train' in config.tasks:
        print("Number of epochs :", config.epochs)
        nnet.train(epochs=config.epochs, valid_interval=config.es_valid_interval, valid_batches=config.es_batches)
    if 'test' in config.tasks:
        nnet.test()
    if 'valid' in config.tasks:
        nnet.validate(plt_worst=config.worst, plt_best=config.best)
    if 'plot' in config.tasks:
        nnet.dump_plots()