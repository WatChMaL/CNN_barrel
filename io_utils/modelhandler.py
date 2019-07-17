"""
Module for automated model selection

Author: Julian Ding
"""

import importlib
import os
import sys
import inspect

MODELS_DIR = 'models'
models = importlib.import_module(MODELS_DIR)

# Helper function to find constructor name
# REQUIRES: The model name string when stripped to "___net" must have a
#           corresponding class in the module named "[Capitalized]Net"
#           (ex. resnet_dropout -> ResNet)
def intuit_constructor(name):
    if 'net' in name:
        constructor = name.split('net', 1)[0].capitalize() + 'Net'
        return constructor
    else:
        print('\tCannot intuit constructor from architecture name that is not of form "*net*".\n')
        return name

# Prints list of all models and constructors associated with each model
# Also returns a list of lists of parameters associated with all available models
def print_models():
    for name in models.__all__:
        print(name+':')
        # Suppress extraneous printing to console
        sys.stdout = open(os.devnull, 'w')
        curr_model = importlib.import_module(MODELS_DIR+'.'+name)
        sys.stdout = sys.__stdout__
        
        constructors = [x for x in dir(curr_model) if x.startswith(name)]
        for c in constructors:
            print('\t'+c)
        name = intuit_constructor(name)
        if hasattr(curr_model, name):
            arglist = inspect.getfullargspec(getattr(curr_model, name).__init__).args
            print('\tConstructor parameters:', arglist, '\n')
        else:
            print('\tNo general constructor (__init__) found.\n')
    
# Returns a function pointer corresponding to the constructor for the specified model
# REQUIRES: All constructors across all models must SHARE THE SAME PARAMETERS,
#           otherwise calling this function may break the program
def select_model(select_params):
    assert len(select_params) == 1 or len(select_params) == 2, "Invalid number of model parameters specified ("+str(len(select_params))+")"
    name = select_params[0]
    if len(select_params) == 1:
        constructor = intuit_constructor(name)
    else:
        constructor = select_params[1]
    # Make sure the specified model exists
    assert(name in models.__all__)
    model = importlib.import_module(MODELS_DIR+'.'+name)
    # Make sure the specified constructor exists
    assert constructor in dir(model), "Specified constructor "+constructor+" is not implemented in model "+name
    # Return specified constructor
    return getattr(model, constructor)

# Check if an argument list is valid for a specified model; stops program if not
# REQUIRES: argslist is a kwargs-interpretable dictionary and specified model string
#           is a valid module in the models package.
def check_params(model, argslist):
    name = model[0]
    base = intuit_constructor(name)
    constructor = base if len(model) < 2 else model[1]
    mod = importlib.import_module(MODELS_DIR+'.'+model[0])
    assert constructor in dir(mod), "Cannot find constructor named "+constructor+" in implementation of "+model[0]
    valid_args = inspect.getfullargspec(getattr(mod, base).__init__).args
    for arg in argslist.keys():
        assert arg in valid_args, "Argument "+str(arg)+" is not in the list of valid arguments for constructor "+constructor+" "+str(valid_args)
    print('Params OK:', argslist)