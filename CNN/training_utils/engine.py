'''
Author: Wojciech Fedorko
Collaborators: Julian Ding, Abhishek Kajal
'''

import copy # Currently unused
import re # Currently unused

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable # Currently unused
from torch.utils.data import DataLoader

import numpy as np # Currently unused
import time

from statistics import mean # Currently unused

import shutil # Currently unused
import os

import sklearn # Currently unused
from sklearn.metrics import roc_curve # Currently unused


from iotools.data_handling import WCH5Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class Engine:
    """The training engine 
    
    Performs training and evaluation
    """

    def __init__(self, model, config):
        self.model = model

        if config.gpu:
            print("requesting gpu ")
            print("gpu list: ")
            print(config.gpu_list)
            self.devids = ["cuda:{0}".format(x) for x in config.gpu_list]

            print("main gpu: "+self.devids[0])
            if torch.cuda.is_available():
                self.device = torch.device(self.devids[0])
                if len(self.devids) > 1:
                    print("using DataParallel on these devices: {}".format(self.devids))
                    self.model = nn.DataParallel(self.model, device_ids=config.gpu_list, dim=0)

                print("cuda is available")
            else:
                self.device=torch.device("cpu")
                print("cuda is not available")
        else:
            print("will not use gpu")
            self.device=torch.device("cpu")

        print(self.device)

        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(),eps=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        #placeholders for data and labels
        self.data=None
        self.labels=None
        self.iteration=None

        # NOTE: The functionality of this block is coupled to the implementation of WCH5Dataset in the iotools module
        self.dset=WCH5Dataset(config.path, config.val_split, config.test_split)

        self.train_iter=DataLoader(self.dset,
                                   batch_size=config.batch_size_train,
                                   shuffle=False,
                                   sampler=SubsetRandomSampler(self.dset.train_indices))
        
        self.val_iter=DataLoader(self.dset,
                                 batch_size=config.batch_size_val,
                                 shuffle=False,
                                 sampler=SubsetRandomSampler(self.dset.val_indices))
        
        self.test_iter=DataLoader(self.dset,
                                  batch_size=config.batch_size_test,
                                  shuffle=False,
                                  sampler=SubsetRandomSampler(self.dset.test_indices))

        

        self.dirpath=config.save_path
        
        self.data_description=config.data_description


        
        try:
            os.stat(self.dirpath)
        except:
            print("making a directory for model data: {}".format(self.dirpath))
            os.mkdir(self.dirpath)

        #add the path for the data type to the dirpath
        self.start_time_str = time.strftime("%Y%m%d_%H%M%S")
        self.dirpath=self.dirpath+'/'+self.data_description + "/" + self.start_time_str

        try:
            os.stat(self.dirpath)
        except:
            print("making a directory for model data for data prepared as: {}".format(self.data_description))
            os.makedirs(self.dirpath,exist_ok=True)

        self.config=config


    def forward(self,train=True):
        """
        Args: self should have attributes, model, criterion, softmax, data, label
        Returns: a dictionary of predicted labels, softmax, loss, and accuracy
        """
        with torch.set_grad_enabled(train):
            # Prediction
            #print("this is the data size before permuting: {}".format(data.size()))
            self.data = self.data.permute(0,3,1,2)
            #print("this is the data size after permuting: {}".format(data.size()))
            prediction = self.model(self.data)
            # Training
            loss,acc=-1,-1 # NOTE: What is acc supposed to do? It's never used....
            
            loss = self.criterion(prediction,self.label)
            self.loss = loss
            
            softmax    = self.softmax(prediction).cpu().detach().numpy()
            prediction = torch.argmax(prediction,dim=-1)
            accuracy   = (prediction == self.label).sum().item() / float(prediction.nelement())        
            prediction = prediction.cpu().detach().numpy()
        
        return {'prediction' : prediction,
                'softmax'    : softmax,
                'loss'       : loss.cpu().detach().item(),
                'accuracy'   : accuracy}

    def backward(self):
        self.opt.zero_grad()  # Reset gradients accumulation
        self.loss.backward()
        self.opt.step()
        
    # ========================================================================
    def train(self, epochs=3.0, report_interval=10, valid_interval=100):
        # CODE BELOW COPY-PASTED FROM [HKML CNN Image Classification.ipynb]
        # (variable names changed to match new Engine architecture. Added comments and minor debugging)
        
        # Prepare attributes for data logging
        from notebook_utils import progress_bar, CSVData
        self.train_log, self.test_log = CSVData(self.dirpath+'/log_train.csv'), CSVData(self.dirpath+'/log_test.csv')
        # Set neural net to training mode
        self.model.train()
        # Initialize epoch counter
        epoch = 0.
        # Initialize iteration counter
        iteration = 0
        # Training loop
        while (int(epoch+0.5) < epochs):
            from IPython.display import display
            print('Epoch',int(epoch+0.5),'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            # Create a progress bar for this epoch
            progress = display(progress_bar(0,len(self.train_iter)),display_id=True)
            # Loop over data samples and into the network forward function
            for i, data in enumerate(self.train_iter):
                # Data and label
                self.data, self.label = data[0:2]
                # Call forward: make a prediction & measure the average error
                res = self.forward(True)
                # Call backward: backpropagate error and update weights
                self.backward()
                # Epoch update
                epoch += 1./len(self.train_iter)
                iteration += 1
                
                #
                # Log/Report
                #
                # Record the current performance on train set
                self.train_log.record(['iteration','epoch','accuracy','loss'],[iteration,epoch,res['accuracy'],res['loss']])
                self.train_log.write()
                # once in a while, report
                if i==0 or (i+1)%report_interval == 0:
                    message = '... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f' % (iteration,epoch,res['loss'],res['accuracy'])
                    progress.update(progress_bar((i+1),len(self.train_iter),message))
                # more rarely, run validation
                if (i+1)%valid_interval == 0:
                    with torch.no_grad():
                        self.model.eval()
                        test_data = next(iter(self.test_iter))
                        self.data, self.label = test_data[0:2]
                        res = self.forward(False)
                        self.test_log.record(['iteration','epoch','accuracy','loss'],[iteration,epoch,res['accuracy'],res['loss']])
                        self.test_log.write()
                    self.model.train()
                if epoch >= epochs:
                    break
            message = '... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f' % (iteration,epoch,res['loss'],res['accuracy'])
            progress.update(progress_bar((i+1),len(self.train_iter),message))
        
        self.test_log.close()
        self.train_log.close()
    
    # ========================================================================

    def save_state(self, prefix='./snapshot'):
        # Output file name
        #filename = '%s-%d.ckpt' % (prefix, self.iteration)
        filename = '%s.ckpt' % (prefix)
    
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.opt.state_dict(),
            'state_dict': self.model.state_dict()
        }, filename)
        return filename

    def restore_state(self,weight_file):
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f)
            # load network weights
            self.net.load_state_dict(checkpoint['state_dict'], strict=False)
            # if optim is provided, load the state of the optim
            if self.opt is not None:
                self.opt.load_state_dict(checkpoint['optimizer'])
            # load iteration count
            self.iteration = checkpoint['global_step']
    
        

        
