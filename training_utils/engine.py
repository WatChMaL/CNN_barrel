'''
Engine class implementation for CNN training engine.
Includes methods to train and validate performance, and save and restore state.

Collaborators: Wojciech Fedorko, Julian Ding, Abhishek Kajal

Notes:
    - DataLoader initializations are coupled to functionality in WCH5Dataset class
      implementation (found in data_handling module)
    - Visualization dumping in training and validation are coupled to implementation
      of the result_visualizer module
'''
import collections
import sys

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import os
import time
import numpy as np

from io_utils.data_handling import WCH5Dataset
from plot_utils.notebook_utils import CSVData
import plot_utils.result_visualizer as rv

from training_utils.doublepriorityqueue import DoublePriority

# ============================== CONSTANTS ====================================

# Directory containing saved states
STATE_DIR = 'saved_states/'
# Directory containing text files pointing to extreme events
EXTR_DIR = 'extreme_events/'
LIST_LO = 'list_lo.txt'
LIST_HI = 'list_hi.txt'
# Name of file containing saved validation data
VAL_STATE = 'val_state.npz'
# Names of training and validation logs
TRAIN_LOG = 'log_train.csv'
VAL_LOG = 'val_test.csv'
BEST_LOG = 'best_states.csv'
# Names and labels corresponding to particle classes
GAMMA, ELECTRON, MUON = 0, 1, 2
EVENT_CLASS = {GAMMA : 'gamma', ELECTRON : 'electron', MUON : 'muon'}
# Flag to distinguish best-so-far save state
BEST_FLAG = 'BEST'
LATEST_FLAG = 'LATEST'
# Loss threshold to define when a model is significantly better than a previous model
LOSS_THRESHOLD = 1e-3

# =============================================================================

class Engine:
    """The training engine 
    
    Performs training and evaluation
    """

    def __init__(self, model, config):
        self.model = model
        if (config.device == 'gpu') and config.gpu_list is not None:
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

        self.optimizer = optim.Adam(self.model.parameters(),eps=1e-3,weight_decay=config.l2_lambda)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        #placeholders for data and labels
        self.data=None
        self.labels=None
        self.iteration=None
        
        self.dirpath=config.save_path
        self.data_description=config.data_description

        # NOTE: The functionality of this block is coupled to the implementation of WCH5Dataset in the iotools module
        if config.path is not None:
            self.dset=WCH5Dataset(config.path,
                                  config.val_split,
                                  config.test_split,
                                  shuffle=config.shuffle,
                                  reduced_dataset_size=config.subset)
    
            self.train_iter=DataLoader(self.dset,
                                       batch_size=config.batch_size_train,
                                       shuffle=False,
                                       sampler=SubsetRandomSampler(self.dset.train_indices),
                                       num_workers=config.num_workers)
            
            self.val_iter=DataLoader(self.dset,
                                     batch_size=config.batch_size_val,
                                     shuffle=False,
                                     sampler=SubsetRandomSampler(self.dset.val_indices),
                                     num_workers=config.num_workers)
            
            self.test_iter=DataLoader(self.dset,
                                      batch_size=config.batch_size_test,
                                      shuffle=False,
                                      sampler=SubsetRandomSampler(self.dset.test_indices),
                                      num_workers=config.num_workers)
            
            try:
                os.stat(self.dirpath)
            except:
                print("making a directory for model data: {}".format(self.dirpath))
                os.mkdir(self.dirpath)
    
            #add the path for the data type to the dirpath
            self.start_time_str = time.strftime("%Y%m%d_%H%M%S")
            self.dirpath=os.path.join(self.dirpath, self.data_description, self.start_time_str)
    
            try:
                os.stat(self.dirpath)
            except:
                print("making a directory for model data for data prepared as: {}".format(self.data_description))
                os.makedirs(self.dirpath,exist_ok=True)
                
            self.state_dir = os.path.join(config.save_path, STATE_DIR)
            
        else:
            print("Warning: No training dataset supplied, can only run visualization tasks.")
            self.dset = None
            dirpath = os.path.join(self.dirpath, self.data_description)
            assert os.path.isdir(dirpath) and len(os.listdir(dirpath)) > 0, "No dataset provided and specified save directory "+dirpath+" does not exist(or is empty), aborting."
            runs = sorted(os.listdir(dirpath))
            self.dirpath = os.path.join(dirpath, runs[-1])
            print("Set target directory to", self.dirpath)

        self.config=config


    def forward(self,train=True):
        """
        Args: self should have attributes, model, criterion, softmax, data, label
        Returns: a dictionary of predicted labels, softmax, loss, and accuracy
        """
        with torch.set_grad_enabled(train):
            #print("this is the data size before permuting: {}".format(data.size()))
            self.data = self.data.float().permute(0,3,1,2)
            #print("this is the data size after permuting: {}".format(data.size()))
            # Move the data and labels on the selected device
            self.data = self.data.to(self.device)
            self.label = self.label.to(self.device)
            #Prediction
            prediction = self.model(self.data)
            # Training
            loss = -1
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
        self.optimizer.zero_grad()  # Reset gradients accumulation
        self.loss.backward()
        self.optimizer.step()
        
    def train(self, epochs=1.0, report_interval=10, valid_interval=100, valid_batches=4, save_interval=1000, save_all=False):
        
        if self.dset is None or len(self.dset.train_indices) == 0:
            print("No examples in training set, skipping training...")
            return
        
        # CODE BELOW COPY-PASTED FROM [HKML CNN Image Classification.ipynb]
        # (variable names changed to match new Engine architecture. Added comments and minor debugging)
        
        # Keep track of the validation accuracy
        best_val_loss = 1e10
        continue_train = True
        run_es = valid_batches > 0 and valid_interval > 0
        if run_es:
            print("Early-stopping is ACTIVE with validation steps of", valid_batches, "batches of size", self.config.batch_size_val)
        else:
            print("Early-stopping is INACTIVE.")
        
        # Prepare attributes for data logging
        self.train_log, self.val_log, self.best_states = CSVData(os.path.join(self.dirpath, TRAIN_LOG)), CSVData(os.path.join(self.dirpath, VAL_LOG)), CSVData(os.path.join(self.dirpath, BEST_LOG))
        # Set neural net to training mode
        self.model.train()
        # Initialize epoch counter
        epoch = 0.
        # Initialize iteration counter
        iteration = 0
        # Training loop
        while int(epoch+0.5) < epochs and continue_train:
            print('\nEpoch',int(epoch+0.5),'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            # Loop over data samples and into the network forward function
            for i, data in enumerate(self.train_iter):
                
                # Data and label
                self.data = data[0]
                self.label = data[1].long()
                
                # Call forward: make a prediction & measure the average error
                res = self.forward(True)
                # Call backward: backpropagate error and update weights
                self.backward()
                # Epoch update
                epoch += 1./len(self.train_iter)
                iteration += 1
                
                # Log/Report
                #
                # Record the current performance on train set
                self.train_log.record(['iteration','epoch','accuracy','loss'],[iteration,epoch,res['accuracy'],res['loss']])
                self.train_log.write()
                # once in a while, report
                if i==0 or (i+1)%report_interval == 0:
                    if i != 0:
                        print('\r', end='')
                    print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f' % (iteration,epoch,res['loss'],res['accuracy']), end='')
                    
                # Run validation every valid_interval training batches
                if run_es and (i+1)%valid_interval == 0:
                    with torch.no_grad():
                        # self.model.eval()
                        # val_data = next(iter(self.val_iter))
                        
                        ## Data and label
                        # self.data = val_data[0]
                        # self.label = val_data[1].long()
                        
                        # res = self.forward(False)
                        
                        print('')
                        res = self.validate(load_best=False, batches=valid_batches, save_state=False, save_plots=False)
                        self.val_log.record(['iteration','epoch','accuracy','loss'],[iteration,epoch,res['accuracy'],res['loss']])
                        self.val_log.write()
                    self.model.train()
                    continue_train = True
                    # Save best-so-far training state and record its position in the training log
                    if(best_val_loss-res["loss"] > LOSS_THRESHOLD):
                        best_val_loss = res["loss"]
                        self.save_state(curr_iter_str=BEST_FLAG)
                        self.best_states.record(['iteration','epoch','accuracy','loss'],[iteration,epoch,res['accuracy'],res['loss']])
                        self.best_states.write()
                    
                if epoch >= epochs:
                    break
                    
                # Save on the given intervals
                if(i+1)%save_interval == 0:
                    self.save_state(curr_iter_str=str(iteration) if save_all else LATEST_FLAG)
            print('\r', end='')
            print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f' % (iteration,epoch,res['loss'],res['accuracy']), end='')
            
            if epoch >= epochs:
                    break
            
        print('')
        
        self.val_log.close()
        self.train_log.close()
        self.best_states.close()

    # Function to test the model performance on the validation
    # dataset ( returns loss, acc, confusion matrix )
    def validate(self, load_best=True, batches=None, plt_worst=0, plt_best=0, save_state=True, save_plots=True):
        r"""Test the trained model on the validation set.
        
        Parameters: None
        
        Outputs : 
            total_val_loss = accumulated validation loss
            avg_val_loss = average validation loss
            total_val_acc = accumulated validation accuracy
            avg_val_acc = accumulated validation accuracy
            
        Returns : None
        """
        
        if self.dset is None or len(self.dset.val_indices) == 0:
            print("No examples in validation set, skipping validation...")
            return
        
        # If requested, load the best state saved so far
        if load_best:
            try:
                candidates = [os.path.join(self.state_dir, name) for name in os.listdir(self.state_dir) if name.endswith(str(BEST_FLAG))]
                if len(candidates) > 0:
                    self.restore_state(candidates[0])
                else:
                    print("Warning: attempted to restore best state from", self.state_dir,"but no best state weight file was detected.")
            except FileNotFoundError:
                print ("Warning:", self.state_dir, "directory not found, cannot restore best state.")
        
        # Variables to output at the end
        val_loss = 0.0
        val_acc = 0.0
        val_iterations = 0
        
        pushing = False
        if plt_worst > 0 or plt_best > 0:
            if self.dset.has_traceback:
                queues = [DoublePriority(plt_worst, plt_best), DoublePriority(plt_worst, plt_best), DoublePriority(plt_worst, plt_best)]
                pushing = True
            else:
                print("Warning: attempted to perform root traceback on dataset without traceback capability, skipping.")
        
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Variables for the confusion matrix
            loss, accuracy, labels, predictions, softmaxes = [],[],[],[],[]
            if self.dset.has_energies: energies = []
            
            # Extract the event data and label from the DataLoader iterator
            for i, val_data in enumerate(iter(self.val_iter)):
                
                if i != 0:
                    print('\r', end='')
                print("val_iterations: "+str(val_iterations)+"/"+str(len(self.val_iter) if batches is None else min(batches, len(self.val_iter))), end='')
                
                # Stop after specified number of batches
                if batches is not None and i >= batches:
                    break
                
                self.data, self.label = val_data[0:2]
                self.label = self.label.long()

                # Run the forward procedure and output the result
                result = self.forward(False)
                val_loss += result['loss']
                val_acc += result['accuracy']
                
                # Add item to priority queues if necessary
                if pushing:
                    PATH, IDX = val_data[3:5]
                    IDX = IDX.long().numpy()
                    for i, lab in enumerate(self.label):
                        queues[lab].insert((result['softmax'][i][lab], PATH[i], IDX[i]))
                
                # Copy the tensors back to the CPU
                self.label = self.label.to("cpu")
                
                # Add the local result to the final result
                loss.append(val_loss)
                accuracy.append(val_acc)
                labels.extend(self.label)
                predictions.extend(result['prediction'])
                softmaxes.extend(result["softmax"])
                if self.dset.has_energies:
                    energies.extend(val_data[2])
                
                val_iterations += 1
        
        avg_loss = val_loss/val_iterations
        avg_acc = val_acc/val_iterations
        print("\nTotal val loss : ", val_loss,
              "\nTotal val acc : ", val_acc,
              "\nAvg val loss : ", val_loss/val_iterations,
              "\nAvg val acc : ", val_acc/val_iterations)
        
        # If requested, dump list of root files + indices to save_path directory
        # TODO: Resolve coupling of formatting with implementation of event_display in DataTools
        if pushing:
            plot_path = os.path.join(self.config.save_path, EXTR_DIR)
            if not os.path.exists(plot_path):
                os.mkdir(plot_path)
            wl_lo = open(plot_path+LIST_LO, 'w+')
            wl_hi = open(plot_path+LIST_HI, 'w+')
            worst, best = [], []
            for i in range(len(queues)):
                q = queues[i]
                # Lowest softmax are worst
                worst.extend(q.getsmallest())
                # Highest softmax are best
                best.extend(q.getlargest())
                
            for event in worst:
                wl_lo.write(str(event[0])+' '+event[1]+' '+str(event[2])+'\n')
            for event in best:
                wl_hi.write(str(event[0])+' '+event[1]+' '+str(event[2])+'\n')
            
            wl_lo.close()
            wl_hi.close()
            
            print("Dumped lists of extreme events at", plot_path)
        
#        np_softmaxes = np.array(softmaxes)
#
#        np.save("labels" + str(run) + ".npy", np.hstack(labels))
#        np.save("energies" + str(run) + ".npy", np.hstack(energies))
#        np.save("predictions" + str(run) + ".npy", np.hstack(predictions))
#        np.save("softmax" + str(run) + ".npy",
#                np_softmaxes.reshape(np_softmaxes.shape[0]*np_softmaxes.shape[1],
#                                    np_softmaxes.shape[2]))
        
        # If requested, save data for analysis
        if save_state:
            plot_data_path = os.path.join(self.config.save_path, VAL_STATE)
            if self.dset.has_energies:
                np.savez_compressed(plot_data_path,
                                    prediction=np.array(predictions),
                                    softmax=np.array(softmaxes),
                                    loss=np.array(loss),
                                    accuracy=np.array(accuracy),
                                    labels=np.array(labels),
                                    energies=np.array(energies),
                                    data=self.config.path)
            else:
                np.savez_compressed(plot_data_path,
                                    prediction=np.array(predictions),
                                    softmax=np.array(softmaxes),
                                    loss=np.array(loss),
                                    accuracy=np.array(accuracy),
                                    labels=np.array(labels),
                                    data=self.config.path)
            print("Dumped result array to", plot_data_path)
            
        return {'loss': avg_loss, 'accuracy': avg_acc}
            
    # Function to test the model performance on the test
    # dataset ( returns loss, acc, confusion matrix )
    def test(self):
        r"""Test the trained model on the test dataset.
        
        Parameters: None
        
        Outputs : 
            total_test_loss = accumulated validation loss
            avg_test_loss = average validation loss
            total_test_acc = accumulated validation accuracy
            avg_test_acc = accumulated validation accuracy
            
        Returns : None
        """
        
        if self.dset is None or len(self.dset.test_indices) == 0:
            print("No examples in testing set, skipping testing...")
            return
        
        # Variables to output at the end
        test_loss = 0.0
        test_acc = 0.0
        test_iterations = 0
        
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Extract the event data and label from the DataLoader iterator
            for test_data in iter(self.test_iter):
                
                sys.stdout.write("\r\r\r" + "test_iterations : " + str(test_iterations))
                
                self.data, self.label = test_data[0:2]
                self.label = self.label.long()
                
                counter = collections.Counter(self.label.tolist())
                sys.stdout.write("\ncounter : " + str(counter))

                # Run the forward procedure and output the result
                result = self.forward(False)
                test_loss += result['loss']
                test_acc += result['accuracy']
                
                test_iterations += 1
         
        print("\nTotal test loss : ", test_loss,
              "\nTotal test acc : ", test_acc,
              "\nAvg test loss : ", test_loss/test_iterations,
              "\nAvg test acc : ", test_acc/test_iterations)

    def save_state(self, curr_iter_str=LATEST_FLAG):
        # If saving a best state, update best_state attribute
        if not os.path.isdir(self.state_dir):
            os.mkdir(self.state_dir)
        filename = self.state_dir+str(self.config.model[1] if len(self.config.model) > 1 else self.config.model[0])+"_"+curr_iter_str
        if os.path.exists(filename):
            os.remove(filename)
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict()
        }, filename)
        print('\tSaved checkpoint as:', filename)
        return filename

    def restore_state(self, weight_file):
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            print('Restoring state from', weight_file)
            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f,map_location="cuda:0" if (self.config.device == 'gpu') else 'cpu')
            # load network weights
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # load iteration count
            if 'global_step' in checkpoint.keys():
                self.iteration = checkpoint['global_step']
        print('Restoration complete.')
        
    # Function that calls the functions of result_visualizer module on the contents of the Engine object
    def dump_plots(self, plot_data_path=None):
        rv.dump_training_visuals(os.path.join(self.dirpath, TRAIN_LOG), os.path.join(self.dirpath, VAL_LOG), best_csv_path=os.path.join(self.dirpath, BEST_LOG),
                                 save_path=self.config.save_path)
        plot_result = rv.open_result(os.path.join(self.config.save_path, VAL_STATE) if plot_data_path is None else plot_data_path)
        rv.dump_validation_visuals(plot_result, save_path=self.config.save_path)
