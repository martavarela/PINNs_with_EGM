import deepxde as dde
import numpy as np
from scipy.ndimage import laplace
from scipy.interpolate import griddata
#from dde.backend import torch
import torch
import os
from callback import AdaptiveLossWeights

class PINN():
    
    def __init__(self, dynamics, dim, heter, inverse):
        
        ## Dynamics
        self.dynamics = dynamics
        self.dim = dim
        self.heter = heter
        self.inverse = inverse
        
        # ## PDE Parameters (2D spatial + temporal)
        self.input = 3
        self.num_hidden_layers = 5
        self.hidden_layer_size = 60
        self.batch_size = 64
        self.num_domain = 10000
        self.num_boundary = 1000
        self.epochs_main = 15000 #150000
        self.output = 3
        
        ## Training Parameters
        self.num_domain = 20000 # number of training points within the domain
        self.num_boundary = 1000 # number of training boundary condition points on the geometry boundary
        self.num_test = 1000 # number of testing points within the domain
        self.MAX_MODEL_INIT = 16 # maximum number of times allowed to initialize the model
        self.MAX_LOSS = 5e2 # upper limit to the initialized loss
        self.epochs_init = 150 #15000 # number of epochs for training initial phase
        self.epochs_main = 1000 #1000 # number of epochs for main training phase
        self.lr = 0.0005 # learning rate
        if self.inverse:
            self.lr = 0.0001
    
    def define_pinn(self, geomtime, input_data, observe_train):

        ## Define the network
        self.net = dde.maps.FNN([self.input] + [self.hidden_layer_size] * self.num_hidden_layers + [self.output], "tanh", "Glorot uniform")
        
        ## Select relevant PDE (Dim, Heterogeneity, forward/inverse)
        if self.dim == 1:
            pde = self.dynamics.pde_1D
        elif self.dim == 2 and self.heter:
            if self.inverse and 'd' in self.inverse:
                pde = self.dynamics.pde_2D_heter
                self.net.apply_output_transform(self.dynamics.modify_inv_heter)
            else:
                pde = self.dynamics.pde_2D_heter_forward
                self.net.apply_output_transform(self.dynamics.modify_heter)
        elif self.dim == 2 and not self.heter:
            pde = self.dynamics.pde_2D     
        
        ## Define PINN model
        self.pde_data = dde.data.TimePDE(geomtime, pde, input_data,
                            num_domain = self.num_domain, 
                            num_boundary=self.num_boundary, 
                            anchors=observe_train,
                            num_test=self.num_test)    
        self.model = dde.Model(self.pde_data, self.net)
        self.model.compile("adam", lr=self.lr)
        return 0
    
    
    def train(self, out_path, params):
        ## Train PINN with corresponding scheme
        torch.cuda.empty_cache()
        print('---------Training-----------')
        losshistory, train_state = self.train_3_phase(out_path, params)
     
        return self.model, losshistory, train_state
    
    def train_3_phase(self, out_path, params):
        #init_weights = [0,0,0,0,0,0,1]
        init_weights = [0,0,0,0,0,0,1] # pde phie
        if self.inverse:
            variables_file = "variables_" + self.inverse + ".dat"
            variable = dde.callbacks.VariableValue(params, period=20, filename=variables_file)    
            ## Initial phase (REDUCED LEARNING RATE)
            self.model.compile("adam", lr=0.0005, loss_weights=init_weights,external_trainable_variables=params)
            losshistory, train_state = self.model.train(epochs=self.epochs_init, model_save_path = out_path, callbacks=[variable],display_every=100,batch_size = self.batch_size)
            ## Main phase
            main_weights = [1,1,1,1,1,1,1]
            # Warm-up for loss to stabilise
            self.model.compile("adam",lr=self.lr,external_trainable_variables=params,loss_weights=main_weights)
            losshistory, train_state = self.model.train(epochs=500, model_save_path = out_path, callbacks=[variable],display_every=100,batch_size = self.batch_size)
            loss_train = np.array(train_state.loss_train)
            epsilon = 1e-8
            main_weights = 1/(loss_train)
            #main_weights = 1/np.log(loss_train+epsilon+1)
            main_weights = np.clip(main_weights,0.001,10)
            main_weights /= np.sum(main_weights)
            print(f"Adaptive weights:{main_weights}")
            # continue training
            self.model.compile("adam",lr=self.lr,external_trainable_variables=params,loss_weights=main_weights)
            losshistory, train_state = self.model.train(epochs=self.epochs_main, model_save_path = out_path, callbacks=[variable],display_every=100,batch_size = self.batch_size)
            # ## Final phase
            #self.model.compile("L-BFGS-B",external_trainable_variables=params)
            #losshistory, train_state = self.model.train(model_save_path = out_path, callbacks=[variable])
        else:
            ## Initial phase: only cares about data-fitting
            self.model.compile("adam", lr=0.0005, loss_weights=init_weights)
            losshistory, train_state = self.model.train(epochs=self.epochs_init, model_save_path = out_path,batch_size = self.batch_size,display_every=100)
            ## Main phase
            main_weights = [1,1,1,1,1,1,1]
            # Warm-up for loss to stabilise
            self.model.compile("adam", lr=self.lr,loss_weights=main_weights)
            losshistory, train_state = self.model.train(epochs=500, model_save_path = out_path,batch_size = self.batch_size,display_every=100)
            # Loss weights adaptation
            loss_train = np.array(train_state.loss_train)
            epsilon = 1e-8
            main_weights = 1/(loss_train)
            #main_weights = 1/np.log(loss_train+epsilon+1)
            main_weights = np.clip(main_weights,0.001,10)
            main_weights /= np.sum(main_weights)
            print(f"Adaptive weights:{main_weights}")
            # Continue main training
            self.model.compile("adam", lr=self.lr,loss_weights=main_weights)
            losshistory, train_state = self.model.train(epochs=self.epochs_main, model_save_path = out_path,batch_size = self.batch_size,display_every=100)
            ## Final phase
            # self.model.compile("L-BFGS-B")
            # losshistory, train_state = self.model.train(model_save_path = out_path,batch_size = self.batch_size)

        return losshistory, train_state
