import deepxde as dde
import numpy as np
from scipy.ndimage import laplace
from scipy.interpolate import griddata
#from dde.backend import torch
import torch
import os

class PINN():
    
    def __init__(self, dynamics, dim, heter, inverse):
        
        ## Dynamics
        self.dynamics = dynamics
        self.dim = dim
        self.heter = heter
        self.inverse = inverse
        
        ## PDE Parameters (initialized for 1D PINN)
        self.input = 2 # network input size 
        self.num_hidden_layers = 4 # number of hidden layers for NN 
        self.hidden_layer_size = 32 # size of each hidden layers 
        self.batch_size = 32
        self.output = 3 # network input size 
        
        ## Training Parameters
        self.num_domain = 20000 # number of training points within the domain
        self.num_boundary = 1000 # number of training boundary condition points on the geometry boundary
        self.num_test = 1000 # number of testing points within the domain
        self.MAX_MODEL_INIT = 16 # maximum number of times allowed to initialize the model
        self.MAX_LOSS = 5 # upper limit to the initialized loss
        self.epochs_init = 15 #15000 # number of epochs for training initial phase
        self.epochs_main = 10 #1000 # number of epochs for main training phase
        self.lr = 0.0005 # learning rate
        
        ## Update constants for 2D and/or heterogeneity geometry
        self.modify_2d_const()
    
    def modify_2d_const(self):
        ## Update the PINN design for 2D and/or heterogeneity geometry
        if self.dim == 2:
            self.input = 3
            self.num_hidden_layers = 5
            self.hidden_layer_size = 60
            self.num_domain = 40000
            self.num_boundary = 4000
            self.epochs_main = 50 #150000
        if self.heter:
            self.output = 3
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
        
    def stable_init(self):
        
        ## Stabalize initialization process by capping the losses
        losshistory, _ = self.model.train(epochs=1, batch_size = self.batch_size)
        initial_loss = max(losshistory.loss_train[0])
        num_init = 1
        while initial_loss>self.MAX_LOSS or np.isnan(initial_loss):
            num_init += 1
            self.model = dde.Model(self.pde_data, self.net)
            self.model.compile("adam", lr=self.lr)
            losshistory, _ = self.model.train(epochs=1,batch_size = self.batch_size)
            initial_loss = max(losshistory.loss_train[0])
            if num_init > self.MAX_MODEL_INIT:
                raise ValueError('Model initialization phase exceeded the allowed limit')
        return 0
    
    
    def train(self, out_path, params):
        print('---------Stabilising-----------')
        ## Stabalize initialization process by capping the losses
        self.stable_init()
        ## Train PINN with corresponding scheme
        print('---------Training-----------')
        if self.dim ==1:
            losshistory, train_state = self.train_1_phase(out_path, params)
        elif self.dim == 2:
            losshistory, train_state = self.train_3_phase(out_path, params)

        # best_model = dde.Model(self.pde_data, self.net)
        # best_model.restore("best_model")
     
        return self.model, losshistory, train_state
        
    def train_1_phase(self, out_path, params):
        if self.inverse:
            variables_file = "variables_" + self.inverse + ".dat"
            variable = dde.callbacks.VariableValue(params, period=10, filename=variables_file)    
            losshistory, train_state = self.model.train(epochs=self.epochs_main, model_save_path = out_path, callbacks=[variable])
        else:
            losshistory, train_state = self.model.train(epochs=self.epochs_main, model_save_path = out_path)
        return losshistory, train_state
    
    def train_3_phase(self, out_path, params):
        #init_weights = [0,0,0,0,0,0,1]
        init_weights = [0,0,0,0,0,1]
        if self.inverse:
            variables_file = "variables_" + self.inverse + ".dat"
            variable = dde.callbacks.VariableValue(params, period=20, filename=variables_file)    
            ## Initial phase
            self.model.compile("adam", lr=0.0005, loss_weights=init_weights,external_trainable_variables=params)
            losshistory, train_state = self.model.train(epochs=self.epochs_init, model_save_path = out_path, callbacks=[variable],display_every=50,batch_size = self.batch_size)
            ## Main phase
            #main_weights = [10,1,1,1,1,1]
            self.model.compile("adam",lr=self.lr,external_trainable_variables=params)
            losshistory, train_state = self.model.train(epochs=self.epochs_main, model_save_path = out_path, callbacks=[variable],display_every=50,batch_size = self.batch_size)
            # ## Final phase
            self.model.compile("L-BFGS-B",external_trainable_variables=params)
            losshistory, train_state = self.model.train(model_save_path = out_path, callbacks=[variable])
        else:
            # os.makedirs("best_model", exist_ok=True)
            # checkpoint = dde.callbacks.ModelCheckpoint("best_model",save_better_only=True, period=100,verbose=1)
            ## Initial phase: only cares about data-fitting
            self.model.compile("adam", lr=0.0005, loss_weights=init_weights)
            losshistory, train_state = self.model.train(epochs=self.epochs_init, model_save_path = out_path,batch_size = self.batch_size,display_every=100)
            ## Main phase
            #main_weights = [1,1,10,1,1,1]
            self.model.compile("adam", lr=self.lr)
            losshistory, train_state = self.model.train(epochs=self.epochs_main, model_save_path = out_path,batch_size = self.batch_size,display_every=100)
            ## Final phase
            self.model.compile("L-BFGS-B")
            losshistory, train_state = self.model.train(model_save_path = out_path,batch_size = self.batch_size)

        return losshistory, train_state

    # def predict(self,obs):
    #     pred = self.model.predict(obs)
    #     #phie_pred = self.num_integration(pred[:,0],obs)
    #     return pred

    def arrange_phie(self,obs, phie_pred):
        arranged_phie = []
        for i in range(self.dynamics.elecpos.shape[0]):
            x,y = self.dynamics.elecpos[i]
            mask = np.isclose(obs[:, 0], x, atol=0.05) & np.isclose(obs[:, 1], y,atol=0.05)
            # Extract time and phie values for that spatial point
            t_values = obs[mask, 2]
            phie_values = phie_pred[mask]
            # Sort by time
            sorted_indices = np.argsort(t_values)
            t_sorted = t_values[sorted_indices]
            phie_sorted = phie_values[sorted_indices].squeeze()
            arranged_phie.append(np.array(phie_sorted))
        return np.array(arranged_phie,dtype=object)
 
    def num_integration(self,v_pred,obs):
            # gridding v_pred data
            N = 25
            x_grid = np.arange(int(self.dynamics.min_x), int(self.dynamics.max_x), self.dynamics.spacing)
            y_grid = np.arange(int(self.dynamics.min_y), int(self.dynamics.max_y), self.dynamics.spacing)
            t_grid = np.arange(int(self.dynamics.min_t), int(self.dynamics.max_t), 1)
            
            X_grid, Y_grid, T_grid = np.meshgrid(x_grid, y_grid, t_grid, indexing='ij')
            
            # Interpolate V at grid points
            V_grid = griddata(obs, v_pred, (X_grid, Y_grid, T_grid), method='nearest')
            V_grid = np.nan_to_num(V_grid, nan=np.nanmean(v_pred))

            X, Y, tend = self.dynamics.max_x, self.dynamics.max_y, int(self.dynamics.max_t)
            if np.isscalar(self.dynamics.D):
                D_array = self.dynamics.D*np.ones([int(X/self.dynamics.spacing),int(Y/self.dynamics.spacing)])
            elif np.shape(self.dynamics.D)[0] == X+2 and np.shape(self.dynamics.D)[1] == Y+2:
                D_array = self.dynamics.D[1:-1,1:-1]
            [Dx,Dy] = np.gradient(D_array,self.dynamics.spacing,self.dynamics.spacing)
            [xel,yel] = np.meshgrid(self.dynamics.elecpos[0,:],self.dynamics.elecpos[1,:], indexing = 'ij')

            tt = 0
            phie_pred = np.zeros([np.shape(self.dynamics.elecpos)[1],tend])
            
            cutoff_radius = 10
            for t in range(tend-1):
                tt=tt+1
                v_t = np.squeeze(V_grid[:,:,t])
                gx,gy = np.gradient(v_t,self.dynamics.spacing)
                du = 4 * D_array * laplace(v_t, mode='nearest') + (Dx * gx) + (Dy * gy)
                for k in range(np.shape(self.dynamics.elecpos)[1]):
                    matx, maty = np.meshgrid((x_grid[2:] - self.dynamics.elecpos[0][k]), (y_grid[2:] - self.dynamics.elecpos[1][k]))
                    
                    distance = np.sqrt(matx**2 + maty**2).T  # Replace zeros with a small value to avoid division by zero
                    mask = distance <= cutoff_radius
                    sum = np.trapz(np.trapz((du[1:-1,1:-1]*mask)/distance,axis=0),axis=0)
                    phie_pred[k,tt-1] = -sum        
            return phie_pred
