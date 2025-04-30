import scipy.io
import deepxde as dde
from deepxde.backend import tf
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import laplace
import torch

class system_dynamics():
    
    def __init__(self):
        
        ## PDE Parameters
        self.a = 0.05 #0.01
        self.b = 0.15
        self.D = 0.1
        self.k = 8
        self.mu_1 = 0.2
        self.mu_2 = 0.3
        self.epsilon = 0.002

        ## Geometry Parameters
        self.min_x = 0.1
        self.max_x = 10            
        self.min_y = 0.1 
        self.max_y = 10
        self.min_t = 1
        self.max_t = 70
        self.spacing = 0.1
        self.t_spacing = 1

        


    def generate_data(self, file_name, dim):
        
        data = scipy.io.loadmat(file_name)
        if dim == 1:
            t, x, Vsav, Wsav = data["t"], data["x"], data["Vsav"], data["Wsav"]
            X, T = np.meshgrid(x, t)
        elif dim == 2:
            t, x, y, Vsav, Wsav = data["t"], data["x"], data["y"], data["Vsav"], data["Wsav"]
            X, T, Y = np.meshgrid(x,t,y)
            Y = Y.reshape(-1, 1)
        else:
            raise ValueError('Dimesion value argument has to be either 1 or 2')
        self.max_t = np.max(t)
        self.max_x = np.max(x)
        X = X.reshape(-1, 1)
        T = T.reshape(-1, 1)
        V = Vsav.reshape(-1, 1)
        W = Wsav.reshape(-1, 1)    
        if dim == 1:     
            return np.hstack((X, T)), V, W
        
        phie, elecpos = data["phie"], data["elecpos"]
        elecpos = np.transpose(elecpos)
        observe_elec = []
        for x, y in elecpos:
            for time in t[0]:
                observe_elec.append([x, y, time])

        # normalisation of phie
        self.mean_phie = np.mean(phie, axis=1, keepdims=True)
        self.std_phie = np.std(phie, axis=1, keepdims=True)
        phie_norm = (phie - self.mean_phie) / self.std_phie
        phie_norm = phie_norm.reshape(-1,1)

        self.elecpos = elecpos
        self.nelec = np.shape(elecpos)[0]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        self.x_grid = torch.linspace(self.min_x, self.max_x, steps=int(self.max_x/self.spacing), device=device, dtype=dtype)
        self.y_grid = torch.linspace(self.min_y, self.max_y, steps=int(self.max_y/self.spacing), device=device, dtype=dtype)
        self.t_grid = torch.linspace(self.min_t, self.max_t, steps=int(self.max_t/self.t_spacing), device=device, dtype=dtype)
        self.V_grid = torch.zeros((len(self.x_grid), len(self.y_grid), len(self.t_grid)),
                          device=device, dtype=dtype)

        #print(f'V_grid shape: {self.V_grid.shape}')
        # print(f'x_grid shape: {self.x_grid.shape}')
        # print(f'y_grid shape: {self.y_grid.shape}')
        # print(f't_grid shape: {self.t_grid.shape}'

        self.mean_phie = torch.tensor(self.mean_phie,device=device,dtype=dtype)
        self.std_phie = torch.tensor(self.std_phie,device=device,dtype=dtype)
        self.D_array = torch.full((int(self.max_x/self.spacing), int(self.max_y/self.spacing)), self.D, device=device, dtype=dtype)
        self.Dx, self.Dy = torch.gradient(self.D_array, spacing=self.spacing)
        return np.hstack((X, Y, T)), V, W, np.array(phie_norm), np.array(observe_elec)

    def geometry_time(self, dim):
        if dim == 1:
            geom = dde.geometry.Interval(self.min_x, self.max_x)
            timedomain = dde.geometry.TimeDomain(self.min_t, self.max_t)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)    
        elif dim == 2:
            geom = dde.geometry.Rectangle([self.min_x,self.min_y], [self.max_x,self.max_y])
            timedomain = dde.geometry.TimeDomain(self.min_t, self.max_t)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        else:
            raise ValueError('Dimesion value argument has to be either 1 or 2')
        return geomtime

    def params_to_inverse(self,args_param):
        
        params = []
        if not args_param:
            return self.a, self.b, self.D, params
        ## If inverse:
        ## The tf.variables are initialized with a positive scalar, relatively close to their ground truth values

        if 'a' in args_param:
            #self.a = tf.math.exp(tf.Variable(-3.92))
            self.a = dde.Variable(np.exp(-3.92))
            #self.log_a = dde.Variable(-3.92)
            params.append(self.a)
        if 'b' in args_param:
            #self.b = tf.math.exp(tf.Variable(-1.2))
            self.b = dde.Variable(np.exp(-3.92))
            params.append(self.b)
        if 'd' in args_param:
            #self.D = tf.math.exp(tf.Variable(-1.6))
            self.D = dde.Variable(np.exp(-1.6))
            params.append(self.D)
        return params

    def pde_1D(self, x, y):
        
    
        V, W = y[:, 0:1], y[:, 1:2]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=1)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=1)
        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  self.D*dv_dxx + self.k*V*(V-self.a)*(V-1) +W*V 
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]

    def pde_1D_2cycle(self,x, y):
    
        V, W = y[:, 0:1], y[:, 1:2]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=1)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=1)
    
        x_space,t_space = x[:, 0:1],x[:, 1:2]
        t_stim_1 = tf.equal(t_space, 0)
        t_stim_2 = tf.equal(t_space, int(self.max_t/2))
        x_stim = tf.less_equal(x_space, 5*self.spacing)
    
        first_cond_stim = tf.logical_and(t_stim_1, x_stim)
        second_cond_stim = tf.logical_and(t_stim_2, x_stim)
    
        I_stim = tf.ones_like(x_space)*0.1
        I_not_stim = tf.ones_like(x_space)*0
        Istim = tf.where(tf.logical_or(first_cond_stim,second_cond_stim),I_stim,I_not_stim)
        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  self.D*dv_dxx + self.k*V*(V-self.a)*(V-1) +W*V -Istim
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]
    
    def pde_2D(self, x, y):
        V, W, phie = y[:, 0:1], y[:, 1:2], y[:,2:3]

        # # Map (x, y, t) to grid indices
        # x_idxs = torch.argmin(torch.abs(self.x_grid[None, :] - x[:, 0:1]), dim=1)
        # y_idxs = torch.argmin(torch.abs(self.y_grid[None, :] - x[:, 1:2]), dim=1)
        # t_idxs = torch.argmin(torch.abs(self.t_grid[None, :] - x[:, 2:3]), dim=1)
        
        x_idxs = ((x[:, 0] / self.spacing).round().long().clamp(0, len(self.x_grid)-1))
        y_idxs = ((x[:, 1] / self.spacing).round().long().clamp(0, len(self.y_grid)-1))
        t_idxs = ((x[:, 2] / self.t_spacing).round().long().clamp(0, len(self.t_grid)-1))

        # Update V_grid
        # for i in range(len(x)):
        #     self.V_grid[x_idxs[i], y_idxs[i], t_idxs[i]] = V[i]
        self.V_grid.data[x_idxs, y_idxs, t_idxs] = V.squeeze()

        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)

        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  self.D*(dv_dxx + dv_dyy) + self.k*V*(V-self.a)*(V-1) +W*V 
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))

        # output minus integration of V after update and normalisation
        # phie_pred = self.gpu_integration(self.V_grid[:,:,t_idxs],x)
        phie_pred = self.gpu_integration(self.V_grid[:,:,t_idxs],x)
        phie_pred = (phie_pred - phie_pred.mean()) / (phie_pred.std() + 1e-8) # normalise w predictions own mean and std
        eq_c = phie - phie_pred
        return [eq_a, eq_b, eq_c]
    
    def phie_constraint(self,x,y):
        phie_pred = self.gpu_integration(self.V_grid,x)
        return y[:,2:3] - phie_pred

    def pde_2D_heter(self, x, y):
    
        V, W, var = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        dv_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dv_dy = dde.grad.jacobian(y, x, i=0, j=1)
        
        ## Heterogeneity
        D_heter = tf.math.sigmoid(var)*0.08+0.02
        dD_dx = dde.grad.jacobian(D_heter, x, i=0, j=0)
        dD_dy = dde.grad.jacobian(D_heter, x, i=0, j=1)
        
        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  D_heter*(dv_dxx + dv_dyy) -dD_dx*dv_dx -dD_dy*dv_dy + self.k*V*(V-self.a)*(V-1) +W*V 
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]
 
    def pde_2D_heter_forward(self, x, y):
                
        V, W, D = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        dv_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dv_dy = dde.grad.jacobian(y, x, i=0, j=1)
        
        ## Heterogeneity
        dD_dx = dde.grad.jacobian(D, x, i=0, j=0)
        dD_dy = dde.grad.jacobian(D, x, i=0, j=1)
        
        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  D*(dv_dxx + dv_dyy) -dD_dx*dv_dx -dD_dy*dv_dy + self.k*V*(V-self.a)*(V-1) +W*V 
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]   
 
    def IC_func(self,observe_train, v_train):
        
        T_ic = observe_train[:,-1].reshape(-1,1)
        idx_init = np.where(np.isclose(T_ic,1))[0]
        v_init = v_train[idx_init]
        observe_init = observe_train[idx_init]
        return dde.PointSetBC(observe_init,v_init,component=0)
    
    def BC_func(self,dim, geomtime):
        if dim == 1:
            bc = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), lambda _, on_boundary: on_boundary, component=0)
        elif dim == 2:
            bc = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), self.boundary_func_2d, component=0)
        return bc
    
    def boundary_func_2d(self,x, on_boundary):
            return on_boundary and ~(x[0:2]==[self.min_x,self.min_y]).all() and  ~(x[0:2]==[self.min_x,self.max_y]).all() and ~(x[0:2]==[self.max_x,self.min_y]).all()  and  ~(x[0:2]==[self.max_x,self.max_y]).all() 
   
    def modify_inv_heter(self, x, y):                
        domain_space = x[:,0:2]
        D = tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(domain_space, 60,
                            tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 1, activation=None)        
        return tf.concat((y[:,0:2],D), axis=1)    
    
    def modify_heter(self, x, y):
        
        x_space, y_space = x[:, 0:1], x[:, 1:2]
        
        x_upper = tf.less_equal(x_space, 54*0.1)
        x_lower = tf.greater(x_space,32*0.1)
        cond_1 = tf.logical_and(x_upper, x_lower)
        
        y_upper = tf.less_equal(y_space, 54*0.1)
        y_lower = tf.greater(y_space,32*0.1)
        cond_2 = tf.logical_and(y_upper, y_lower)
        
        D0 = tf.ones_like(x_space)*0.02 
        D1 = tf.ones_like(x_space)*0.1
        D = tf.where(tf.logical_and(cond_1, cond_2),D0,D1)
        return tf.concat((y[:,0:2],D), axis=1)

    def num_integration(self,v_pred,obs):
        # obs: (x,y,t)
        # v_pred is defined at 
        # use indices from obs to find corresponding V that is predicted, no need of interpolation
        # recalling network at every training step
        # try: have a fixed grid connected to the model
        # have v_pred match obs shape
        # gridding v_pred data
        # N = 25
        # x_grid = np.arange(int(self.min_x), int(self.max_x), self.spacing)
        # y_grid = np.arange(int(self.min_y), int(self.max_y), self.spacing)
        # t_grid = np.arange(int(self.min_t), int(self.max_t), 1)
        
        # X_grid, Y_grid, T_grid = np.meshgrid(x_grid, y_grid, t_grid, indexing='ij')# make sure its consistent with the synthetic data
        
        # Interpolate V at grid points
        # V_grid = griddata(obs, v_pred, (X_grid, Y_grid, T_grid), method='nearest') 
        # V_grid = np.nan_to_num(V_grid, nan=np.nanmean(v_pred))

        # steps = int(v_pred.size/(int(self.max_x)*int(self.max_y)))
        # v_pred = v_pred.reshape(steps, int(self.max_x), int(self.max_y))
        # v_pred = np.transpose(v_pred, (1, 2, 0))
       
       # remember to change np to tf and 32 64 casting
        X, Y, tend = self.max_x, self.max_y, int(self.max_t)
        # if np.isscalar(self.D): # stay with homogeneity
        #     D_array = self.D*np.ones([int(X/self.spacing),int(Y/self.spacing)])
        # elif np.shape(self.D)[0] == X+2 and np.shape(self.D)[1] == Y+2:
        #     D_array = self.D[1:-1,1:-1]

        D_array = self.D*np.ones([int(X/self.spacing),int(Y/self.spacing)])
        [Dx,Dy] = np.gradient(D_array,self.spacing,self.spacing)
        #[xel,yel] = np.meshgrid(self.elecpos[0,:],self.elecpos[1,:], indexing = 'ij')

        tt = 0
        #phie_pred = np.zeros([np.shape(self.elecpos)[1],tend])
        phie_pred = np.zeros(obs.shape)
        cutoff_radius = 10
        #for t in range(tend-1):
        #tt=tt+1
        for o in obs:
            x_obs,y_obs,t_obs = o
            x_idx = np.argmin(np.abs(self.x_grid - x_obs.item()))
            y_idx = np.argmin(np.abs(self.y_grid - y_obs.item()))
            t_idx = np.argmin(np.abs(self.t_grid - t_obs.item()))
            v_t = np.squeeze(self.V_grid[:,:,t_idx])
            #v_t = np.squeeze(self.V_grid[:,:,int(t.item())])
            gx,gy = np.gradient(v_t,self.spacing)
            du = 4 * D_array * laplace(v_t, mode='nearest') + (Dx * gx) + (Dy * gy)
            #for k in range(np.shape(self.elecpos)[1]):
            #matx, maty = np.meshgrid((self.x_grid[2:] - self.elecpos[0][k]), (self.y_grid[2:] - self.elecpos[1][k]))
            matx, maty = np.meshgrid((self.x_grid[2:] - x_obs.item()), (self.y_grid[2:] - y_obs.item()))
            
            distance = np.sqrt(matx**2 + maty**2).T  # Replace zeros with a small value to avoid division by zero
            distance = np.where(distance == 0, 1e-8, distance)
            mask = distance <= cutoff_radius
            phie_obs = np.trapz(np.trapz((du[1:-1,1:-1]*mask)/distance,axis=0),axis=0)
                    #phie_pred[k,tt-1] = -sum  # find for only one obs
            phie_pred = np.append(phie_pred,phie_obs) 
        phie_pred = torch.tensor(phie_pred, dtype=torch.float32, requires_grad=True)         
        return phie_pred
    
    def phie_integration(self, v_pred, obs):
        device = v_pred.device
        dtype = v_pred.dtype
        
        # Precompute values
        X, Y = self.max_x, self.max_y
        spacing = self.spacing
        spacing = self.spacing
        cutoff_radius = 5
        cutoff_radius_sq = cutoff_radius**2
        
        # Convert grids to tensors if needed
        x_grid_t = self.x_grid.to(device) if isinstance(self.x_grid, torch.Tensor) else torch.tensor(self.x_grid, device=device, dtype=dtype)
        y_grid_t = self.y_grid.to(device) if isinstance(self.y_grid, torch.Tensor) else torch.tensor(self.y_grid, device=device, dtype=dtype)
        t_grid_t = self.t_grid.to(device) if isinstance(self.t_grid, torch.Tensor) else torch.tensor(self.t_grid, device=device, dtype=dtype)
        V_grid_t = self.V_grid.to(device) if isinstance(self.V_grid, torch.Tensor) else torch.tensor(self.V_grid, device=device, dtype=dtype)
        
        # Pre-allocate output tensor (detached from computation graph)
        phie_pred = torch.zeros(len(obs), device=device, dtype=dtype)
        
        # Convert obs to tensor if needed
        obs_t = obs if isinstance(obs, torch.Tensor) else torch.tensor(np.array([(o[0].item(), o[1].item(), o[2].item()) for o in obs]), 
                                    device=device, dtype=dtype)
        D_array = torch.full((int(X/spacing), int(Y/spacing)), self.D, 
                        device=device, dtype=dtype)
        Dx, Dy = torch.gradient(D_array, spacing=spacing)
        
        # Main computation
        for i in range(len(obs_t)):
            x_obs, y_obs, t_obs = obs_t[i]
            
            # Find nearest grid indices
            x_idx = torch.argmin(torch.abs(x_grid_t - x_obs))
            y_idx = torch.argmin(torch.abs(y_grid_t - y_obs))
            t_idx = torch.argmin(torch.abs(t_grid_t - t_obs))
            
            # Get voltage field
            v_t = V_grid_t.to(device)[:, :, t_idx].squeeze()
            
            # Compute gradients
            gx, gy = torch.gradient(v_t, spacing=spacing)
            
            # Compute Laplacian
            laplacian = self.compute_laplacian(v_t, spacing)
            
            # Compute du
            du = 4 * D_array * laplacian + (Dx * gx) + (Dy * gy)
            
            # Compute distances
            x_dist = x_grid_t[2:] - x_obs
            y_dist = y_grid_t[2:] - y_obs
            xx, yy = torch.meshgrid(x_dist, y_dist, indexing='xy')
            distance_sq = xx**2 + yy**2
            
            # Compute integrand
            with torch.no_grad():  # Temporarily disable gradients for this part
                mask = distance_sq <= cutoff_radius_sq
                inv_distance = torch.where(mask, 1/torch.sqrt(distance_sq + 1e-8), torch.tensor(0.0, device=device))
                integrand = du[1:-1, 1:-1] * inv_distance.T
            
            # Compute integral (create new tensor for assignment)
            # integral_value = torch.trapz(torch.trapz(integrand, dim=0), dim=0)
            # phie_pred = torch.cat([phie_pred[:i], integral_value.unsqueeze(0), phie_pred[i+1:]])

            # Compute integral
            integral_value = torch.sum(integrand) * (spacing**2)
            # Assign directly
            phie_pred[i] = integral_value
        
        # Set requires_grad only on the final output
        return phie_pred.detach().requires_grad_(True)

    def compute_laplacian(self, v_t, spacing):
        kernel = torch.tensor([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]], device=v_t.device, dtype=v_t.dtype) / (spacing**2)
        return torch.nn.functional.conv2d(v_t.unsqueeze(0).unsqueeze(0), 
                                        kernel.unsqueeze(0).unsqueeze(0),
                                        padding=1).squeeze()
    

    def gpu_integration(self, v_t, obs):
        device = obs.device
        dtype = obs.dtype

        # Laplacian operation on batch
        gx, gy = torch.gradient(v_t, spacing=(self.spacing, self.spacing), dim=(0,1))
        gxx, _ = torch.gradient(gx, spacing=(self.spacing, self.spacing), dim=(0,1))
        _, gyy = torch.gradient(gy, spacing=(self.spacing, self.spacing), dim=(0,1))
        laplacian = gxx + gyy

        du = 4 * self.D * laplacian + self.Dx.unsqueeze(-1)*gx + self.Dy.unsqueeze(-1)*gy
    
        # Vectorized distance computation
        x_dist = self.x_grid[1:-1, None] - obs[:, 0]  # [nx-2, n_obs]
        y_dist = self.y_grid[1:-1, None] - obs[:, 1]   # [ny-2, n_obs]
        r_sq = x_dist[:, None]**2 + y_dist[None, :]**2  # [nx-2, ny-2, n_obs]

        mask = r_sq <= 4.0
        weights = torch.where(mask, 1/(torch.sqrt(r_sq) + 1e-8), 0)
        return torch.trapz(torch.trapz(du[1:-1, 1:-1] * weights, dx=self.spacing), dx=self.spacing)

        ###########################################################
        # # Precompute constants
        # X, Y = self.max_x, self.max_y
        # spacing = self.spacing
        # cutoff_radius_sq = 4  # 5**2
        
        # # obs_t = torch.stack([torch.tensor([o[0].item(), o[1].item(), o[2].item()], 
        # #                     device=device, dtype=dtype) for o in obs])
        
        # # Precompute D terms
        # D_array = torch.full((int(X/spacing), int(Y/spacing)), self.D, device=device, dtype=dtype)
        # Dx, Dy = torch.gradient(D_array, spacing=spacing)
        
        # # Batch-compute all indices first
        # x_idxs = torch.argmin(torch.abs(self.x_grid - obs[:, 0, None]), dim=1)
        # y_idxs = torch.argmin(torch.abs(self.y_grid - obs[:, 1, None]), dim=1)
        # t_idxs = torch.argmin(torch.abs(self.t_grid - obs[:, 2, None]), dim=1)
        
        # # Pre-allocate output tensor
        # phie_pred = torch.zeros(len(obs),1, device=device, dtype=dtype)
        
        # # Vectorized distance computation
        # # x_dists = self.x_grid[2:].unsqueeze(0) - obs[:, 0].unsqueeze(1)  # shape: (n_obs, n_x)
        # # y_dists = self.y_grid[2:].unsqueeze(0) - obs[:, 1].unsqueeze(1)  # shape: (n_obs, n_y)
        
        # # Main computation (batched)
        # for i, (x_idx, y_idx, t_idx) in enumerate(zip(x_idxs, y_idxs, t_idxs)):
        #     v_t = v_grid[:, :, t_idx]
            
        #     # Compute gradients and Laplacian
        #     gx, gy = torch.gradient(v_t, spacing=spacing)
        #     laplacian = self.compute_laplacian(v_t, spacing)

        #     du = 4 * D_array * laplacian + (Dx * gx) + (Dy * gy)
            
        #     x_dist = (self.x_grid[1:-1] - obs[i, 0]).unsqueeze(1)
        #     y_dist = (self.y_grid[1:-1] - obs[i, 1]).unsqueeze(0)
        #     # Batched distance computation
        #     #xx, yy = torch.meshgrid(x_dist, y_dist, indexing='xy')
        #     distance_sq = x_dist**2 + y_dist**2
            
        #     # Vectorized integrand
        #     with torch.no_grad():
        #         mask = distance_sq <= cutoff_radius_sq
        #         inv_distance = torch.where(mask, 1/torch.sqrt(distance_sq + 1e-8), 0)
        #         integrand = du[1:-1, 1:-1] * inv_distance.T
            
        #     # Efficient assignment
        #     phie_pred[i,0] = torch.trapz(torch.trapz(integrand, dim=0), dim=0)
    
        # return phie_pred.requires_grad_(True)
    
