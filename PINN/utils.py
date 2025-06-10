import scipy.io
import deepxde as dde
from deepxde.backend import tf
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import laplace
import torch
import torch.nn.functional as F

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
        self.max_y = np.max(y)
        X = X.reshape(-1, 1)
        T = T.reshape(-1, 1)
        V = Vsav.reshape(-1, 1)
        W = Wsav.reshape(-1, 1)    
        if dim == 1:     
            return np.hstack((X, T)), V, W
        
        self.spacing = 0.1 #as in matlab
        self.t_spacing = (t[:,1]-t[:,0]).mean()
        phie, elecpos = data["phie"], data["elecpos"]  
        elecpos = np.transpose(elecpos)
        observe_elec = []
        for i, j in elecpos:
            for time in t[0]:
                observe_elec.append([j, i, time])

        # normalisation of phie
        # self.mean_phie = np.mean(phie, axis=1, keepdims=True)
        # self.std_phie = np.std(phie, axis=1, keepdims=True)
        self.mean_phie = np.mean(phie)
        self.std_phie = np.std(phie)
        phie_norm = (phie - self.mean_phie) / self.std_phie
        phie_norm = phie_norm.reshape(-1,1) #UNNORMALISED

        self.elecpos = elecpos
        self.nelec = np.shape(elecpos)[0]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        
        # self.x_grid = torch.linspace(self.min_x, self.max_x, steps=int(self.max_x), device=device, dtype=dtype)
        # self.y_grid = torch.linspace(self.min_y, self.max_y, steps=int(self.max_y/self.spacing), device=device, dtype=dtype)
        # self.t_grid = torch.linspace(self.min_t, self.max_t, steps=int(self.max_t/self.t_spacing), device=device, dtype=dtype)
        # self.V_grid = torch.zeros((len(self.x_grid), len(self.y_grid), len(self.t_grid)),
        #                     device=device, dtype=dtype)
        self.x_grid = x.squeeze()
        self.y_grid = y.squeeze()
        self.t_grid = t.squeeze()
        self.V_grid = torch.zeros(Vsav.shape, dtype=torch.float32)
        print(f'Shape of V_grid:{self.V_grid.shape}')
        # replacing with ground truth V
        #Vsav_tensor = torch.tensor(Vsav, dtype=self.V_grid.dtype, device=self.V_grid.device)
        #self.V_grid = Vsav_tensor

        self.mean_phie = torch.tensor(self.mean_phie,device=device,dtype=dtype)
        self.std_phie = torch.tensor(self.std_phie,device=device,dtype=dtype)
        
        self.D_array = torch.full((int(self.max_x/self.spacing), int(self.max_y/self.spacing)), self.D, device=device, dtype=dtype)
        return np.hstack((X, Y, T)), V, W, phie_norm, np.array(observe_elec)

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
        
        x_idxs = torch.round(x[:, 0]).long().clamp(0, len(self.x_grid)-1)
        y_idxs = torch.round(x[:, 1]).long().clamp(0, len(self.y_grid)-1)
        t_idxs = torch.round(x[:, 2]).long().clamp(0, len(self.t_grid)-1)
        
        self.V_grid.data[x_idxs, y_idxs, t_idxs] = V.squeeze()
        # updating V_grid slowly
        #alpha = 0.1
        # self.V_grid[x_idxs, y_idxs, t_idxs] = (
        #     (1 - alpha) * self.V_grid[x_idxs, y_idxs, t_idxs] + alpha * V.squeeze()
        # )
        self.mask = (self.V_grid != 0).float()
        #self.fill_missing_V()

        ## Aliev-Panfilov PDE
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)

        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  self.D*(dv_dxx + dv_dyy) + self.k*V*(V-self.a)*(V-1) +W*V 
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))

        # output minus integration of V after update and normalisation
        self.V_grid = self.fill_V_grid_with_interp(self.V_grid,self.mask)
        
        phie_pred = self.compute_phie(self.V_grid,x)
        phie_pred = (phie_pred - self.mean_phie) / (self.std_phie + 1e-8)
        eq_c = phie-phie_pred 
        return [eq_a, eq_b, eq_c]

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

    def del2_torch(self,V,h):
        """Approximate Laplacian using 5-point stencil."""

        # Pad with Neumann boundary conditions (replicate)
        #V_pad = torch.nn.functional.pad(V.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='reflect')
        #V_pad = V_pad.squeeze()
        V_pad = V
        lap = torch.zeros_like(V)

        # Calculate Laplacian
        lap[1:-1, 1:-1] = (V_pad[2:, 1:-1] + V_pad[0:-2, 1:-1] + V_pad[1:-1, 2:] + V_pad[1:-1, 0:-2] - 4 * V[1:-1, 1:-1])
        
        # Linear extrapolation to edges ---
        lap[0   , :] = 2 * lap[1   , :] - lap[2   , :]
        lap[-1  , :] = 2 * lap[-2  , :] - lap[-3  , :]
        # left and right columns
        lap[:, 0   ] = 2 * lap[:, 1   ] - lap[:, 2   ]
        lap[:, -1  ] = 2 * lap[:, -2  ] - lap[:, -3  ]

        # proper scaling with spacing
        scale = 10 / (4*(h**2))
        return lap*scale

    def compute_phie(self, Vsav, observations):
        """
        Compute phie for a batch of electrode positions and times.
        Vsav: (X, Y, T) — torch.Tensor
        observations: (N, 3) — each row is (x, y, t)
        Returns: (N,) torch.Tensor of phie values
        """
        device = Vsav.device
        X, Y, T = Vsav.shape
        D = self.D

        # Expand scalar D if necessary
        if isinstance(D, (float, int)):
            D = torch.full((X, Y), D, device=device)
        elif D.shape == (X+2, Y+2):
            D = D[1:-1, 1:-1]

        h = self.spacing
        Dy, Dx = torch.gradient(D, spacing=(h, h), dim=(0, 1))

        # Get time indices (convert to 0-based if needed)
        t_indices = observations[:, 2].long() - 1  # Assuming t starts at 1
        #unique_t, inverse_t = torch.unique(t_indices, return_inverse=True)
        
        du_all = []
        for t in range(T): #unique_t:
            V = Vsav[:, :, t]
            gy, gx = torch.gradient(V, spacing=(h, h), dim=(0, 1))
            du = 4 * D * self.del2_torch(V,h) + (Dx * gx) + (Dy * gy)
            #if t==1: # math with t=2 in matlab
                # print('-----------V-----------')
                # print(V)
                # print('-----------du-----------')
                # print(du)
            du_all.append(du)  # Trim boundaries
        
        # Stack results and select appropriate du for each observation
        du_stack = torch.stack(du_all)  # (unique_T, X-2, Y-2)
        #du_selected = du_stack[inverse_t]  # (N, X-2, Y-2)

        # Prepare coordinate grids
        x_grid = torch.arange(2,X, device=device)
        y_grid = torch.arange(2,Y, device=device)
        grid_x, grid_y = torch.meshgrid(x_grid, y_grid, indexing='ij')

        # Compute distances for all observations
        obs_xy = observations[:, :2].unsqueeze(1).unsqueeze(1)  # (N, 1, 1, 2)
        grid_pos = torch.stack([grid_x, grid_y], dim=-1)  # (X-2, Y-2, 2)
        
        # Broadcasting to compute all distances at once
        deltas = grid_pos - obs_xy  # (N, X-2, Y-2, 2) 
        distances = torch.norm(deltas, dim=-1)  # (N, X-2, Y-2)
        # Avoid division by zero
        mask_radius = 5
        distances = torch.where(distances < 1e-10, torch.ones_like(distances), distances)
        distances = torch.where(distances > mask_radius, torch.tensor(float('inf')),distances) 
        du_selected = du_stack[t_indices]
        # Compute phie for all observations
        phie = -torch.sum(du_selected[:,1:X-1, 1:Y-1].permute(0,2,1) / distances, dim=(1, 2))  # (N,)
        scale = 1.0 # K in matlab
        phie = scale*phie
        return phie
    
    def arrange_phie(self,obs, phie_pred):
        arranged_phie = []
        for i in range(self.elecpos.shape[0]):
            x,y = self.elecpos[i]
            mask = np.isclose(obs[:, 0], x, atol=0.05) & np.isclose(obs[:, 1], y,atol=0.05)
            # Extract time and phie values for that spatial point
            t_values = obs[mask, 2]
            phie_values = phie_pred[mask]
            # Sort by time
            sorted_indices = np.argsort(t_values)
            t_sorted = t_values[sorted_indices]
            phie_sorted = phie_values[sorted_indices].squeeze()
            arranged_phie.append(phie_sorted)
        return np.array(arranged_phie,dtype=object)
    
    def fill_missing_V(self):
        """Fill missing values without breaking autograd."""
        with torch.no_grad():
            mask = (self.V_grid == 0) | torch.isnan(self.V_grid)
            if not mask.any():
                return self.V_grid  # Return early if nothing to fill

            # Detach a copy for filling (no gradients)
            V_filled = self.V_grid.detach().clone()
            
            # 2D Laplace kernel
            laplace_kernel = torch.tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], device=self.V_grid.device, dtype=self.V_grid.dtype).view(1, 1, 3, 3)

            # Process each timestep
            for t in range(V_filled.size(2)):
                V_t = V_filled[:, :, t].unsqueeze(0).unsqueeze(0)  # [1, 1, X, Y]
                V_padded = F.pad(V_t, (1, 1, 1, 1), mode='reflect')
                
                for _ in range(50):
                    laplacian = F.conv2d(V_padded, laplace_kernel).squeeze()
                    V_filled[:, :, t][mask[:, :, t]] += 0.1 * laplacian[mask[:, :, t]]
            
            self.V_grid.data.copy_(V_filled)
    
    def fill_V_grid_with_interp(self,V_grid, mask):
        """
        Interpolates missing values in a partially filled V_grid using trilinear interpolation.

        Args:
            V_grid (torch.Tensor): Tensor of shape (nx, ny, nt) containing known values and zeros elsewhere.
            mask (torch.Tensor): Tensor of shape (nx, ny, nt) with 1 where values are known, 0 where unknown.

        Returns:
            torch.Tensor: Fully filled V_grid of shape (nx, ny, nt) with interpolated values.
        """
        device = V_grid.device
        nx, ny, nt = V_grid.shape

        # Normalised grid coordinates in [-1, 1]
        x = torch.linspace(-1, 1, nx, device=device)
        y = torch.linspace(-1, 1, ny, device=device)
        t = torch.linspace(-1, 1, nt, device=device)
        grid_x, grid_y, grid_t = torch.meshgrid(x, y, t, indexing='ij')

        # Grid for grid_sample: shape (1, nx, ny, nt, 3)
        coords = torch.stack((grid_t, grid_y, grid_x), dim=-1).unsqueeze(0)

        # Prepare input for grid_sample: shape (1, 1, nx, ny, nt)
        V_input = V_grid.unsqueeze(0).unsqueeze(0)

        # Interpolate missing values with trilinear interpolation
        V_interpolated = F.grid_sample(
            V_input, coords, mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze()  # shape (nx, ny, nt)

        # Replace missing values only (preserve known values)
        V_filled = V_grid * mask + V_interpolated * (1 - mask)

        return V_filled


