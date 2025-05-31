import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
file_name = 'ground_truth_dataset.mat'
dir_path = os.path.dirname(os.path.realpath(file_name))
sys.path.append(dir_path)
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde # version 0.11 or higher
#from generate_plots_1d import plot_1D
#from generate_plots_2d import plot_2D
import utils
import pinn
from generate_plots import plot_variable, plot_loss, plot_phie, plot_multiple_phies
import matplotlib.pyplot as plt
import torch
torch.cuda.empty_cache()
print("CUDA Available:", torch.cuda.is_available())

# Parameters
file_name = 'ground.mat'
model_folder_name = '/output_model'
dim = 2
add_noise = False
add_w_input = False
inverse = None
heter = False
plot_results = True
create_animation = False

# Other parameters
noise_factor = 0.1
test_size = 0.75
seed = [42]

# Set random seed
dde.config.set_random_seed(seed[0])

# System dynamics
dynamics = utils.system_dynamics()
params = dynamics.params_to_inverse(inverse)

# Load data
observe_x, V, W, phie, observe_elec = dynamics.generate_data(file_name, dim)

# Split data
observe_train, observe_test, v_train, v_test, w_train, w_test = train_test_split(
    observe_x, V, W, test_size=test_size
)
elec_train, elec_test, phie_train, phie_test = train_test_split(
    observe_elec, phie, test_size=test_size
)

# Add noise if needed
if add_noise:
    v_train += noise_factor * np.random.randn(*v_train.shape)
print(dynamics.x_grid)

geomtime = dynamics.geometry_time(dim)
bc = dynamics.BC_func(dim, geomtime)
ic = dynamics.IC_func(observe_train, v_train)
ic2 = dynamics.IC_func(elec_train, phie_train)

observe_phie = dde.PointSetBC(elec_train, phie_train, component=2)
input_data = [bc,ic,ic2,observe_phie]
if add_w_input:
    input_data.append(dde.PointSetBC(observe_train, w_train, component=1))

# Define model
torch.cuda.empty_cache()
model_pinn = pinn.PINN(dynamics, dim, heter, inverse)
model_pinn.define_pinn(geomtime, input_data, observe_train)

# Train model
out_path = dir_path + model_folder_name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on device:", device)
model, losshistory, train_state = model_pinn.train(out_path, params)

# Plot loss history
dde.utils.external.plot_loss_history(losshistory, 'losshistory.png')
print('Final loss weights:',model.loss_weights)

# Predict V
pred = model.predict(observe_test)
v_pred = pred[:, 0:1]

# Predict Phie
pred = model.predict(observe_elec)
phie_pred = pred[:, 2:3]
phie_pred = dynamics.arrange_phie(observe_elec, phie_pred)

true_phie = phie.reshape(dynamics.nelec, int(dynamics.max_t))
plot_phie(phie_pred, true_phie, 'true_vs_pred.png')

# Compute rMSE
rmse_v = np.sqrt(np.square(v_pred - v_test).mean())
rmse_phie = np.sqrt(np.square(phie_pred - true_phie).mean())
print("---------------------------------")
print("V rMSE:", rmse_v)
print("Phie rMSE:", rmse_phie)
print("---------------------------------")




