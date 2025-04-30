import sys
import os        
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde # version 0.11 or higher
from generate_plots_1d import plot_1D
from generate_plots_2d import plot_2D
import utils
import pinn
from generate_plots import plot_variable, plot_loss, plot_phie, plot_multiple_phies
import matplotlib.pyplot as plt
import torch
torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file-name', dest='file_name', required = True, type = str, help='File name for input data')
    parser.add_argument('-m', '--model-folder-name', dest='model_folder_name', required = False, type = str, help='Folder name to save model (prefix /)')
    parser.add_argument('-d', '--dimension', dest='dim', required = True, type = int, help='Model dimension. Needs to match the input data')
    parser.add_argument('-n', '--noise', dest='noise', action='store_true', help='Add noise to the data')
    parser.add_argument('-w', '--w-input', dest='w_input', action='store_true', help='Add W to the model input data')
    parser.add_argument('-v', '--inverse', dest='inverse', required = False, type = str, help='Solve the inverse problem, specify variables to predict (e.g. a / ad / abd')
    parser.add_argument('-ht', '--heter', dest='heter', required = False, action='store_true', help='Predict heterogeneity - only in 2D')    
    parser.add_argument('-p', '--plot', dest='plot', required = False, action='store_true', help='Create and save plots')
    parser.add_argument('-a', '--animation', dest='animation', required = False, action='store_true', help='Create and save 2D Animation')
    args = parser.parse_args()

## General Params
noise = 0.1 # noise factor
test_size = 0.9 # precentage of testing data

def main(args):
    phie_preds = []

    # random seed setting
    seed = [42,19,3407]
    for i in range(len(seed)):
        dde.config.set_random_seed(seed[i])

        ## Get Dynamics Class
        dynamics = utils.system_dynamics()
        
        ## Parameters to inverse (if needed)
        params = dynamics.params_to_inverse(args.inverse)
        
        ## Generate Data 
        file_name = args.file_name
        observe_x, V, W, phie, observe_elec = dynamics.generate_data(file_name, args.dim) 
        
        ## Split data to train and test
        observe_train, observe_test, v_train, v_test, w_train, w_test = train_test_split(observe_x,V,W,test_size=test_size)
        
        # elec is subset of observe
        elec_train,elec_test,phie_train,phie_test = train_test_split(observe_elec,phie,test_size = test_size)
    
        ## Add noise to training data if needed
        if args.noise:
            v_train = v_train + noise*np.random.randn(v_train.shape[0], v_train.shape[1])

        ## Geometry and Time domains
        geomtime = dynamics.geometry_time(args.dim)
        ## Define Boundary Conditions
        bc = dynamics.BC_func(args.dim, geomtime)
        ## Define Initial Conditions
        ic = dynamics.IC_func(observe_train, v_train)
        ic2 = dynamics.IC_func(elec_train, phie_train)
        
        ## Model observed data
        observe_v = dde.PointSetBC(observe_train, v_train, component=0)
        input_data = [bc, ic, observe_v]
        if args.w_input: ## If W required as an input
            observe_w = dde.PointSetBC(observe_train, w_train, component=1)
            input_data = [bc, ic, observe_v, observe_w]

        observe_phie = dde.PointSetBC(elec_train,phie_train,component=2)
        #input_data = [bc,ic,observe_v,observe_phie] # have to remove observe_v
        input_data = [bc,ic2,observe_phie]
        ## Select relevant PDE (Dim, Heterogeneity) and define the Network
        model_pinn = pinn.PINN(dynamics, args.dim,args.heter, args.inverse)
        model_pinn.define_pinn(geomtime, input_data, observe_train)
                
        ## Train Network
        out_path = dir_path + args.model_folder_name
        print("CUDA available:",torch.cuda.is_available())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Training using device:",device)
        #model, losshistory, train_state = model_pinn.train(out_path, params)
        #dde.utils.external.plot_loss_history(losshistory,'loss_hist_plot.png')
        model = model_pinn.model

        ## Compute rMSE
        pred = model.predict(observe_test)
        v_pred = pred[:,0:1]

        pred = model.predict(elec_test)
        phie_pred = pred[:,2:3]
        phie_pred = model_pinn.arrange_phie(elec_test,phie_pred)
        phie_preds.append(phie_pred)
        


        rmse_v = np.sqrt(np.square(v_pred - v_test).mean())
        rmse_phie = np.sqrt(np.square(np.concatenate(phie_pred) - phie_test).mean())
        print('--------------------------')
        print("V rMSE for test data:", rmse_v)
        print('--------------------------')
        print("Phie rMSE for test data:", rmse_phie)
        print('--------------------------')
        print("Arguments: ", args)

    
    
    ## Save predictions, data
    # np.savetxt("train_data.dat", np.hstack((observe_train, v_train, w_train)),header="observe_train,v_train, w_train")
    # np.savetxt("test_pred_data.dat", np.hstack((observe_test, v_test,v_pred, w_test, w_pred)),header="observe_test,v_test, v_pred, w_test, w_pred")
    
    ## Generate Figures
    # data_list = [observe_x, observe_train, v_train, V]
    # if args.plot and args.dim == 1:
    #     plot_1D(data_list,dynamics, model, args.model_folder_name)
    # elif args.plot and args.dim == 2:
    #     plot_2D(data_list,dynamics, model, args.animation, args.model_folder_name)   
    true_phie = phie.reshape(dynamics.nelec,int(dynamics.max_t))
    #print(phie_preds)
    plot_multiple_phies(phie_preds,true_phie)
    #plot_loss(losshistory)

    if args.inverse:
        plot_variable("variables_" + args.inverse + ".dat",args.inverse)
    return model

## Run main code
model = main(args)
