import deepxde as dde
import numpy as np
from collections import deque

class AdaptiveLossWeights(dde.callbacks.Callback):
    def __init__(self,window_size=5):
        self.initial_losses = None
        self.model = None
        self.loss_history = deque(maxlen=window_size)

    def set_model(self,model):
        self.model = model

    # def on_batch_end(self):
    #     losses = np.array(self.model.train_state.loss_train)
    #     if self.running_losses is None:
    #         self.running_losses = losses
    #     else:
    #         self.running_losses += losses
    #     self.num_batches += 1

    def modify_weights(self,mean_losses):
        mean_losses = np.array(mean_losses)

        # On first call, store initial loss values
        if self.initial_losses is None:
            self.initial_losses = mean_losses.copy()

        # Avoid zero division
        ratios = mean_losses / self.initial_losses
        ratios = np.maximum(ratios, 1e-8)

        # Inverse of relative loss
        weights = 1.0 / ratios
        weights = np.clip(weights,0.001, 5.0)
        # Optional: Normalize weights to keep sum = 1 (or other scaling)
        weights /= np.sum(weights)

        return weights.tolist()

    def on_epoch_end(self):
        losses = np.array(self.model.train_state.loss_train)
        self.loss_history.append(losses)

        mean_losses = np.mean(self.loss_history,axis=0)
        if self.initial_losses is None:
            self.initial_losses = mean_losses.copy()
        
        ratios = mean_losses/self.initial_losses
        ratios = np.maximum(ratios,1e-8)
        weights = np.clip(weights,0.001,5.0)
        weights /= np.sum(weights)

        self.model.loss_weights = weights.tolist()

        #if self.num_batches>0:
        # Get last epoch's individual loss values
            # mean_losses = self.running_losses/self.num_batches
            # new_weights = self.modify_weights(mean_losses)
            # self.model.loss_weights = new_weights
            # print(f"Epoch {self.model.train_state.epoch}: Loss weights = {new_weights}")
            # self.running_losses = None
            # self.num_batches = 0
