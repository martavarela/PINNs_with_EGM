from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

def plot_phie(dynamics,phie_pred,true_phie,filename):
    #true_phie = np.reshape()
    plt.figure(figsize=(15, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.plot(phie_pred[i], label="Predicted", linestyle="--")
        plt.plot(true_phie[i, :], label="True", alpha=0.7)
        plt.title(f"Signal {dynamics.elecpos[i,:]}")
        #plt.xlabel("Time Steps")
        plt.ylabel("φₑ")
        plt.grid(True)
        plt.legend(fontsize=8)
    plt.suptitle("Predicted vs True Electropotential (φₑ) for Each Electrode", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300)
    return 0

def plot_multiple_phies(phie_preds,true_phie):
    plt.figure(figsize=(15, 10))
    # plotting predictions
    signals = []
    for i in range(np.shape(phie_preds)[0]):
        plt.plot(phie_preds[i][0], label=f"Predicted{i}", linestyle="--",alpha=0.5)
        signals.append(phie_preds[i][0])
    # plotting averages
    max_len = max(len(sig) for sig in signals)
    resampled = []
    for sig in signals:
        x_old = np.linspace(0, 1, len(sig))
        x_new = np.linspace(0, 1, max_len)
        f = interpolate.interp1d(x_old, sig, kind='linear', fill_value="extrapolate")
        resampled.append(f(x_new))
    average_signal = np.mean(resampled, axis=0)
    plt.plot(average_signal, label="Average", alpha=0.7,linewidth=3)


    plt.plot(true_phie[0, :], label="True")
    plt.title(f"Comparison between Seeds")
    #plt.xlabel("Time Steps")
    plt.ylabel("φₑ")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.savefig("phie_compare.png", dpi=300)
    return 0


def plot_loss(losshistory):
    labels=['V from PDE(bc)','W from PDE(bc)','V from PDE(ic)','W from PDE(ic)','Data (V)','Data(phie)']
    loss_matrix = np.vstack(losshistory.loss_train)
    num_components = loss_matrix.shape[1]

    plt.figure(figsize=(15, 10))
    for i in range(num_components):
        steps = np.array(losshistory.steps)
        values = np.array(loss_matrix[:,i])
        sorted_indices = np.argsort(steps)
        steps_sorted = steps[sorted_indices]
        values_sorted = values[sorted_indices]
        unique_steps = []
        unique_values = []
        current_step=1
        for step, val in zip(steps_sorted, values_sorted):
            if step != current_step & step != 0:
                unique_steps.append(step)
                unique_values.append(val)
                current_step = step
        plt.semilogy(unique_steps, unique_values, label=labels[i], marker='o', linestyle='-')
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss History")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.savefig("losshistory.png", dpi=300)
    return 0

def plot_variable(filename, var):
    if 'a' in var:
        true = 0.05
    if 'b' in var:
        true = 0.15
    if 'D' in var:
        true = 0.1
    with open(filename, "r") as f:
            lines = f.readlines()
    steps = []
    values = []
    for line in lines:
        parts = line.strip().split()
        step = int(parts[0])
        value = float(parts[1].strip("[]"))  # remove square brackets and convert to float
        steps.append(step)
        values.append(value)
    steps = np.array(steps)
    values = np.array(values)
    sorted_indices = np.argsort(steps)
    steps_sorted = steps[sorted_indices]
    values_sorted = values[sorted_indices]
    unique_steps = []
    unique_values = []
    current_step=0

    for step, val in zip(steps_sorted, values_sorted):
        if step != current_step:
            unique_steps.append(step)
            unique_values.append(val)
            current_step = step
    # Plot
    plt.clf()
    plt.scatter(unique_steps,unique_values)
    plt.axhline(y=true, color='r', linestyle='--', label='True value')
    plt.xlabel("Training step")
    plt.ylabel("Estimated parameter value")
    plt.title("Parameter Estimation Over Time")
    plt.grid(True)
    plt.savefig(f"{var}_estimate.png")
    return 0
