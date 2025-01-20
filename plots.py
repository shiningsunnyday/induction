import matplotlib.pyplot as plt
import numpy as np

param_vals = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
val_loss = np.array([4.017,3.935,3.882,  
    3.885,  
    3.888, 
    3.903, 
    3.889, 
    3.878,  
    3.882,  
    3.888,  
    3.883,  
    3.874, 
    3.884,  
    3.879, 
    3.894,  
    3.876 
])

def plot_param_vs_val_loss():
    best_idx = np.argmin(val_loss)
    fig, ax = plt.subplots(dpi=150)
    ax.plot(param_vals, val_loss, marker='o', color='C0', label='Validation Loss')    
    ax.plot(param_vals[best_idx], val_loss[best_idx],
            marker='o', color='red', markersize=10, 
            label=f'Best (# encoder layers = {param_vals[best_idx]}, {val_loss[best_idx]:.3f})')
    ax.set_xlabel('# encoder layers')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Hyperparameter Scan')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    plt.show()
    plt.savefig("encoder_layers_scan.png", dpi=300)

if __name__ == "__main__":
    plot_param_vs_val_loss()
