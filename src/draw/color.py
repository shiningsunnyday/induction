import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.config import CMAP_NAME
import numpy as np
CMAP = plt.get_cmap(CMAP_NAME)

# n_labels = 180
# # Create a colormap with n_labels unique colors
# color_array = plt.cm.hsv(np.linspace(0, 1, n_labels))
# CMAP = LinearSegmentedColormap.from_list("custom_cmap", color_array, N=n_labels)

def to_hex(rgba):
    # Convert from [0, 1] range to [0, 255]
    r, g, b, a = rgba
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}{int(a * 255):02x}"
