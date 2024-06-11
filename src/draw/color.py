import matplotlib.pyplot as plt
from src.config import CMAP_NAME
CMAP = plt.get_cmap(CMAP_NAME)

def to_hex(rgba):
    # Convert from [0, 1] range to [0, 255]
    r, g, b, a = rgba
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}{int(a * 255):02x}"
