
# Required packages

# Core numerical & computation libraries
import numpy as np
np.random.seed(1606)    
import random
random.seed(1606)       
import time
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# functions
def loss_function_plot(train_losses, step=10, figsize=(17, 5)):  
    epochs = range(len(train_losses))

    plt.figure(figsize=figsize)
    plt.plot(
        epochs,
        train_losses,
        marker='o',
        linestyle='-',
        color='teal',
        alpha=0.5
    )

    xticks = sorted(set(list(range(0, len(train_losses), step)) + [len(train_losses) - 1]))
    plt.xticks(xticks)

    plt.tick_params(axis='both', labelsize=9)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Epoch-Based Training Loss Curve")
    plt.show()
