# Required packages
# Core numerical & computation libraries
import torch
torch.manual_seed(1606) 
import numpy as np
np.random.seed(1606)    
import random
random.seed(1606)       
import time
import pandas as pd

# Machine learning (scikit-learn)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

# function
import matplotlib.pyplot as plt

def embedding_plots(
    data,
    classes,
    random_state=1606,
    figsize=(9, 5),
    point_size=10,
    alpha=0.7,
    legend_title="Cohort"
):
    le = LabelEncoder()
    y_encoded = le.fit_transform(classes)
    palette = sns.color_palette("tab20", 20) + sns.color_palette("Set3", 13)

    tsne_model = TSNE(
        n_components=2,      
        perplexity=30,       
        random_state=random_state,   
        metric="euclidean"
    )
    tsne_embeddings = tsne_model.fit_transform(data)

    umap_model = umap.UMAP(
        random_state=random_state,
        metric="euclidean",
        n_neighbors=10,
        min_dist=0.5
    )
    umap_embeddings = umap_model.fit_transform(data)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=y_encoded, 
                    cmap="tab20", s=point_size, alpha=alpha)
    axes[0].set_title("UMAP", fontsize=11)

    axes[1].scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=y_encoded, 
                    cmap="tab20", s=point_size, alpha=alpha)
    axes[1].set_title("t-SNE", fontsize=11)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1) 
    # ----------------------------------------------

    labels = [str(x).split("-", 1)[-1] for x in le.classes_]
    handles = [plt.Line2D([], [], marker="o", color=palette[i % len(palette)], 
               linestyle="", markersize=6, label=label) for i, label in enumerate(labels)]

    fig.legend(handles=handles, title=legend_title, loc="lower center", 
               bbox_to_anchor=(0.5, -0.1), ncol=min(len(labels), 6), 
               prop={"family": "Arial", "size": 7}, frameon=False)

    plt.subplots_adjust(wspace=0.15)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()