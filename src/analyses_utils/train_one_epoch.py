
# Required packages

# Core numerical & computation libraries
# Required packages
# Single-cell analysis & embedding
import scanpy as sc 
from scimilarity.cell_embedding import CellEmbedding
from scimilarity.utils import align_dataset, lognorm_counts

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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# functions
# Creation of the Training Loop Function
# It enables the deep learning model to be trained in batches over a specified number of epochs on the training data.
def train_one_epoch(model,train_loader,optimizer,loss_fn):
    running_loss = 0.
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)
