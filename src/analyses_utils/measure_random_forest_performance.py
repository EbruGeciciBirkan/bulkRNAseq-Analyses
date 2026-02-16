
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

# function
def measure_random_forest_performance(X, y, test_size = 0.2, n_splits = 4, random_state = 1234):
  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, shuffle = True,
                                                      test_size = test_size, random_state = random_state)

  kfold_cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
  param_grid = {"n_estimators": [100, 200, 300]}
  grid_search = GridSearchCV(RandomForestClassifier(random_state = random_state), param_grid = param_grid, cv = kfold_cv)
  grid_search.fit(X_train, y_train)

  y_train_hat = grid_search.predict(X_train)
  print(accuracy_score(y_train, y_train_hat))

  y_test_hat = grid_search.predict(X_test)
  print(accuracy_score(y_test, y_test_hat))
