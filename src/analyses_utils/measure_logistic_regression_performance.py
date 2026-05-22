
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# function
def measure_logistic_regression_performance(X, y, test_size=0.2, n_splits=4, random_state=1234):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, test_size=test_size, random_state=random_state)

    kfold_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(random_state=random_state))
    ])

    param_grid = [
        {   "logistic_regression__penalty": ["l2"],
            "logistic_regression__solver": ["lbfgs", "newton-cg", "sag", "saga"],
            "logistic_regression__C": [ 0.1, 1, 10]},
        {   "logistic_regression__penalty": ["l1"],
            "logistic_regression__solver": [ "saga"],
            "logistic_regression__C": [0.1, 1, 10, 100]},
        {   "logistic_regression__penalty": ["elasticnet"],
            "logistic_regression__solver": ["saga"],
            "logistic_regression__C": [0.1, 1, 10],
            "logistic_regression__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}
    ]

    grid_search = GridSearchCV(model, param_grid=param_grid, cv=kfold_cv)
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)

    y_train_hat = grid_search.predict(X_train)
    print("Train Predition Accuracy Value:", accuracy_score(y_train, y_train_hat))

    y_test_hat = grid_search.predict(X_test)
    print("Test Prediction Accuracy Value:", accuracy_score(y_test, y_test_hat))