import pandas as pd
import numpy as np
from lifelines.datasets import load_rossi
from sklearn.preprocessing import StandardScaler

# Load and preprocess SUPPORT dataset
def load_and_preprocess():
    data = load_rossi().dropna()
    features = ['age', 'prio', 'fin', 'race', 'wexp', 'mar']
    X = data[features].copy()
    y_time = data['week'].values
    y_event = data['arrest'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_time, y_event, features