import shap
import torch
import numpy as np
from model import DeepSurv
from data_preprocessing import load_and_preprocess

# Load data and model
X, y_time, y_event, features = load_and_preprocess()
model = DeepSurv(X.shape[1])
model.load_state_dict(torch.load("model_fold1.pth"))  # Load fold 1 model
model.eval()

X_test = X[:10]  # Subset for demo

# Define prediction function for SHAP
def predict_fn(x):
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy()

# Compute SHAP values
explainer = shap.KernelExplainer(predict_fn, X[:100])
shap_values = explainer.shap_values(X_test, nsamples=100)

# SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=features)