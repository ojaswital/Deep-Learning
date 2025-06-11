import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from model import DeepSurv
from utils import cox_ph_loss, evaluate_concordance
from data_preprocessing import load_and_preprocess

# Dataset wrapper for PyTorch DataLoader
class SurvivalDataset(Dataset):
    def __init__(self, x, time, event):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.time = torch.tensor(time, dtype=torch.float32)
        self.event = torch.tensor(event, dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.time[idx], self.event[idx]

# Load data
X, y_time, y_event, features = load_and_preprocess()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
concordances = []

# For plotting concordance index
fold_ids = []
fold_scores = []

# 5-fold cross-validation
for train_idx, val_idx in kf.split(X):
    fold += 1
    print(f"\nFold {fold}")
    X_train, X_val = X[train_idx], X[val_idx]
    ytime_train, ytime_val = y_time[train_idx], y_time[val_idx]
    yevent_train, yevent_val = y_event[train_idx], y_event[val_idx]

    train_ds = SurvivalDataset(X_train, ytime_train, yevent_train)
    val_ds = SurvivalDataset(X_val, ytime_val, yevent_val)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

    model = DeepSurv(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(100):
        for xb, t, e in train_dl:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = cox_ph_loss(pred, t, e)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # Save model per fold
    torch.save(model.state_dict(), f"model_fold{fold}.pth")

    # Evaluate concordance index
    concordance = evaluate_concordance(model, X_val, ytime_val, yevent_val)
    print(f"Concordance index (Fold {fold}): {concordance:.4f}")
    concordances.append(concordance)
    fold_ids.append(fold)
    fold_scores.append(concordance)

# Average concordance index
mean_concordance = sum(concordances)/len(concordances)
print("\nMean Concordance Index:", mean_concordance)

# Plot concordance index across folds
plt.figure(figsize=(8, 5))
plt.plot(fold_ids, fold_scores, marker='o', label='Fold-wise C-index')
plt.axhline(mean_concordance, color='red', linestyle='--', label='Mean C-index')
plt.xlabel("Fold")
plt.ylabel("Concordance Index")
plt.title("C-index across K-Folds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("c_index_plot.png")
plt.show()
