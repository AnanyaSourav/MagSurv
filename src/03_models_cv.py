import pandas as pd, numpy as np, json, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
import warnings; warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

X      = np.load("project/results/data/kmer4_features.npy")
cohort = pd.read_csv("project/results/data/cohort_final.csv")

le = LabelEncoder()
y  = le.fit_transform(cohort["planetary_analog"])
class_names = le.classes_

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ── Random Forest
rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                             random_state=SEED, n_jobs=-1)
y_pred_rf = cross_val_predict(rf, X_scaled, y, cv=cv)
ba_rf = balanced_accuracy_score(y, y_pred_rf)
print(f"RF balanced accuracy: {ba_rf:.4f}")
print(classification_report(y, y_pred_rf, target_names=class_names))

# ── MLP
class MLP(nn.Module):
    def __init__(self, in_dim, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,64),     nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, n))
    def forward(self, x): return self.net(x)

def train_mlp(X_tr, y_tr, X_val, cw_arr, epochs=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = MLP(X_tr.shape[1], len(np.unique(y_tr))).to(device)
    w      = torch.tensor(cw_arr, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32).to(device),
                      torch.tensor(y_tr, dtype=torch.long).to(device)),
        batch_size=32, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_val, dtype=torch.float32).to(device)
                      ).argmax(1).cpu().numpy()
    return preds, model

mlp_preds = np.zeros(len(y), dtype=int)
fold_scores = []

for fold, (tr, val) in enumerate(cv.split(X_scaled, y)):
    cw = compute_class_weight("balanced", classes=np.unique(y[tr]), y=y[tr])
    preds, model = train_mlp(X_scaled[tr], y[tr], X_scaled[val], cw)
    mlp_preds[val] = preds
    ba = balanced_accuracy_score(y[val], preds)
    fold_scores.append(ba)
    torch.save(model.state_dict(),
               f"project/results/models/mlp_fold{fold+1}.pt")
    print(f"  Fold {fold+1}: {ba:.4f}")

ba_mlp = balanced_accuracy_score(y, mlp_preds)
print(f"MLP balanced accuracy: {ba_mlp:.4f}")
print(classification_report(y, mlp_preds, target_names=class_names))

# ── Save predictions and metrics
np.save("project/results/data/rf_predictions.npy",  y_pred_rf)
np.save("project/results/data/mlp_predictions.npy", mlp_preds)
np.save("project/results/data/true_labels.npy",      y)
np.save("project/results/data/class_names.npy",      class_names)

metrics = {
    "RF":  {"balanced_accuracy": round(ba_rf,4),
            "report": classification_report(y, y_pred_rf,
                        target_names=class_names, output_dict=True)},
    "MLP": {"balanced_accuracy": round(ba_mlp,4),
            "fold_scores": [round(s,4) for s in fold_scores],
            "report": classification_report(y, mlp_preds,
                        target_names=class_names, output_dict=True)}
}

with open("project/results/data/model_metrics.json","w") as f:
    json.dump(metrics, f, indent=2)
print("Saved: model_metrics.json, predictions, models")
