import pandas as pd, numpy as np, json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, classification_report
import warnings; warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

X      = np.load("project/results/data/kmer4_features.npy")
cohort = pd.read_csv("project/results/data/cohort_final.csv")

le = LabelEncoder()
y  = le.fit_transform(cohort["planetary_analog"])
class_names = le.classes_

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def get_phylum(s):
    for p in str(s).split(";"):
        if p.strip().startswith("p__"):
            return p.strip().replace("p__","")
    return "Unknown"

cohort["phylum"] = cohort["GTDB_r220_classification"].apply(get_phylum)
phylum_dummies   = pd.get_dummies(cohort["phylum"], prefix="phy").values.astype(float)
X_with_phy       = np.hstack([X_scaled, phylum_dummies])

le_phy = LabelEncoder()
y_phy  = le_phy.fit_transform(cohort["phylum"])

rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                             random_state=SEED, n_jobs=-1)

# Test 1 — phylum prediction from k-mers
y_pred_phy = cross_val_predict(rf, X_scaled, y_phy, cv=cv)
t1 = balanced_accuracy_score(y_phy, y_pred_phy)

# Test 2 — within Pseudomonadota
mask   = cohort["phylum"] == "Pseudomonadota"
y_p    = y[mask]; X_p = X_scaled[mask]
y_pred_p = cross_val_predict(rf, X_p, y_p,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED))
t2 = balanced_accuracy_score(y_p, y_pred_p)

# Test 3 — k-mers only vs k-mers + phylum
y_pred_km  = cross_val_predict(rf, X_scaled,   y, cv=cv)
y_pred_kmp = cross_val_predict(rf, X_with_phy, y, cv=cv)
t3_km  = balanced_accuracy_score(y, y_pred_km)
t3_kmp = balanced_accuracy_score(y, y_pred_kmp)

# Test 4 — phylum only
y_pred_phonly = cross_val_predict(rf, phylum_dummies, y, cv=cv)
t4 = balanced_accuracy_score(y, y_pred_phonly)

results = {
    "T1_kmer_predicts_phylum":          round(t1,4),
    "T2_within_Pseudomonadota":         round(t2,4),
    "T3_kmer_only":                     round(t3_km,4),
    "T3_kmer_plus_phylum":              round(t3_kmp,4),
    "T3_delta":                         round(t3_kmp - t3_km,4),
    "T4_phylum_only_null":              round(t4,4),
}
print(json.dumps(results, indent=2))

# Table 1 — main results table for paper
import json as _json
model_metrics = _json.load(open("project/results/data/model_metrics.json"))

rows = []
for model, d in model_metrics.items():
    rpt = d["report"]
    for cls in list(class_names) + ["macro avg"]:
        if cls in rpt:
            rows.append({
                "Model": model,
                "Class": cls,
                "Precision": round(rpt[cls]["precision"],3),
                "Recall":    round(rpt[cls]["recall"],3),
                "F1":        round(rpt[cls]["f1-score"],3),
                "Support":   int(rpt[cls]["support"]),
                "Balanced_Accuracy": d["balanced_accuracy"]
            })
pd.DataFrame(rows).to_csv("project/results/tables/table1_classification_metrics.csv",
                           index=False)

# Table 2 — phylogenetic confound tests
pd.DataFrame([results]).T.rename(columns={0:"value"}).to_csv(
    "project/results/tables/table2_phylo_confound_tests.csv")

# Table S3 — phylum composition per class
pivot = (cohort.groupby(["planetary_analog","phylum"])
               .size().reset_index(name="n"))
pivot["pct"] = pivot.groupby("planetary_analog")["n"].transform(
    lambda x: (x / x.sum() * 100).round(1))
pivot.to_csv("project/results/tables/tableS3_phylum_composition.csv", index=False)

with open("project/results/data/phylo_confound.json","w") as f:
    json.dump(results, f, indent=2)
print("Saved: table1, table2, tableS3, phylo_confound.json")
