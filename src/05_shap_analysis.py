import pandas as pd, numpy as np, shap
from itertools import product
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

bases       = ["A","T","G","C"]
kmer_labels = ["".join(p) for p in product(bases, repeat=4)]

def gc_frac(k): return (k.count("G") + k.count("C")) / len(k)
kmer_gc = np.array([gc_frac(k) for k in kmer_labels])

# Fit RF on full data for attribution
rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                             random_state=SEED, n_jobs=-1)
rf.fit(X_scaled, y)

explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_scaled) 

if isinstance(shap_values, list):
    sv_arr = np.array(shap_values)
else:
    sv_arr = shap_values.transpose(2, 0, 1) if shap_values.ndim == 3 else shap_values

np.save("project/results/data/shap_values.npy", sv_arr)

# ── Full SHAP summary table
rows = []
for ci, cname in enumerate(class_names):
    sv = np.abs(sv_arr[ci]).mean(axis=0)
    for ki, kmer in enumerate(kmer_labels):
        rows.append({"class": cname, "kmer": kmer,
                     "mean_abs_shap": round(float(sv[ki]),6),
                     "gc_content":    round(kmer_gc[ki],4)})
shap_df = pd.DataFrame(rows)
shap_df.to_csv("project/results/tables/tableS4_shap_full.csv", index=False)

# ── Table 3 — unique k-mers per class
top50_per_class = {}
for ci, cname in enumerate(class_names):
    sv = np.abs(sv_arr[ci]).mean(axis=0)
    top50_per_class[cname] = set(np.argsort(sv)[-50:])

unique_rows = []
for cname in class_names:
    others = set().union(*[top50_per_class[c] for c in class_names if c != cname])
    unique_idx = top50_per_class[cname] - others
    for idx in sorted(unique_idx,
                      key=lambda i: np.abs(sv_arr[list(class_names).index(cname)]).mean(0)[i],
                      reverse=True):
        sv_val = np.abs(sv_arr[list(class_names).index(cname)]).mean(0)[idx]
        unique_rows.append({
            "class":          cname,
            "kmer":           kmer_labels[idx],
            "mean_abs_shap":  round(float(sv_val),6),
            "gc_content":     round(kmer_gc[idx],4)
        })
pd.DataFrame(unique_rows).to_csv(
    "project/results/tables/table3_unique_kmers.csv", index=False)

# ── Overlap table
overlap_rows = []
classes = list(class_names)
for i in range(len(classes)):
    for j in range(i+1, len(classes)):
        n = len(top50_per_class[classes[i]] & top50_per_class[classes[j]])
        overlap_rows.append({"class_A": classes[i], "class_B": classes[j],
                              "shared_top50_kmers": n})
pd.DataFrame(overlap_rows).to_csv(
    "project/results/tables/tableS5_kmer_overlap.csv", index=False)

print("Saved: shap_values.npy, tableS4_shap_full.csv, table3_unique_kmers.csv,",
      "tableS5_kmer_overlap.csv")
