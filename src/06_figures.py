import pandas as pd, numpy as np, matplotlib.pyplot as plt, json, torch, torch.nn as nn
import matplotlib
from itertools import product
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import warnings; warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight"
})

FIG = "project/results/figures"

# ── Load
X      = np.load("project/results/data/kmer4_features.npy")
cohort = pd.read_csv("project/results/data/cohort_final.csv")
cohort.columns = cohort.columns.str.strip()
cohort["Accession"] = cohort["Accession"].astype(str).str.strip()

sv_arr      = np.load("project/results/data/shap_values.npy")
y_pred_rf   = np.load("project/results/data/rf_predictions.npy")
y_pred_mlp  = np.load("project/results/data/mlp_predictions.npy")
y_true      = np.load("project/results/data/true_labels.npy")
class_names = np.load("project/results/data/class_names.npy", allow_pickle=True)

le = LabelEncoder()
y  = le.fit_transform(cohort["planetary_analog"])

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

bases       = ["A","T","G","C"]
kmer_labels = ["".join(p) for p in product(bases, repeat=4)]
def gc_frac(k): return (k.count("G") + k.count("C")) / len(k)
kmer_gc = np.array([gc_frac(k) for k in kmer_labels])

COLORS = {"Europa_ocean_analog":"#2196F3",
          "Mars_acidic_analog": "#FF5722",
          "Mars_lake_analog":   "#4CAF50"}
SHORT  = {"Europa_ocean_analog":"Europa ocean",
          "Mars_acidic_analog": "Mars acidic",
          "Mars_lake_analog":   "Mars lake"}

# ════════════════════════════════════════════════════════════════
# FIG 1 — PCA
# ════════════════════════════════════════════════════════════════
pca   = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(6,5))
for cn in class_names:
    mask = cohort["planetary_analog"] == cn
    ax.scatter(X_pca[mask,0], X_pca[mask,1],
               c=COLORS[cn], label=SHORT[cn], alpha=0.7, s=35, linewidths=0)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
ax.set_title("PCA of 4-mer composition profiles")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{FIG}/fig1_pca.png"); plt.close(); print("fig1_pca.png")

# ════════════════════════════════════════════════════════════════
# FIG 2 — Model comparison
# ════════════════════════════════════════════════════════════════
with open("project/results/data/phylo_confound.json") as f:
    pc = json.load(f)
with open("project/results/data/model_metrics.json") as f:
    mm = json.load(f)

labels = ["Phylum-only\n(null)", "Random Forest\n(k-mers)", "MLP\n(k-mers)"]
vals   = [pc["T4_phylum_only_null"],
          mm["RF"]["balanced_accuracy"],
          mm["MLP"]["balanced_accuracy"]]
bcolors = ["#9E9E9E","#FF9800","#2196F3"]

fig, ax = plt.subplots(figsize=(5.5,4))
bars = ax.bar(labels, vals, color=bcolors, width=0.5, zorder=2)
ax.axhline(1/3, color="black", linestyle="--", lw=0.9, label="Random chance (0.333)")
ax.set_ylim(0,1); ax.set_ylabel("Balanced accuracy (5-fold CV)")
ax.set_title("Model performance comparison"); ax.legend(frameon=False, fontsize=9)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
for bar,val in zip(bars,vals):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f"{val:.3f}",
            ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG}/fig2_model_comparison.png"); plt.close(); print("fig2_model_comparison.png")

# ════════════════════════════════════════════════════════════════
# FIG 3 — Confusion matrices (RF + MLP side by side)
# ════════════════════════════════════════════════════════════════
short_names = [SHORT[c] for c in class_names]
fig, axes = plt.subplots(1, 2, figsize=(11,4.5))
for ax, preds, title, ba in zip(
        axes,
        [y_pred_rf, y_pred_mlp],
        ["Random Forest", "MLP"],
        [mm["RF"]["balanced_accuracy"], mm["MLP"]["balanced_accuracy"]]):
    cm = confusion_matrix(y_true, preds)
    ConfusionMatrixDisplay(cm, display_labels=short_names).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{title} — 5-fold CV\n(bal. acc = {ba:.3f})")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
plt.tight_layout()
plt.savefig(f"{FIG}/fig3_confusion_matrices.png"); plt.close()
print("fig3_confusion_matrices.png")

# ════════════════════════════════════════════════════════════════
# FIG 4 — SHAP top-15 k-mers, GC-coloured
# ════════════════════════════════════════════════════════════════
cmap_shap = plt.cm.get_cmap("RdYlGn") if hasattr(plt.cm, "get_cmap") else matplotlib.colormaps["RdYlGn"]
fig, axes = plt.subplots(1,3, figsize=(14,5.5), sharey=False)
for ci, (cn, ax) in enumerate(zip(class_names, axes)):
    sv   = np.abs(sv_arr[ci]).mean(axis=0)
    idx  = np.argsort(sv)[-15:]
    kmers= [kmer_labels[i] for i in idx]
    vals = sv[idx]
    gc   = kmer_gc[idx]
    ax.barh(kmers, vals, color=[cmap_shap(g) for g in gc])
    ax.set_title(SHORT[cn], fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    if ci == 0: ax.set_ylabel("4-mer")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
sm = plt.cm.ScalarMappable(cmap=cmap_shap, norm=plt.Normalize(0,1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, shrink=0.55, pad=0.02)
cbar.set_label("k-mer GC content")
cbar.set_ticks([0,.25,.5,.75,1])
cbar.set_ticklabels(["0%","25%","50%","75%","100%"])
plt.suptitle("Top 15 discriminative 4-mers per planetary analog class", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG}/fig4_shap_kmers.png"); plt.close(); print("fig4_shap_kmers.png")

# ════════════════════════════════════════════════════════════════
# FIG 5 — GC violin
# ════════════════════════════════════════════════════════════════
gc_col = [c for c in cohort.columns if "GC" in c][0]

fig, ax = plt.subplots(figsize=(5,4))
data = [cohort[cohort["planetary_analog"]==c][gc_col].values for c in class_names]
vp   = ax.violinplot(data, positions=range(len(class_names)),
                      showmedians=True, showextrema=True)
for pc, cn in zip(vp["bodies"], class_names):
    pc.set_facecolor(COLORS[cn]); pc.set_alpha(0.7)
for key in ["cmedians","cbars","cmins","cmaxes"]:
    vp[key].set_color("black")
ax.set_xticks(range(len(class_names)))
ax.set_xticklabels([SHORT[c] for c in class_names])
ax.set_ylabel("Genome GC content (%)")
ax.set_title("Genome GC% by planetary analog class")
plt.tight_layout()
plt.savefig(f"{FIG}/fig5_gc_violin.png"); plt.close(); print("fig5_gc_violin.png")

# ════════════════════════════════════════════════════════════════
# FIG 6 — Phylum stacked bar
# ════════════════════════════════════════════════════════════════
def get_phylum(s):
    for p in str(s).split(";"):
        if p.strip().startswith("p__"): return p.strip().replace("p__","")
    return "Unknown"

cohort["phylum"] = cohort["GTDB_r220_classification"].apply(get_phylum)
top6 = cohort["phylum"].value_counts().head(6).index.tolist()
cohort["phylum_plot"] = cohort["phylum"].apply(lambda p: p if p in top6 else "Other")

cmap6 = matplotlib.colormaps["tab10"].resampled(7) if hasattr(matplotlib, "colormaps") else plt.cm.get_cmap("tab10", 7)
phy_colors = {p: cmap6(i) for i,p in enumerate(top6)}
phy_colors["Other"] = "#9E9E9E"

pivot     = cohort.groupby(["planetary_analog","phylum_plot"]).size().unstack(fill_value=0)
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)*100

fig, ax = plt.subplots(figsize=(6.5,4.5))
bottom  = np.zeros(len(pivot_pct))
for phy in pivot_pct.columns:
    ax.bar([SHORT[c] for c in pivot_pct.index], pivot_pct[phy].values,
           bottom=bottom, color=phy_colors.get(phy,"#9E9E9E"), label=phy, width=0.5)
    bottom += pivot_pct[phy].values
ax.set_ylabel("Proportion (%)"); ax.set_title("Phylum composition per class")
ax.legend(bbox_to_anchor=(1.01,1), loc="upper left", frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG}/fig6_phylum_composition.png"); plt.close()
print("fig6_phylum_composition.png")

# ════════════════════════════════════════════════════════════════
# FIG S1 — Per-fold MLP balanced accuracy
# ════════════════════════════════════════════════════════════════
fold_scores = mm["MLP"]["fold_scores"]
fig, ax = plt.subplots(figsize=(5,3.5))
ax.bar(range(1,6), fold_scores, color="#2196F3", alpha=0.8)
ax.axhline(np.mean(fold_scores), color="black", linestyle="--", lw=1,
           label=f"Mean = {np.mean(fold_scores):.3f}")
ax.set_xlabel("Fold"); ax.set_ylabel("Balanced accuracy")
ax.set_title("MLP per-fold performance")
ax.set_xticks(range(1,6)); ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{FIG}/figS1_mlp_folds.png"); plt.close(); print("figS1_mlp_folds.png")

# ════════════════════════════════════════════════════════════════
# FIG S2 — k-mer GC distribution (all 256 k-mers, shaded by class rank)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1,3, figsize=(14,4), sharey=True)
for ci, (cn, ax) in enumerate(zip(class_names, axes)):
    sv_mean = np.abs(sv_arr[ci]).mean(axis=0)
    rank    = sv_mean.argsort().argsort()   # rank 0=lowest
    sc = ax.scatter(kmer_gc, sv_mean,
                    c=rank, cmap="viridis", s=18, alpha=0.7)
    ax.set_xlabel("k-mer GC content")
    if ci == 0: ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(SHORT[cn])
    plt.colorbar(sc, ax=ax, label="SHAP rank")
plt.suptitle("SHAP value vs k-mer GC content (all 256 4-mers)", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG}/figS2_shap_vs_gc.png"); plt.close(); print("figS2_shap_vs_gc.png")

print("\nAll figures saved to", FIG)
