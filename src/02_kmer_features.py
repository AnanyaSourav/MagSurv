import pandas as pd, numpy as np, os
from itertools import product
from Bio import SeqIO
from tqdm import tqdm

cohort = pd.read_csv("project/results/data/cohort_final.csv")
cohort["Accession"] = cohort["Accession"].astype(str).str.strip()

bases      = ["A","T","G","C"]
kmer_labels = ["".join(p) for p in product(bases, repeat=4)]
kmer_index  = {k: i for i, k in enumerate(kmer_labels)}

def genome_kmer_freq(fasta_path, k=4):
    counts = np.zeros(4**k)
    total  = 0
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = "".join(b for b in str(rec.seq).upper() if b in "ATGC")
        for i in range(len(seq) - k + 1):
            km = seq[i:i+k]
            if km in kmer_index:
                counts[kmer_index[km]] += 1
                total += 1
    if total > 0:
        counts /= total
    return counts

genome_dir = "MTB_genomes"
fa_map = {}
for f in os.listdir(genome_dir):
    if not f.endswith(".fa"): continue
    parts = f.split("_")
    acc = parts[0] + "_" + parts[1]
    fa_map[acc] = f

X, valid_idx = [], []
for i, row in tqdm(cohort.iterrows(), total=len(cohort), desc="k-mer extraction"):
    acc   = row["Accession"]
    fname = fa_map.get(acc)
    fpath = os.path.join(genome_dir, fname) if fname else None
    if fpath and os.path.exists(fpath):
        X.append(genome_kmer_freq(fpath))
        valid_idx.append(i)
    else:
        print(f"  Missing: {acc}")

X = np.array(X)
cohort_valid = cohort.iloc[valid_idx].reset_index(drop=True)

np.save("project/results/data/kmer4_features.npy", X)
np.save("project/results/data/kmer4_labels.npy", np.array(kmer_labels))
cohort_valid.to_csv("project/results/data/cohort_final.csv", index=False)

print(f"Feature matrix: {X.shape}")
print(cohort_valid["planetary_analog"].value_counts())
print("Saved: kmer4_features.npy, kmer4_labels.npy")
