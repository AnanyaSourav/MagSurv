import pandas as pd, os

qual = pd.read_csv("qualityMetrics.csv");      qual.columns = qual.columns.str.strip()
env  = pd.read_csv("environmentalMetadata.csv"); env.columns  = env.columns.str.strip()
cls  = pd.read_csv("classification.csv");       cls.columns  = cls.columns.str.strip()
sub  = pd.read_csv("submissionMetadata.csv");   sub.columns  = sub.columns.str.strip()
ref  = pd.read_csv("referenceMetadata.csv");    ref.columns  = ref.columns.str.strip()

# ── QC filter
filtered = qual[
    (qual["CheckM2 completeness(%)(NN_model)"] >= 70) &
    (qual["CheckM2 contamination(%)"]          <= 10) &
    (qual["pass.GUNC"]                         == True)
].copy()
print(f"Genomes passing QC: {len(filtered)} / {len(qual)}")

# ── Merge metadata
merged = (filtered
    .merge(env, on="Accession")
    .merge(cls, on="Accession")
    .merge(sub[["Accession","Assembly level","Sequencing platform"]], on="Accession", how="left")
)

# ── Planetary analog mapping
planet_map = {
    "freshwater":                                     "Mars_lake_analog",
    "brackish":                                       "Mars_lake_analog",
    "bog/peatland (freshwater)":                      "Mars_lake_analog",
    "bog/peatland (brackish)":                        "Mars_lake_analog",
    "marine":                                         "Europa_ocean_analog",
    "hypersaline":                                    "Europa_ocean_analog",
    "marine oxygen minimum zone (OMZ)":               "Europa_ocean_analog",
    "bog/peatland (acidic)":                          "Mars_acidic_analog",
    "subsurface/deep subsurface":                      "Enceladus_subsurface_analog",
    "thermal environment (hot spring water)":          "Enceladus_subsurface_analog",
    "thermal environment (hydrothermal vent chimney)":"Enceladus_subsurface_analog",
}
merged["planetary_analog"] = merged["Environment"].map(planet_map)
print(merged["planetary_analog"].value_counts())

# ── Drop Enceladus (n=5, underpowered)
cohort = merged[merged["planetary_analog"] != "Enceladus_subsurface_analog"].reset_index(drop=True)
print(f"Final cohort: {len(cohort)} genomes, 3 classes")

cohort.to_csv("project/results/data/cohort_final.csv", index=False)

# ── Table S1 — full cohort metadata
cols = ["Accession","ID_x","Organism Name","planetary_analog","Environment",
        "Geographic location","latitude","longitude",
        "Assembly level","Sequencing platform",
        "CheckM2 completeness(%)(NN_model)","CheckM2 contamination(%)","pass.GUNC",
        "Number_of_contigs","Size (bp)","GC (%)","N50",
        "GTDB_r220_classification"]
cols_present = [c for c in cols if c in cohort.columns]
cohort[cols_present].to_csv("project/results/tables/tableS1_cohort_metadata.csv", index=False)
print("Saved: tableS1_cohort_metadata.csv")

# ── Table S2 — QC summary per class
gc_col   = [c for c in cohort.columns if "GC" in c][0]
comp_col = [c for c in cohort.columns if "CheckM2" in c and "NN" in c][0]
cont_col = [c for c in cohort.columns if "contamination" in c and "CheckM2" in c][0]
n50_col  = [c for c in cohort.columns if "N50" in c][0]

qc_summary = cohort.groupby("planetary_analog").agg(
    n=("Accession","count"),
    gc_mean=(gc_col,"mean"),   gc_std=(gc_col,"std"),
    gc_min=(gc_col,"min"),     gc_max=(gc_col,"max"),
    completeness_mean=(comp_col,"mean"), completeness_std=(comp_col,"std"),
    contamination_mean=(cont_col,"mean"),
    n50_median=(n50_col,"median"),
).round(2)
qc_summary.to_csv("project/results/tables/tableS2_qc_summary.csv")
print("Saved: tableS2_qc_summary.csv")
