python - <<'PY'
import glob
import pandas as pd

base = "results/analysis_reports/ppa_eval"
files = glob.glob(base + "/**/shard_*/*_gp_best.csv", recursive=True)
if not files:
raise SystemExit("No shard csv found")

dfs = [pd.read_csv(f) for f in sorted(files)]
df = pd.concat(dfs, ignore_index=True)

key_cols = [c for c in ["seed","benchmark","case","formulation","def_path","def_rank"] if c in df.columns]
if key_cols:
df = df.drop_duplicates(subset=key_cols, keep="last")

out = base + "/merged_3shards_gp_best.csv"
df.to_csv(out, index=False)
print("Merged rows:", len(df))
print("Saved:", out)
PY