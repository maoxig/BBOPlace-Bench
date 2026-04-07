#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash moscripts/merge_ppa_results.sh \
#     --seed-group seeds_1_2_3 \
#     --hv-json results/analysis_reports/hv/json/hv_summary_seeds_1_2_3.json
#
#   bash moscripts/merge_ppa_results.sh \
#     --seed-group seed_1 \
#     --hv-json results/analysis_reports/hv/json/hv_summary_seed_1.json

SEED_GROUP="seeds_1_2_3"
BASE="results/analysis_reports/ppa_eval"
HV_JSON=""
BENCHMARKS="OpenROAD"
FORMULATIONS="MGO,HPO"
ONLY_EVAL_OK="true"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--seed-group)
			SEED_GROUP="$2"; shift 2 ;;
		--base)
			BASE="$2"; shift 2 ;;
		--hv-json)
			HV_JSON="$2"; shift 2 ;;
		--benchmarks)
			BENCHMARKS="$2"; shift 2 ;;
		--formulations)
			FORMULATIONS="$2"; shift 2 ;;
		--only-eval-ok)
			ONLY_EVAL_OK="$2"; shift 2 ;;
		*)
			echo "Unknown arg: $1" >&2
			exit 2 ;;
	esac
done

TARGET_DIR="${BASE}/${SEED_GROUP}"
if [[ ! -d "${TARGET_DIR}" ]]; then
	echo "Target dir not found: ${TARGET_DIR}" >&2
	exit 1
fi

python - "${TARGET_DIR}" "${SEED_GROUP}" <<'PY'
import json
import os
import sys
import pandas as pd

target_dir = sys.argv[1]
seed_group = sys.argv[2]

KEY_CANDIDATES = ["seed", "benchmark", "case", "formulation", "def_path", "def_rank"]


def collect_files(root, names):
		found = []
		for cur, _, files in os.walk(root):
				for fn in files:
						if fn in names and "shard_" in cur:
								found.append(os.path.join(cur, fn))
		return sorted(found)


def dedup_rows_df(df):
		key_cols = [c for c in KEY_CANDIDATES if c in df.columns]
		if key_cols:
				return df.drop_duplicates(subset=key_cols, keep="last")
		return df


def dedup_rows_list(rows):
		latest = {}
		passthrough = []
		for i, r in enumerate(rows):
				if not isinstance(r, dict):
						continue
				keys = [k for k in KEY_CANDIDATES if k in r]
				if not keys:
						passthrough.append((i, r))
						continue
				key = tuple(r.get(k) for k in keys)
				latest[key] = (i, r)
		out = [x[1] for x in sorted(latest.values(), key=lambda t: t[0])]
		out.extend([x[1] for x in passthrough])
		return out


base_name = f"ppa_eval_{seed_group}_gp_best"
csv_names = [f"{base_name}.csv", f"{base_name}.partial.csv"]
json_names = [f"{base_name}.json", f"{base_name}.partial.json"]

csv_files = collect_files(target_dir, csv_names)
json_files = collect_files(target_dir, json_names)

if not csv_files and not json_files:
		raise SystemExit(f"No shard csv/json found under {target_dir}")

merged_rows = []
if csv_files:
		frames = []
		for p in csv_files:
				df = pd.read_csv(p)
				df["_source_file"] = p
				frames.append(df)
		merged_df = dedup_rows_df(pd.concat(frames, ignore_index=True))
		merged_df = merged_df.drop(columns=["_source_file"], errors="ignore")
		merged_rows = merged_df.to_dict(orient="records")
		out_csv = os.path.join(target_dir, f"{base_name}.csv")
		merged_df.to_csv(out_csv, index=False)
		print("Saved merged CSV:", out_csv)
		print("Merged CSV rows:", len(merged_df))
else:
		out_csv = None

if json_files:
		all_rows = []
		meta = {}
		for p in json_files:
				with open(p, "r") as f:
						data = json.load(f)
				if isinstance(data, dict):
						if not meta:
								meta = {k: v for k, v in data.items() if k != "rows"}
						all_rows.extend(data.get("rows", []))
		dedup = dedup_rows_list(all_rows)
		payload = dict(meta) if meta else {}
		payload["n_rows"] = len(dedup)
		payload["rows"] = dedup
		payload["merged_from"] = json_files
		out_json = os.path.join(target_dir, f"{base_name}.json")
		with open(out_json, "w") as f:
				json.dump(payload, f, indent=2)
		print("Saved merged JSON:", out_json)
		print("Merged JSON rows:", len(dedup))
else:
		out_json = None

if out_csv is None and merged_rows:
		out_csv = os.path.join(target_dir, "merged_gp_best.csv")
		pd.DataFrame(merged_rows).to_csv(out_csv, index=False)
		print("Saved merged CSV (from JSON rows):", out_csv)

print("MERGED_CSV=", out_csv or "")
print("MERGED_JSON=", out_json or "")
PY

if [[ -z "${HV_JSON}" ]]; then
	if [[ "${SEED_GROUP}" == seeds_* ]]; then
		HV_JSON="results/analysis_reports/hv/json/hv_summary_${SEED_GROUP}.json"
	elif [[ "${SEED_GROUP}" == seed_* ]]; then
		HV_JSON="results/analysis_reports/hv/json/hv_summary_${SEED_GROUP}.json"
	fi
fi

MERGED_CSV=""
if [[ "${SEED_GROUP}" == seeds_* ]]; then
	MERGED_CSV="${TARGET_DIR}/ppa_eval_${SEED_GROUP}_gp_best.csv"
else
	MERGED_CSV="${TARGET_DIR}/ppa_eval_${SEED_GROUP}_gp_best.csv"
fi

OUT_DIR="results/analysis_reports/ppa_proxy/study_${SEED_GROUP}"
ANALYZE_CMD="python3 src/utils/analyze_ppa_proxy.py --ppa_csv ${MERGED_CSV} --hv_json ${HV_JSON} --output_dir ${OUT_DIR} --benchmarks ${BENCHMARKS} --formulations ${FORMULATIONS}"
if [[ "${ONLY_EVAL_OK}" == "true" ]]; then
	ANALYZE_CMD="${ANALYZE_CMD} --only_eval_ok"
fi

echo ""
echo "Suggested analysis command:"
echo "${ANALYZE_CMD}"