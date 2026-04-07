import argparse
import json
import os
import pickle
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OPENROAD_PPA_METRICS = ["GRT_WL", "DRT_WL", "WNS", "TNS", "DRC", "Power", "Area", ]
ICCAD_PPA_METRICS = ["WNS", "TNS"]
MAXIMIZE_METRICS = ["WNS", "TNS", "n_tns", "n_wns"]
MINIMIZE_METRICS = ["GRT_WL", "DRT_WL", "DRC", "Power", "Area", "hpwl"]
VALID_MODES = ["GP", "MP"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def configure_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_filters(csv_text):
    if csv_text is None or csv_text.strip() == "" or csv_text.strip().lower() == "all":
        return None
    return [x.strip() for x in csv_text.split(",") if x.strip()]


def parse_modes(mode_text):
    if mode_text is None or str(mode_text).strip() == "":
        return ["GP"]
    if str(mode_text).strip().lower() == "all":
        return list(VALID_MODES)
    items = [x.strip().upper() for x in str(mode_text).split(",") if x.strip()]
    if not items:
        return ["GP"]
    out = []
    for m in items:
        if m not in VALID_MODES:
            raise ValueError(f"Unsupported mode '{m}', valid options: {VALID_MODES}")
        if m not in out:
            out.append(m)
    return out


def normalize_proxy_metric_name(name):
    x = str(name).strip()
    lx = x.lower()
    if lx == "rudy2":
        return "rudy"
    if lx == "mp_rudy2":
        return "mp_rudy"
    return x


def parse_seeds(seed_text):
    items = [x.strip() for x in str(seed_text).split(",") if x.strip()]
    if not items:
        raise ValueError("--seeds is empty")
    return [int(x) for x in items]


def parse_csv_list(csv_text):
    if csv_text is None or str(csv_text).strip() == "":
        return []
    return [x.strip() for x in str(csv_text).split(",") if x.strip()]


def infer_seed_from_path(path_text):
    m = re.search(r"seed_(\d+)", str(path_text))
    if m:
        return int(m.group(1))
    return None


def load_ppa_frames(ppa_csv, ppa_csvs):
    paths = []
    if ppa_csv:
        paths.append(ppa_csv)
    paths.extend(ppa_csvs)

    # Keep order and deduplicate.
    uniq_paths = []
    seen = set()
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq_paths.append(ap)

    if not uniq_paths:
        raise ValueError("No PPA CSV provided. Use --ppa_csv or --ppa_csvs.")

    frames = []
    for p in uniq_paths:
        df = pd.read_csv(p)
        if "seed" not in df.columns:
            seed_guess = infer_seed_from_path(p)
            if seed_guess is not None:
                df["seed"] = int(seed_guess)
        df["source_csv"] = p
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def infer_metric_columns(df):
    candidate_cols = [
        "GRT_WL",
        "DRT_WL",
        "WNS",
        "TNS",
        "DRC",
        "Power",
        "Area",
        "n_tns",
        "n_wns",
        "hpwl",
    ]
    return [c for c in candidate_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def harmonize_openroad_columns(df):
    out = df.copy()
    # Prefer reporting CellArea as "Area" while keeping original columns intact.
    if "Area" not in out.columns and "StdCellArea" in out.columns:
        out["Area"] = pd.to_numeric(out["StdCellArea"], errors="coerce")
    elif "Area" in out.columns:
        out["Area"] = pd.to_numeric(out["Area"], errors="coerce")

    if "DRC" in out.columns:
        out["DRC"] = pd.to_numeric(out["DRC"], errors="coerce")
    return out


def _pick_existing_columns(df, preferred):
    lower_map = {c.lower(): c for c in df.columns}
    selected = []
    for p in preferred:
        key = p.lower()
        if key in lower_map:
            selected.append(lower_map[key])
    return selected


def choose_ppa_metrics(df):
    benches = sorted(df["benchmark"].dropna().astype(str).unique()) if "benchmark" in df.columns else []

    if benches == ["ICCAD2015"]:
        metrics = _pick_existing_columns(df, ICCAD_PPA_METRICS)
    elif benches == ["OpenROAD"]:
        metrics = _pick_existing_columns(df, OPENROAD_PPA_METRICS)
    else:
        # Overall analysis: keep OpenROAD and ICCAD relevant PPA metrics, avoid runtime-like metrics.
        metrics = _pick_existing_columns(df, OPENROAD_PPA_METRICS + ICCAD_PPA_METRICS)

    if not metrics:
        metrics = infer_metric_columns(df)

    # Keep order and de-duplicate.
    out = []
    seen = set()
    for m in metrics:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def normalize_metric_direction(df, metrics):
    """
    Convert maximize-style metrics into minimize-style proxies for consistent analysis.
    For readability, WNS/TNS are transformed to -WNS/-TNS.
    """
    out = df.copy()
    norm_metrics = []
    metric_labels = {}

    for m in metrics:
        if str(m).upper() in {"WNS", "TNS"}:
            col = f"NEG_{m.upper()}"
            out[col] = -pd.to_numeric(out[m], errors="coerce")
            norm_metrics.append(col)
            metric_labels[col] = f"-{m.upper()}"
        else:
            norm_metrics.append(m)
            metric_labels[m] = m

    # Keep a deterministic and readable order for plots/tables.
    preferred_order = [
        "GRT_WL",
        "DRT_WL",
        "NEG_WNS",
        "NEG_TNS",
        "DRC",
        "Power",
        "Area",
        "n_tns",
        "n_wns",
        "hpwl",
    ]
    rank = {k: i for i, k in enumerate(preferred_order)}
    norm_metrics = sorted(norm_metrics, key=lambda x: rank.get(x, 10_000))

    return out, norm_metrics, metric_labels


def metric_label(metric, metric_labels=None):
    if metric_labels and metric in metric_labels:
        return metric_labels[metric]
    return str(metric)


def save_df(df, out_csv, out_tex):
    df.to_csv(out_csv, index=False)
    try:
        df.to_latex(out_tex, index=False, float_format=lambda x: f"{x:.4g}")
    except Exception:
        # Minimal fallback writer to keep LaTeX artifact available.
        with open(out_tex, "w") as f:
            f.write("\\begin{tabular}{" + "l" * len(df.columns) + "}\n")
            f.write(" \\hline\n")
            f.write(" & ".join([str(c) for c in df.columns]) + " \\\\ \n")
            f.write(" \\hline\n")
            for _, r in df.iterrows():
                vals = []
                for v in r.tolist():
                    if isinstance(v, float):
                        if np.isfinite(v):
                            vals.append(f"{v:.6g}")
                        else:
                            vals.append("nan")
                    else:
                        vals.append(str(v))
                f.write(" & ".join(vals) + " \\\\ \n")
            f.write(" \\hline\n")
            f.write("\\end{tabular}\n")


def save_dual(fig, out_dir, stem):
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), bbox_inches="tight")
    plt.close(fig)


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip()).strip("_").lower()


def safe_corr(a, b, method):
    sub = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(sub) < 3:
        return np.nan
    if sub["a"].nunique() <= 1 or sub["b"].nunique() <= 1:
        return np.nan
    return float(sub["a"].corr(sub["b"], method=method))


def is_metric_minimize(metric_name):
    m = str(metric_name)
    if m in MAXIMIZE_METRICS:
        return False
    return True


def bootstrap_ci_mean(values, n_boot=3000, alpha=0.05, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = values[idx].mean(axis=1)
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def paired_bootstrap_diff(a, b, n_boot=5000, alpha=0.05, seed=0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if a.size == 0:
        return np.nan, np.nan, np.nan
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = d[idx].mean(axis=1)
    mean_d = float(np.mean(d))
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return mean_d, lo, hi


def load_hv_table_all_modes(hv_json_path):
    with open(hv_json_path, "r") as f:
        data = json.load(f)

    rows = []
    for bench, bench_data in data.get("benchmarks", {}).items():
        for mode, mode_data in bench_data.get("modes", {}).items():
            for case, case_data in mode_data.get("cases", {}).items():
                for form, form_data in case_data.items():
                    if "best_run_path" in form_data:
                        rows.append(
                            {
                                "benchmark": bench,
                                "mode": mode,
                                "case": case,
                                "formulation": form,
                                "seed": data.get("seed", np.nan),
                                "best_hv": form_data.get("best_hv", np.nan),
                                "best_hv_mean": form_data.get("best_hv", np.nan),
                                "best_algo": form_data.get("best_algo", ""),
                                "run_path": form_data.get("best_run_path", ""),
                            }
                        )
                        continue

                    best_algo = form_data.get("best_algo_by_mean")
                    best_hv_mean = form_data.get("best_hv_mean", np.nan)
                    algo_data = form_data.get("algorithms", {}).get(best_algo, {}) if best_algo else {}
                    hv_by_seed = algo_data.get("hv_by_seed", {}) if isinstance(algo_data, dict) else {}
                    run_by_seed = algo_data.get("run_path_by_seed", {}) if isinstance(algo_data, dict) else {}

                    for seed_key, run_path in run_by_seed.items():
                        try:
                            seed_val = int(seed_key)
                        except Exception:
                            seed_val = np.nan
                        rows.append(
                            {
                                "benchmark": bench,
                                "mode": mode,
                                "case": case,
                                "formulation": form,
                                "seed": seed_val,
                                "best_hv": hv_by_seed.get(seed_key, np.nan),
                                "best_hv_mean": best_hv_mean,
                                "best_algo": best_algo,
                                "run_path": run_path,
                            }
                        )
    return pd.DataFrame(rows)


def read_metrics_current_labels(run_path):
    metrics_path = os.path.join(run_path, "metrics.csv")
    if not os.path.exists(metrics_path):
        return []
    try:
        mdf = pd.read_csv(metrics_path, nrows=1)
        labels = [normalize_proxy_metric_name(c[len("current_") :]) for c in mdf.columns if c.startswith("current_")]
        return labels
    except Exception:
        return []


def read_final_solutions_y(run_path):
    ckpt = os.path.join(run_path, "checkpoint")
    final_path = os.path.join(ckpt, "final_solutions.pkl")
    elite_path = os.path.join(ckpt, "elite_pool.pkl")

    sols = None
    if os.path.exists(final_path):
        try:
            sols = pickle.load(open(final_path, "rb"))
        except Exception:
            sols = None
    elif os.path.exists(elite_path):
        try:
            sols = pickle.load(open(elite_path, "rb"))
        except Exception:
            sols = None

    if not sols:
        return np.empty((0, 0), dtype=float)

    ys = []
    for s in sols:
        if isinstance(s, dict) and s.get("Y") is not None:
            y = np.asarray(s["Y"], dtype=float).reshape(-1)
            ys.append(y)
    if not ys:
        return np.empty((0, 0), dtype=float)
    return np.vstack(ys)


def attach_proxy_vectors_by_defrank(df):
    out = df.copy()
    run_cache = {}

    for rp in out["run_path"].dropna().unique():
        labels = read_metrics_current_labels(rp)
        ymat = read_final_solutions_y(rp)
        run_cache[rp] = {"labels": labels, "ymat": ymat}

    max_dim = 0
    for rp, info in run_cache.items():
        ymat = info["ymat"]
        if ymat.size > 0:
            max_dim = max(max_dim, ymat.shape[1])

    for i in range(max_dim):
        out[f"proxy_obj_{i+1}"] = np.nan

    out["proxy_obj_names"] = ""
    out["proxy_dim"] = 0
    out["proxy_map_ok"] = False

    for idx, row in out.iterrows():
        rp = row.get("run_path", "")
        info = run_cache.get(rp)
        if info is None:
            continue

        ymat = info["ymat"]
        labels = info["labels"]
        out.at[idx, "proxy_obj_names"] = "|".join(labels)

        if ymat.size == 0:
            continue

        try:
            rank = int(row["def_rank"]) - 1
        except Exception:
            continue

        if rank < 0 or rank >= ymat.shape[0]:
            continue

        y = ymat[rank]
        out.at[idx, "proxy_dim"] = int(len(y))
        out.at[idx, "proxy_map_ok"] = True
        for i, v in enumerate(y):
            out.at[idx, f"proxy_obj_{i+1}"] = float(v)

    # If objective names are stable, add semantic proxy columns.
    name_map = {}
    for rp, info in run_cache.items():
        labels = info["labels"]
        if labels:
            for i, lb in enumerate(labels):
                if i not in name_map:
                    name_map[i] = lb

    for i, lb in name_map.items():
        src = f"proxy_obj_{i+1}"
        dst = f"proxy_{sanitize_name(lb)}"
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    return out


def build_proxy_long_df(df, ppa_metrics):
    rows = []
    for _, row in df.iterrows():
        names_raw = str(row.get("proxy_obj_names", ""))
        names = [x.strip() for x in names_raw.split("|") if x.strip()] if names_raw and names_raw != "nan" else []
        mode = str(row.get("mode", "")).strip() or "unknown"

        for i in range(1, 32):
            col = f"proxy_obj_{i}"
            if col not in df.columns:
                break
            v = row.get(col, np.nan)
            if not np.isfinite(v):
                continue
            pname = names[i - 1] if i - 1 < len(names) else col
            pretty_name = f"{mode}:{pname}"

            item = {
                "row_uid": row.get("row_uid"),
                "benchmark": row.get("benchmark"),
                "case": row.get("case"),
                "formulation": row.get("formulation"),
                "mode": row.get("mode"),
                "run_path": row.get("run_path"),
                "def_rank": row.get("def_rank"),
                "proxy_name": pretty_name,
                "proxy_value": float(v),
            }
            for m in ppa_metrics:
                item[m] = row.get(m, np.nan)
            rows.append(item)

    return pd.DataFrame(rows)


def build_proxy_catalog_from_hv(hv_df):
    rows = []
    for _, row in hv_df.iterrows():
        rp = str(row.get("run_path", ""))
        if rp == "":
            continue
        labels = read_metrics_current_labels(rp)
        ymat = read_final_solutions_y(rp)
        rows.append(
            {
                "benchmark": row.get("benchmark"),
                "mode": row.get("mode"),
                "case": row.get("case"),
                "formulation": row.get("formulation"),
                "best_algo": row.get("best_algo"),
                "best_hv": row.get("best_hv"),
                "run_path": rp,
                "proxy_labels": "|".join(labels),
                "proxy_dim": int(ymat.shape[1]) if ymat.size else 0,
                "n_saved_solutions": int(ymat.shape[0]) if ymat.size else 0,
            }
        )
    return pd.DataFrame(rows)


def summarize_by_group(df, metrics):
    rows = []
    gcols = ["benchmark", "case", "formulation"]
    for keys, sub in df.groupby(gcols, sort=True):
        row = dict(zip(gcols, keys))
        row["n_defs"] = int(len(sub))
        row["best_hv"] = float(sub["best_hv"].iloc[0]) if "best_hv" in sub.columns else np.nan
        row["best_algo"] = str(sub["best_algo"].iloc[0]) if "best_algo" in sub.columns else ""
        for m in metrics:
            vals = sub[m].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
                row[f"{m}_best"] = np.nan
                row[f"{m}_ci95_lo"] = np.nan
                row[f"{m}_ci95_hi"] = np.nan
            else:
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                row[f"{m}_best"] = float(np.min(vals)) if is_metric_minimize(m) else float(np.max(vals))
                lo, hi = bootstrap_ci_mean(vals, seed=13)
                row[f"{m}_ci95_lo"] = lo
                row[f"{m}_ci95_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_group_seed(df, metrics):
    if "seed" not in df.columns:
        return pd.DataFrame()

    rows = []
    gcols = ["seed", "benchmark", "case", "formulation"]
    for keys, sub in df.groupby(gcols, sort=True):
        row = dict(zip(gcols, keys))
        row["n_defs"] = int(len(sub))
        row["best_hv"] = float(sub["best_hv"].dropna().iloc[0]) if "best_hv" in sub.columns and sub["best_hv"].notna().any() else np.nan
        row["best_hv_mean"] = float(sub["best_hv_mean"].dropna().iloc[0]) if "best_hv_mean" in sub.columns and sub["best_hv_mean"].notna().any() else np.nan
        row["best_algo"] = str(sub["best_algo"].iloc[0]) if "best_algo" in sub.columns else ""

        for m in metrics:
            vals = sub[m].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
                row[f"{m}_best"] = np.nan
            else:
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                row[f"{m}_best"] = float(np.min(vals)) if is_metric_minimize(m) else float(np.max(vals))
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_seed_summary(summary_seed_df, metrics):
    if summary_seed_df.empty:
        return pd.DataFrame()

    rows = []
    gcols = ["benchmark", "case", "formulation"]
    for keys, sub in summary_seed_df.groupby(gcols, sort=True):
        row = dict(zip(gcols, keys))
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub.columns else 0
        row["n_seed_rows"] = int(len(sub))

        if "best_hv" in sub.columns:
            hv_vals = sub["best_hv"].dropna().to_numpy(dtype=float)
            if hv_vals.size:
                row["best_hv_seed_mean"] = float(np.mean(hv_vals))
                row["best_hv_seed_std"] = float(np.std(hv_vals, ddof=1)) if hv_vals.size > 1 else 0.0

        if "best_hv_mean" in sub.columns and sub["best_hv_mean"].notna().any():
            row["best_hv_mean"] = float(sub["best_hv_mean"].dropna().iloc[0])

        for m in metrics:
            best_col = f"{m}_best"
            if best_col in sub.columns:
                vals = sub[best_col].dropna().to_numpy(dtype=float)
                if vals.size:
                    row[f"{m}_best_seed_mean"] = float(np.mean(vals))
                    row[f"{m}_best_seed_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def compute_hv_metric_correlation(summary_df, metrics):
    rows = []
    if "best_hv" not in summary_df.columns:
        return pd.DataFrame()

    for m in metrics:
        col = f"{m}_best"
        if col not in summary_df.columns:
            continue
        sub = summary_df[["best_hv", col]].dropna()
        if len(sub) < 4:
            continue
        rows.append(
            {
                "metric": m,
                "n": int(len(sub)),
                "spearman": safe_corr(sub["best_hv"], sub[col], "spearman"),
                "pearson": safe_corr(sub["best_hv"], sub[col], "pearson"),
            }
        )
    return pd.DataFrame(rows)


def compare_hpo_mgo(summary_df, metrics):
    rows = []
    idx_cols = ["benchmark", "case"]
    if "seed" in summary_df.columns:
        idx_cols = ["seed"] + idx_cols
    piv = summary_df.pivot(index=idx_cols, columns="formulation")
    if piv.empty:
        return pd.DataFrame()

    for m in metrics:
        best_col = f"{m}_best"
        if (best_col, "MGO") not in piv.columns or (best_col, "HPO") not in piv.columns:
            continue

        mgo = piv[(best_col, "MGO")].to_numpy(dtype=float)
        hpo = piv[(best_col, "HPO")].to_numpy(dtype=float)
        valid = np.isfinite(mgo) & np.isfinite(hpo)
        if not np.any(valid):
            continue
        mgo = mgo[valid]
        hpo = hpo[valid]

        if is_metric_minimize(m):
            rel = (mgo - hpo) / np.maximum(np.abs(mgo), 1e-12)
            a = mgo
            b = hpo
        else:
            rel = (hpo - mgo) / np.maximum(np.abs(mgo), 1e-12)
            a = hpo
            b = mgo

        mean_diff, ci_lo, ci_hi = paired_bootstrap_diff(a, b, seed=29)
        rows.append(
            {
                "metric": m,
                "n_cases": int(len(rel)),
                "hpo_better_rate": float(np.mean(rel > 0)),
                "median_rel_improvement": float(np.median(rel)),
                "mean_rel_improvement": float(np.mean(rel)),
                "mean_abs_diff": mean_diff,
                "mean_abs_diff_ci95_lo": ci_lo,
                "mean_abs_diff_ci95_hi": ci_hi,
            }
        )

    return pd.DataFrame(rows)


def compute_proxy_ppa_correlation(proxy_long_df, ppa_metrics):
    rows = []
    if proxy_long_df.empty:
        return pd.DataFrame()

    for pname, sub_p in proxy_long_df.groupby("proxy_name"):
        for m in ppa_metrics:
            if m not in sub_p.columns:
                continue
            sub = sub_p[["proxy_value", m]].dropna()
            if len(sub) < 5:
                continue
            rows.append(
                {
                    "proxy_name": pname,
                    "ppa_metric": m,
                    "n": int(len(sub)),
                    "spearman": safe_corr(sub["proxy_value"], sub[m], "spearman"),
                    "pearson": safe_corr(sub["proxy_value"], sub[m], "pearson"),
                }
            )
    return pd.DataFrame(rows)


def compute_proxy_ppa_correlation_by_design(proxy_long_df, ppa_metrics, agg_mode="design_mean"):
    if proxy_long_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    gcols = ["benchmark", "case"]
    detail_rows = []
    for keys, sub_d in proxy_long_df.groupby(gcols, dropna=False):
        bench, case = keys
        for pname, sub_p in sub_d.groupby("proxy_name"):
            for m in ppa_metrics:
                if m not in sub_p.columns:
                    continue
                sub = sub_p[["proxy_value", m]].dropna()
                if len(sub) < 5:
                    continue
                detail_rows.append(
                    {
                        "benchmark": bench,
                        "case": case,
                        "proxy_name": pname,
                        "ppa_metric": m,
                        "n": int(len(sub)),
                        "spearman": safe_corr(sub["proxy_value"], sub[m], "spearman"),
                        "pearson": safe_corr(sub["proxy_value"], sub[m], "pearson"),
                    }
                )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return detail_df, pd.DataFrame()

    summary_rows = []
    for keys, sub in detail_df.groupby(["proxy_name", "ppa_metric"], dropna=False):
        pname, metric = keys
        sub = sub[np.isfinite(sub["spearman"])].copy()
        if sub.empty:
            continue

        if agg_mode == "design_weighted":
            w = sub["n"].to_numpy(dtype=float)
            s = sub["spearman"].to_numpy(dtype=float)
            p = sub["pearson"].to_numpy(dtype=float)
            spearman_aggr = float(np.sum(w * s) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(s))
            pearson_aggr = float(np.sum(w * p) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(p))
        else:
            spearman_aggr = float(np.mean(sub["spearman"].to_numpy(dtype=float)))
            pearson_aggr = float(np.mean(sub["pearson"].to_numpy(dtype=float)))

        summary_rows.append(
            {
                "proxy_name": pname,
                "ppa_metric": metric,
                "n_designs": int(len(sub)),
                "n_total": int(sub["n"].sum()),
                "spearman": spearman_aggr,
                "pearson": pearson_aggr,
                "spearman_design_std": float(np.std(sub["spearman"].to_numpy(dtype=float), ddof=1)) if len(sub) > 1 else 0.0,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return detail_df, summary_df


def compute_groupwise_rank_consistency(proxy_long_df, ppa_metrics):
    rows = []
    if proxy_long_df.empty:
        return pd.DataFrame()

    gcols = ["benchmark", "case", "formulation", "run_path"]
    for pname, sub_proxy in proxy_long_df.groupby("proxy_name"):
        for m in ppa_metrics:
            corrs = []
            for _, sub in sub_proxy.groupby(gcols):
                s = sub[["proxy_value", m]].dropna()
                if len(s) < 3:
                    continue
                c = safe_corr(s["proxy_value"], s[m], "spearman")
                if np.isfinite(c):
                    corrs.append(c)
            if not corrs:
                continue
            arr = np.asarray(corrs, dtype=float)
            rows.append(
                {
                    "proxy_name": pname,
                    "ppa_metric": m,
                    "n_groups": int(arr.size),
                    "spearman_mean": float(np.mean(arr)),
                    "spearman_median": float(np.median(arr)),
                    "positive_rate": float(np.mean(arr > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def compute_top1_hit_rate(proxy_long_df, ppa_metrics):
    detail_rows = []
    if proxy_long_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    gcols = ["benchmark", "case", "formulation", "run_path"]

    for pname, sub_proxy in proxy_long_df.groupby("proxy_name"):
        for m in ppa_metrics:
            for keys, sub in sub_proxy.groupby(gcols):
                s = sub[["proxy_value", m, "row_uid", "def_rank"]].dropna()
                if len(s) < 2:
                    continue
                row_proxy = s.loc[s["proxy_value"].astype(float).idxmin(), "row_uid"]
                if is_metric_minimize(m):
                    row_metric = s.loc[s[m].astype(float).idxmin(), "row_uid"]
                else:
                    row_metric = s.loc[s[m].astype(float).idxmax(), "row_uid"]
                hit = int(row_proxy == row_metric)
                detail_rows.append(
                    {
                        "benchmark": keys[0],
                        "case": keys[1],
                        "formulation": keys[2],
                        "run_path": keys[3],
                        "proxy_name": pname,
                        "ppa_metric": m,
                        "hit": hit,
                    }
                )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return detail_df, pd.DataFrame()
    summary_df = detail_df.groupby(["proxy_name", "ppa_metric"], as_index=False).agg(
        top1_hit_rate=("hit", "mean"),
        n_groups=("hit", "count"),
    )
    return detail_df, summary_df


def make_boxplots(df, metrics, fig_dir, metric_labels=None):
    for m in metrics:
        fig, ax = plt.subplots(figsize=(6.3, 4.2))
        data = []
        labels = []
        for form in ["MGO", "HPO"]:
            vals = df.loc[df["formulation"] == form, m].dropna().to_numpy(dtype=float)
            if len(vals):
                data.append(vals)
                labels.append(form)
        if not data:
            plt.close(fig)
            continue
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55)
        for patch, color in zip(bp["boxes"], ["#4C78A8", "#F58518"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ml = metric_label(m, metric_labels)
        ax.set_title(f"{ml}: Distribution by Formulation")
        ax.set_ylabel(ml)
        save_dual(fig, fig_dir, f"box_{m}_by_formulation")


def make_hv_scatter(df, metrics, fig_dir, metric_labels=None):
    if "best_hv" not in df.columns:
        return
    for m in metrics:
        sub = df[["best_hv", m, "formulation"]].dropna()
        if len(sub) < 5:
            continue
        fig, ax = plt.subplots(figsize=(6.6, 4.5))
        for form, color in [("MGO", "#4C78A8"), ("HPO", "#F58518")]:
            s = sub[sub["formulation"] == form]
            if len(s):
                ax.scatter(s["best_hv"], s[m], s=28, alpha=0.8, color=color, label=form)

        if sub["best_hv"].nunique() > 1:
            x = sub["best_hv"].to_numpy(dtype=float)
            y = sub[m].to_numpy(dtype=float)
            k, b = np.polyfit(x, y, 1)
            xx = np.linspace(np.min(x), np.max(x), 120)
            ax.plot(xx, k * xx + b, linestyle="--", color="black", linewidth=1.1)

        ml = metric_label(m, metric_labels)
        ax.set_title(
            f"HV vs {ml} (spearman={safe_corr(sub['best_hv'], sub[m], 'spearman'):.3f}, "
            f"pearson={safe_corr(sub['best_hv'], sub[m], 'pearson'):.3f})"
        )
        ax.set_xlabel("best_hv")
        ax.set_ylabel(ml)
        ax.legend(frameon=False)
        save_dual(fig, fig_dir, f"scatter_hv_vs_{m}")


def make_case_metric_heatmap(summary_df, metric, fig_dir, metric_labels=None):
    col = f"{metric}_best"
    if col not in summary_df.columns:
        return
    piv = summary_df.pivot(index="case", columns="formulation", values=col)
    if piv.empty:
        return
    arr = piv.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(4.9, max(3.2, 0.45 * len(piv.index) + 1.6)))
    im = ax.imshow(arr, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(list(piv.columns))
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(list(piv.index))
    ml = metric_label(metric, metric_labels)
    ax.set_title(f"Best {ml} by Case and Formulation")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            txt = "nan" if not np.isfinite(arr[i, j]) else f"{arr[i, j]:.3g}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_dual(fig, fig_dir, f"heatmap_best_{metric}")


def make_performance_profile(summary_df, metrics, fig_dir, metric_labels=None):
    for m in metrics:
        col = f"{m}_best"
        if col not in summary_df.columns:
            continue
        mat = summary_df.pivot(index="case", columns="formulation", values=col)
        if mat.empty or len(mat.columns) < 2:
            continue

        if is_metric_minimize(m):
            best = mat.min(axis=1)
            ratio = mat.div(best, axis=0)
        else:
            best = mat.max(axis=1)
            ratio = best.to_frame().div(mat)

        taus = np.linspace(1.0, 1.30, 160)
        fig, ax = plt.subplots(figsize=(6.2, 4.25))
        for form, color in [("MGO", "#4C78A8"), ("HPO", "#F58518")]:
            if form not in ratio.columns:
                continue
            vals = ratio[form].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            rho = [(vals <= t).mean() for t in taus]
            ax.plot(taus, rho, color=color, linewidth=2.1, label=form)
        ax.set_xlabel("tau")
        ax.set_ylabel("rho(tau)")
        ax.set_ylim(0.0, 1.03)
        ml = metric_label(m, metric_labels)
        ax.set_title(f"Performance Profile ({ml})")
        ax.legend(frameon=False)
        save_dual(fig, fig_dir, f"profile_{m}")


def make_proxy_ppa_corr_heatmap(proxy_corr_df, fig_dir):
    if proxy_corr_df.empty:
        return

    piv = proxy_corr_df.pivot(index="proxy_name", columns="ppa_metric", values="spearman")
    if piv.empty:
        return

    arr = piv.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.2 * len(piv.columns) + 2.8, 0.65 * len(piv.index) + 2.0))
    im = ax.imshow(arr, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(list(piv.columns), rotation=20, ha="right")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(list(piv.index))
    ax.set_title("Proxy-Real Metric Spearman Correlation")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            txt = "nan" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_dual(fig, fig_dir, "heatmap_proxy_ppa_spearman")


def make_rank_consistency_heatmap(rank_df, fig_dir):
    if rank_df.empty:
        return
    piv = rank_df.pivot(index="proxy_name", columns="ppa_metric", values="spearman_median")
    if piv.empty:
        return
    arr = piv.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.2 * len(piv.columns) + 2.8, 0.65 * len(piv.index) + 2.0))
    im = ax.imshow(arr, cmap="PiYG", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(list(piv.columns), rotation=20, ha="right")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(list(piv.index))
    ax.set_title("Within-Run Rank Consistency (Median Spearman)")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            txt = "nan" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_dual(fig, fig_dir, "heatmap_rank_consistency_median")


def make_top1_hit_bar(hit_summary_df, fig_dir):
    if hit_summary_df.empty:
        return

    plot_df = hit_summary_df.copy().sort_values(["ppa_metric", "proxy_name"]).reset_index(drop=True)
    labels = [f"{r.proxy_name}->{r.ppa_metric}" for r in plot_df.itertuples()]
    y = plot_df["top1_hit_rate"].to_numpy(dtype=float)

    fig_h = max(4.0, 0.28 * len(labels) + 1.6)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    ax.barh(np.arange(len(labels)), y, color="#54A24B", alpha=0.82)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Top-1 Hit Rate")
    ax.set_title("Proxy Argmin == PPA Best Match Rate")
    save_dual(fig, fig_dir, "bar_top1_hit_rate")


def make_proxy_mode_dim_bar(catalog_df, fig_dir):
    if catalog_df.empty:
        return

    c = catalog_df.copy()
    c = c[c["proxy_dim"] > 0]
    c = c[c["proxy_labels"].fillna("").astype(str).str.strip() != ""]
    if c.empty:
        return

    agg = c.groupby(["mode", "proxy_dim"], as_index=False).size()
    if agg.empty:
        return
    modes = sorted(agg["mode"].dropna().unique())
    dims = sorted(agg["proxy_dim"].dropna().unique())

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(dims))
    width = 0.36

    for i, mode in enumerate(modes):
        vals = []
        for d in dims:
            s = agg[(agg["mode"] == mode) & (agg["proxy_dim"] == d)]
            vals.append(float(s["size"].iloc[0]) if len(s) else 0.0)
        shift = (-0.5 + i / max(1, len(modes) - 1)) * width if len(modes) > 1 else 0.0
        ax.bar(x + shift, vals, width=width / max(1, len(modes) / 1.5), label=mode, alpha=0.82)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(d)) for d in dims])
    ax.set_xlabel("Proxy Objective Dimension")
    ax.set_ylabel("Number of Best Runs")
    ax.set_title("Proxy Objective Dimension by Mode (GP/MP)")
    ax.legend(frameon=False)
    save_dual(fig, fig_dir, "bar_proxy_dim_by_mode")


def make_all_metrics_corr_heatmap(ppa_df, proxy_long_df, ppa_metrics, fig_dir, tab_dir, metric_labels=None):
    if proxy_long_df.empty:
        return

    proxy_wide = proxy_long_df.pivot_table(index="row_uid", columns="proxy_name", values="proxy_value", aggfunc="first")
    ppa_wide = ppa_df[["row_uid"] + ppa_metrics].drop_duplicates(subset=["row_uid"]).set_index("row_uid")

    merged = proxy_wide.join(ppa_wide, how="inner")
    if merged.empty:
        return

    valid_cols = []
    for c in merged.columns:
        s = merged[c].dropna()
        if len(s) >= 3 and s.nunique() > 1:
            valid_cols.append(c)

    if len(valid_cols) < 2:
        return

    merged = merged[valid_cols]

    proxy_cols = [c for c in merged.columns if c in proxy_wide.columns]
    ppa_cols = [c for c in merged.columns if c in ppa_metrics]
    if not proxy_cols or not ppa_cols:
        return

    ordered = proxy_cols + ppa_cols
    corr = merged[ordered].corr(method="spearman")

    # Remove diagonal self-correlation to avoid visually dominant trivial 1.0 values.
    corr_plot = corr.copy()
    np.fill_diagonal(corr_plot.values, np.nan)

    corr_plot.to_csv(os.path.join(tab_dir, "all_metrics_spearman_corr.csv"), index=True)

    arr = corr_plot.to_numpy(dtype=float)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white")

    display_order = []
    for c in ordered:
        if c in ppa_cols:
            display_order.append(metric_label(c, metric_labels))
        else:
            display_order.append(c)

    fig, ax = plt.subplots(figsize=(0.56 * len(ordered) + 3.0, 0.56 * len(ordered) + 2.8))
    im = ax.imshow(arr, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels(display_order, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(display_order, fontsize=8)
    ax.set_title("All-Metrics Spearman Correlation (Proxy vs PPA)")

    # Match style with proxy-vs-ppa heatmap: show values in cells.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black")

    # Black separator lines between proxy block and PPA block.
    boundary = len(proxy_cols) - 0.5
    ax.axvline(boundary, color="black", linewidth=2.0)
    ax.axhline(boundary, color="black", linewidth=2.0)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_dual(fig, fig_dir, "heatmap_all_metrics_spearman")


def make_all_metrics_corr_heatmap_by_design(ppa_df, proxy_long_df, ppa_metrics, fig_dir, tab_dir, metric_labels=None, agg_mode="design_mean"):
    if proxy_long_df.empty:
        return

    merged_parts = []
    gcols = ["benchmark", "case"]
    for keys, sub_ppa in ppa_df.groupby(gcols, dropna=False):
        bench, case = keys
        sub_proxy = proxy_long_df[
            (proxy_long_df["benchmark"].astype(str) == str(bench))
            & (proxy_long_df["case"].astype(str) == str(case))
        ]
        if sub_proxy.empty:
            continue

        proxy_wide = sub_proxy.pivot_table(index="row_uid", columns="proxy_name", values="proxy_value", aggfunc="first")
        ppa_wide = sub_ppa[["row_uid"] + ppa_metrics].drop_duplicates(subset=["row_uid"]).set_index("row_uid")
        merged = proxy_wide.join(ppa_wide, how="inner")
        if merged.empty:
            continue

        valid_cols = []
        for c in merged.columns:
            s = merged[c].dropna()
            if len(s) >= 3 and s.nunique() > 1:
                valid_cols.append(c)
        if len(valid_cols) < 2:
            continue
        merged = merged[valid_cols]
        merged_parts.append((bench, case, merged))

    if not merged_parts:
        return

    all_proxy_cols = sorted(set().union(*[set(m.columns) for _, _, m in merged_parts]) - set(ppa_metrics))
    all_ppa_cols = [c for c in ppa_metrics if any(c in m.columns for _, _, m in merged_parts)]
    ordered = all_proxy_cols + all_ppa_cols
    if not all_proxy_cols or not all_ppa_cols:
        return

    corr_stack = []
    meta_rows = []
    for bench, case, merged in merged_parts:
        mat = merged.reindex(columns=ordered)
        corr = mat.corr(method="spearman")
        corr = corr.reindex(index=ordered, columns=ordered)
        corr_stack.append(corr.to_numpy(dtype=float))
        meta_rows.append({"benchmark": bench, "case": case, "n_rows": int(len(merged))})

    arr3 = np.stack(corr_stack, axis=0)
    if agg_mode == "design_weighted":
        ws = np.asarray([max(1, int(x["n_rows"])) for x in meta_rows], dtype=float)
        ws = ws / np.sum(ws)
        corr_mean = np.tensordot(ws, arr3, axes=(0, 0))
    else:
        corr_mean = np.nanmean(arr3, axis=0)
    np.fill_diagonal(corr_mean, np.nan)

    corr_plot = pd.DataFrame(corr_mean, index=ordered, columns=ordered)
    corr_plot.to_csv(os.path.join(tab_dir, "all_metrics_spearman_corr.csv"), index=True)

    pd.DataFrame(meta_rows).to_csv(os.path.join(tab_dir, "all_metrics_spearman_corr_design_groups.csv"), index=False)

    arr = corr_plot.to_numpy(dtype=float)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white")

    display_order = []
    for c in ordered:
        if c in all_ppa_cols:
            display_order.append(metric_label(c, metric_labels))
        else:
            display_order.append(c)

    fig, ax = plt.subplots(figsize=(0.56 * len(ordered) + 3.0, 0.56 * len(ordered) + 2.8))
    im = ax.imshow(arr, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels(display_order, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(display_order, fontsize=8)
    ax.set_title("All-Metrics Spearman Correlation (Design-Aggregated)")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black")

    boundary = len(all_proxy_cols) - 0.5
    ax.axvline(boundary, color="black", linewidth=2.0)
    ax.axhline(boundary, color="black", linewidth=2.0)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_dual(fig, fig_dir, "heatmap_all_metrics_spearman")


def write_report(report_path, args, ppa_df, ppa_metrics, summary_df, hv_corr_df, improve_df, proxy_corr_df, rank_df, hit_summary_df, catalog_df):
    lines = []
    lines.append("# PPA and Proxy Objective Joint Analysis Report")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Input PPA CSV: {args.ppa_csv}")
    lines.append(f"- Input HV JSON: {args.hv_json if args.hv_json else 'N/A'}")
    lines.append(f"- Output directory: {args.output_dir}")
    lines.append(f"- Rows after filtering: {len(ppa_df)}")
    if "seed" in ppa_df.columns:
        seeds = sorted(ppa_df["seed"].dropna().astype(int).unique().tolist())
        lines.append(f"- Seeds in PPA: {seeds}")
    lines.append(f"- Proxy map success rows: {int(ppa_df['proxy_map_ok'].sum()) if 'proxy_map_ok' in ppa_df.columns else 0}")
    lines.append(f"- Benchmarks in PPA: {', '.join(sorted(ppa_df['benchmark'].dropna().astype(str).unique()))}")
    lines.append(f"- Cases in PPA: {', '.join(sorted(ppa_df['case'].dropna().astype(str).unique()))}")
    lines.append(f"- PPA metrics analyzed: {', '.join(ppa_metrics)}")
    lines.append("")

    lines.append("## Key Findings")
    if not hv_corr_df.empty:
        top = hv_corr_df.iloc[hv_corr_df["spearman"].abs().values.argmax()]
        lines.append(
            f"- Strongest HV-vs-PPA relation: {top['metric']} (|spearman|={abs(top['spearman']):.3f}, n={int(top['n'])})."
        )
    else:
        lines.append("- HV-vs-PPA relation could not be computed from available rows.")

    if not proxy_corr_df.empty:
        top2 = proxy_corr_df.iloc[proxy_corr_df["spearman"].abs().values.argmax()]
        if "n" in proxy_corr_df.columns:
            n_text = f"n={int(top2['n'])}"
        elif "n_total" in proxy_corr_df.columns and "n_designs" in proxy_corr_df.columns:
            n_text = f"n_total={int(top2['n_total'])}, n_designs={int(top2['n_designs'])}"
        elif "n_total" in proxy_corr_df.columns:
            n_text = f"n_total={int(top2['n_total'])}"
        else:
            n_text = "n=NA"
        lines.append(
            f"- Strongest proxy-object-vs-PPA relation: {top2['proxy_name']} vs {top2['ppa_metric']} "
            f"(|spearman|={abs(top2['spearman']):.3f}, {n_text})."
        )
    else:
        lines.append("- Proxy-object-vs-PPA relation could not be computed.")

    if not hit_summary_df.empty:
        best_hit = hit_summary_df.iloc[hit_summary_df["top1_hit_rate"].values.argmax()]
        lines.append(
            f"- Best top-1 matching pair: {best_hit['proxy_name']} -> {best_hit['ppa_metric']} "
            f"(hit_rate={best_hit['top1_hit_rate']:.3f}, n_groups={int(best_hit['n_groups'])})."
        )

    if not improve_df.empty:
        best_imp = improve_df.iloc[improve_df["mean_rel_improvement"].values.argmax()]
        lines.append(
            f"- Largest average HPO gain over MGO: {best_imp['metric']} "
            f"(mean_rel_improvement={best_imp['mean_rel_improvement']:.3%})."
        )

    lines.append("")
    lines.append("## EDA/AI-Oriented Views Included")
    lines.append("- HV ↔ PPA correlation analysis (case/formulation aggregate).")
    lines.append("- Proxy objective Y(def_rank) ↔ real PPA direct correlation matrix.")
    lines.append("- Within-run rank consistency (Spearman) across top-5 saved DEFs.")
    lines.append("- Proxy-argmin vs PPA-best top-1 hit-rate analysis.")
    lines.append("- GP/MP mode-level proxy objective catalog and dimension distribution.")

    lines.append("")
    lines.append("## Tables")
    lines.append("- tables/summary_by_case_formulation.csv/.tex")
    lines.append("- tables/hv_metric_correlation.csv/.tex")
    lines.append("- tables/hpo_vs_mgo_improvement.csv/.tex")
    lines.append("- tables/proxy_metric_correlation.csv/.tex")
    lines.append("- tables/proxy_groupwise_rank_consistency.csv/.tex")
    lines.append("- tables/proxy_top1_hit_rate.csv/.tex")
    lines.append("- tables/proxy_top1_hit_detail.csv")
    lines.append("- tables/proxy_object_catalog_all_modes.csv/.tex")
    lines.append("- tables/proxy_mode_object_sets.csv/.tex")
    lines.append("- tables/all_metrics_spearman_corr.csv")
    lines.append("- tables/proxy_object_catalog_missing_labels.csv (if exists)")

    lines.append("")
    lines.append("## Figures")
    lines.append("- figures/box_<metric>_by_formulation.pdf")
    lines.append("- figures/scatter_hv_vs_<metric>.pdf")
    lines.append("- figures/profile_<metric>.pdf")
    lines.append("- figures/heatmap_best_<metric>.pdf")
    lines.append("- figures/heatmap_proxy_ppa_spearman.pdf")
    lines.append("- figures/heatmap_all_metrics_spearman.pdf")
    lines.append("- figures/heatmap_rank_consistency_median.pdf")
    lines.append("- figures/bar_top1_hit_rate.pdf")
    lines.append("- figures/bar_proxy_dim_by_mode.pdf")

    lines.append("")
    lines.append("## Data Notes")
    lines.append("- proxy_obj_i 来源于 checkpoint/final_solutions.pkl 中第 i 维 Y。")
    lines.append("- 相关图表中代理目标显示为 mode:objective_name（例如 GP:gp_hpwl, MP:hpwl）。")
    lines.append("- proxy 与 DEF 的映射使用 def_rank -> final_solutions[def_rank-1]。")
    lines.append("- 指标方向统一：分析阶段将 WNS/TNS 映射为 -WNS/-TNS，与其余指标统一为“越小越好”口径。")
    lines.append("- 当当前 PPA CSV 仅覆盖 GP（如 OpenROAD GP）时，MP 仅参与 proxy-object catalog，不参与 proxy↔PPA 直接相关。")
    lines.append(f"- 全局相关性聚合方式：{getattr(args, 'global_corr_mode', 'design_mean')}（先按 design 计算，再聚合）。")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def run_analysis_bundle(ppa_df, hv_df, args, output_dir, scope_title="overall"):
    ensure_dir(output_dir)
    report_dir = os.path.join(output_dir, "reports")
    fig_dir = os.path.join(output_dir, "figures")
    tab_dir = os.path.join(output_dir, "tables")
    ensure_dir(report_dir)
    ensure_dir(fig_dir)
    ensure_dir(tab_dir)

    if len(ppa_df) == 0:
        return

    ppa_df = ppa_df.copy().reset_index(drop=True)
    ppa_df["row_uid"] = np.arange(len(ppa_df), dtype=int)

    ppa_df = harmonize_openroad_columns(ppa_df)

    ppa_metrics = choose_ppa_metrics(ppa_df)
    if not ppa_metrics:
        raise ValueError(f"No numeric PPA metric columns found for scope={scope_title}.")

    ppa_df, ppa_metrics, metric_labels = normalize_metric_direction(ppa_df, ppa_metrics)
    ppa_metrics_display = [metric_label(m, metric_labels) for m in ppa_metrics]

    ppa_df = attach_proxy_vectors_by_defrank(ppa_df)
    proxy_long_df = build_proxy_long_df(ppa_df, ppa_metrics)

    summary_df = summarize_by_group(ppa_df, ppa_metrics)
    hv_corr_df = compute_hv_metric_correlation(summary_df, ppa_metrics)
    improve_df = compare_hpo_mgo(summary_df, ppa_metrics)
    if scope_title == "overall":
        proxy_corr_detail_df, proxy_corr_df = compute_proxy_ppa_correlation_by_design(
            proxy_long_df,
            ppa_metrics,
            agg_mode=getattr(args, "global_corr_mode", "design_mean"),
        )
    else:
        proxy_corr_df = compute_proxy_ppa_correlation(proxy_long_df, ppa_metrics)
        proxy_corr_detail_df = pd.DataFrame()
    rank_df = compute_groupwise_rank_consistency(proxy_long_df, ppa_metrics)
    hit_detail_df, hit_summary_df = compute_top1_hit_rate(proxy_long_df, ppa_metrics)

    # Convert metric identifiers to display labels for readability in tables.
    for df_metric in [hv_corr_df, improve_df]:
        if not df_metric.empty and "metric" in df_metric.columns:
            df_metric["metric"] = df_metric["metric"].map(lambda x: metric_label(x, metric_labels))
    for df_metric in [proxy_corr_df, rank_df, hit_summary_df, hit_detail_df]:
        if not df_metric.empty and "ppa_metric" in df_metric.columns:
            df_metric["ppa_metric"] = df_metric["ppa_metric"].map(lambda x: metric_label(x, metric_labels))

    if args.hv_json and not hv_df.empty:
        hv_subset = hv_df.copy()
        if "benchmark" in ppa_df.columns:
            benches = set(ppa_df["benchmark"].dropna().astype(str).unique())
            hv_subset = hv_subset[hv_subset["benchmark"].astype(str).isin(benches)]
        if "case" in ppa_df.columns:
            cases = set(ppa_df["case"].dropna().astype(str).unique())
            hv_subset = hv_subset[hv_subset["case"].astype(str).isin(cases)]
        if "formulation" in ppa_df.columns:
            forms = set(ppa_df["formulation"].dropna().astype(str).unique())
            hv_subset = hv_subset[hv_subset["formulation"].astype(str).isin(forms)]
        catalog_df = build_proxy_catalog_from_hv(hv_subset)
    else:
        catalog_df = pd.DataFrame()

    save_df(summary_df, os.path.join(tab_dir, "summary_by_case_formulation.csv"), os.path.join(tab_dir, "summary_by_case_formulation.tex"))
    save_df(hv_corr_df, os.path.join(tab_dir, "hv_metric_correlation.csv"), os.path.join(tab_dir, "hv_metric_correlation.tex"))
    save_df(improve_df, os.path.join(tab_dir, "hpo_vs_mgo_improvement.csv"), os.path.join(tab_dir, "hpo_vs_mgo_improvement.tex"))
    save_df(proxy_corr_df, os.path.join(tab_dir, "proxy_metric_correlation.csv"), os.path.join(tab_dir, "proxy_metric_correlation.tex"))
    if not proxy_corr_detail_df.empty:
        proxy_corr_detail_df.to_csv(os.path.join(tab_dir, "proxy_metric_correlation_by_design_detail.csv"), index=False)
    save_df(rank_df, os.path.join(tab_dir, "proxy_groupwise_rank_consistency.csv"), os.path.join(tab_dir, "proxy_groupwise_rank_consistency.tex"))
    save_df(hit_summary_df, os.path.join(tab_dir, "proxy_top1_hit_rate.csv"), os.path.join(tab_dir, "proxy_top1_hit_rate.tex"))

    hit_detail_df.to_csv(os.path.join(tab_dir, "proxy_top1_hit_detail.csv"), index=False)

    if not catalog_df.empty:
        save_df(catalog_df, os.path.join(tab_dir, "proxy_object_catalog_all_modes.csv"), os.path.join(tab_dir, "proxy_object_catalog_all_modes.tex"))
        missing_df = catalog_df[catalog_df["proxy_labels"].fillna("").astype(str).str.strip() == ""]
        if not missing_df.empty:
            missing_df.to_csv(os.path.join(tab_dir, "proxy_object_catalog_missing_labels.csv"), index=False)

        known_df = catalog_df[catalog_df["proxy_labels"].fillna("").astype(str).str.strip() != ""]
        mode_sets = (
            known_df.groupby(["mode", "proxy_labels"], as_index=False)
            .size()
            .rename(columns={"size": "n_best_runs"})
            .sort_values(["mode", "n_best_runs"], ascending=[True, False])
        )
        save_df(mode_sets, os.path.join(tab_dir, "proxy_mode_object_sets.csv"), os.path.join(tab_dir, "proxy_mode_object_sets.tex"))

    ppa_df.to_csv(os.path.join(tab_dir, "ppa_with_proxy_alignment.csv"), index=False)
    proxy_long_df.to_csv(os.path.join(tab_dir, "proxy_long_alignment.csv"), index=False)

    make_boxplots(ppa_df, ppa_metrics, fig_dir, metric_labels=metric_labels)
    make_hv_scatter(ppa_df, ppa_metrics, fig_dir, metric_labels=metric_labels)
    make_performance_profile(summary_df, ppa_metrics, fig_dir, metric_labels=metric_labels)
    for m in ppa_metrics:
        make_case_metric_heatmap(summary_df, m, fig_dir, metric_labels=metric_labels)

    make_proxy_ppa_corr_heatmap(proxy_corr_df, fig_dir)
    make_rank_consistency_heatmap(rank_df, fig_dir)
    make_top1_hit_bar(hit_summary_df, fig_dir)
    make_proxy_mode_dim_bar(catalog_df, fig_dir)
    if scope_title == "overall":
        make_all_metrics_corr_heatmap_by_design(
            ppa_df,
            proxy_long_df,
            ppa_metrics,
            fig_dir,
            tab_dir,
            metric_labels=metric_labels,
            agg_mode=getattr(args, "global_corr_mode", "design_mean"),
        )
    else:
        make_all_metrics_corr_heatmap(ppa_df, proxy_long_df, ppa_metrics, fig_dir, tab_dir, metric_labels=metric_labels)

    report_args = argparse.Namespace(ppa_csv=args.ppa_csv, hv_json=args.hv_json, output_dir=output_dir)
    write_report(
        report_path=os.path.join(report_dir, "analysis_report.md"),
        args=report_args,
        ppa_df=ppa_df,
        ppa_metrics=ppa_metrics_display,
        summary_df=summary_df,
        hv_corr_df=hv_corr_df,
        improve_df=improve_df,
        proxy_corr_df=proxy_corr_df,
        rank_df=rank_df,
        hit_summary_df=hit_summary_df,
        catalog_df=catalog_df,
    )


def main():
    parser = argparse.ArgumentParser(description="Joint analysis for PPA metrics and proxy objectives")
    parser.add_argument("--ppa_csv", required=True, help="Path to PPA evaluation CSV")
    parser.add_argument("--hv_json", default=None, help="Path to hv_summary_seed_*.json")
    parser.add_argument(
        "--output_dir",
        default=os.path.join("results", "analysis_reports", "ppa_proxy", "study"),
        help="Output directory",
    )
    parser.add_argument("--benchmarks", default="all", help="all or comma list")
    parser.add_argument("--cases", default="all", help="all or comma list")
    parser.add_argument("--formulations", default="MGO,HPO", help="Comma list")
    parser.add_argument("--modes", default="GP", help="Comma list from GP,MP (default: GP)")
    parser.add_argument("--only_eval_ok", action="store_true", help="Only keep eval_ok rows if available")
    parser.add_argument(
        "--global_corr_mode",
        default="design_mean",
        choices=["design_mean", "design_weighted"],
        help="Aggregation mode for overall correlation tables/heatmaps",
    )
    args = parser.parse_args()

    mode_filter = parse_modes(args.modes)
    mode_tag = "_".join([m.lower() for m in mode_filter])

    # Avoid accidentally overwriting existing GP analysis outputs when running MP or mixed modes.
    if mode_tag != "gp":
        base_name = os.path.basename(os.path.normpath(args.output_dir)).lower()
        if f"mode_{mode_tag}" not in base_name and mode_tag not in base_name:
            args.output_dir = f"{args.output_dir}_{mode_tag}"
            print(f"[INFO] output_dir adjusted to avoid overwrite: {args.output_dir}")

    configure_style()
    ensure_dir(args.output_dir)

    ppa_df = pd.read_csv(args.ppa_csv)

    if args.hv_json:
        hv_df = load_hv_table_all_modes(args.hv_json)
        if not hv_df.empty:
            hv_sel = hv_df[hv_df["mode"].astype(str).str.upper().isin(mode_filter)].copy()
            if "mode" in ppa_df.columns:
                ppa_df = ppa_df.merge(
                    hv_sel,
                    on=["benchmark", "case", "formulation", "run_path", "mode"],
                    how="left",
                    suffixes=("", "_hvjson"),
                )
            else:
                hv_sel_drop_mode = hv_sel.drop(columns=["mode"], errors="ignore")
                ppa_df = ppa_df.merge(
                    hv_sel_drop_mode,
                    on=["benchmark", "case", "formulation", "run_path"],
                    how="left",
                    suffixes=("", "_hvjson"),
                )

            if "best_hv" not in ppa_df.columns and "best_hv_hvjson" in ppa_df.columns:
                ppa_df["best_hv"] = ppa_df["best_hv_hvjson"]
            elif "best_hv" in ppa_df.columns and "best_hv_hvjson" in ppa_df.columns:
                ppa_df["best_hv"] = ppa_df["best_hv"].fillna(ppa_df["best_hv_hvjson"])

            if "best_algo" not in ppa_df.columns and "best_algo_hvjson" in ppa_df.columns:
                ppa_df["best_algo"] = ppa_df["best_algo_hvjson"]
            elif "best_algo" in ppa_df.columns and "best_algo_hvjson" in ppa_df.columns:
                ppa_df["best_algo"] = ppa_df["best_algo"].fillna(ppa_df["best_algo_hvjson"])
    else:
        hv_df = pd.DataFrame()

    bench_filter = parse_filters(args.benchmarks)
    case_filter = parse_filters(args.cases)
    form_filter = parse_filters(args.formulations)

    if bench_filter is not None and "benchmark" in ppa_df.columns:
        ppa_df = ppa_df[ppa_df["benchmark"].isin(bench_filter)]
    if case_filter is not None and "case" in ppa_df.columns:
        ppa_df = ppa_df[ppa_df["case"].isin(case_filter)]
    if form_filter is not None and "formulation" in ppa_df.columns:
        ppa_df = ppa_df[ppa_df["formulation"].isin(form_filter)]
    if "mode" in ppa_df.columns:
        ppa_df = ppa_df[ppa_df["mode"].astype(str).str.upper().isin(mode_filter)]
    if args.only_eval_ok and "eval_ok" in ppa_df.columns:
        ppa_df = ppa_df[ppa_df["eval_ok"].astype(bool)]

    if len(ppa_df) == 0:
        raise ValueError("No rows remain after filtering.")

    run_analysis_bundle(ppa_df, hv_df, args, args.output_dir, scope_title="overall")

    # Additional per-design analysis for OpenROAD cases (1 + N batches output).
    per_design_count = 0
    if "benchmark" in ppa_df.columns and "case" in ppa_df.columns:
        openroad_df = ppa_df[ppa_df["benchmark"].astype(str) == "OpenROAD"].copy()
        openroad_cases = sorted(openroad_df["case"].dropna().astype(str).unique().tolist())
        for case_name in openroad_cases:
            sub_df = openroad_df[openroad_df["case"].astype(str) == case_name].copy()
            if sub_df.empty:
                continue
            case_out = os.path.join(args.output_dir, "by_design", sanitize_name(case_name))
            run_analysis_bundle(sub_df, hv_df, args, case_out, scope_title=f"design:{case_name}")
            per_design_count += 1

    print("Analysis complete.")
    print(f"Overall output dir: {args.output_dir}")
    print(f"Per-design batches generated: {per_design_count}")


if __name__ == "__main__":
    main()
