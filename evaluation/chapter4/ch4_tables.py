"""
Recomputes the Chapter 4 tables from the saved per-sentence scores in
evaluation/data/. No translation and no model loading.

1) AUC of each candidate trigger signal against the bad-translation label
   (bottom chrF++ quartile per language): Table 4.1, ch4_auc_replication.csv.
2) Discriminative power of the trigger: mean chrF++ and COMET in the flagged
   quartile (lowest 25% of logprob_norm) versus the rest, with a bootstrap
   interval of the difference: Table 4.2, ch4_flagged_gap.csv.
3) Paired bootstrap of LaBSE against logprob_norm for Nepali and Tagalog:
   ch4_labse_vs_logprob_bootstrap.csv.

AUC is computed from ranks (Mann-Whitney), without sklearn.
Seed 42, 5,000 resamples.
"""
import numpy as np, pandas as pd, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
rng = np.random.default_rng(42)
B = 5000

def auc(y, s):
    # s: higher means more likely bad; y = 1 marks a bad translation
    y = np.asarray(y); s = np.asarray(s, float)
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def load():
    d = {}
    h = pd.read_csv(f"{DATA}/hin_dev_scores.csv")
    d["hin"] = pd.DataFrame({"logprob": h.logprob_norm, "cometkiwi": h.cometkiwi_score,
                             "len": h.n_tokens_src, "chrf": h.chrf, "comet": h.comet})
    base = pd.read_csv(f"{DATA}/npi_tgl_dev_scores.csv")
    cand = pd.read_csv(f"{DATA}/npi_tgl_candidate_scores.csv")
    for L in ("npi", "tgl"):
        lp = pd.read_csv(f"{DATA}/logprob_{L}.csv")
        m = base[base.language == L].merge(lp[["sentence_id", "logprob_norm"]], on="sentence_id") \
            .merge(cand[cand.language == L][["sentence_id", "src_len", "cometkiwi_score", "labse_cosine"]], on="sentence_id")
        d[L] = pd.DataFrame({"logprob": m.logprob_norm, "cometkiwi": m.cometkiwi_score,
                             "labse": m.labse_cosine, "len": m.src_len, "chrf": m.chrf, "comet": m.comet})
    g = pd.read_csv(f"{DATA}/ige_val_scores.csv")
    d["ige"] = pd.DataFrame({"logprob": g.logprob_norm, "cometkiwi": g.cometkiwi,
                             "len": g.n_tokens_src, "chrf": g.chrfpp, "comet": g.comet})
    return d


d = load()
rows_auc, rows_gap, rows_boot = [], [], []
for L, df in d.items():
    df = df.dropna(subset=["logprob", "chrf"]).reset_index(drop=True)
    n = len(df)
    y = (df.chrf <= df.chrf.quantile(0.25)).astype(int).to_numpy()
    # Orientation fixed in advance: lower logprob/cometkiwi/labse = worse; longer = worse
    sig = {"logprob": -df.logprob, "cometkiwi": -df.cometkiwi, "len": df["len"]}
    if "labse" in df: sig["labse"] = -df.labse
    for k, s in sig.items():
        rows_auc.append({"language": L, "candidate": k, "auc": round(auc(y, s), 4), "n": n, "n_bad": int(y.sum())})
    # Discriminative power: flagged = lowest 25% of logprob_norm
    flag = (df.logprob <= df.logprob.quantile(0.25)).to_numpy()
    for met in ("chrf", "comet"):
        v = df[met].to_numpy()
        diffs = []
        for _ in range(B):
            idx = rng.integers(0, n, n)
            f = flag[idx]; vv = v[idx]
            diffs.append(vv[~f].mean() - vv[f].mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows_gap.append({"language": L, "metric": met, "n": n, "n_flagged": int(flag.sum()),
                         "mean_flagged": round(v[flag].mean(), 4), "mean_unflagged": round(v[~flag].mean(), 4),
                         "diff_unflagged_minus_flagged": round(v[~flag].mean() - v[flag].mean(), 4),
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
    if "labse" in df:
        a = -df.labse.to_numpy(); b = -df.logprob.to_numpy()
        diffs = []
        for _ in range(B):
            idx = rng.integers(0, n, n)
            if y[idx].sum() in (0, len(idx)): continue
            diffs.append(auc(y[idx], a[idx]) - auc(y[idx], b[idx]))
        diffs = np.array(diffs)
        rows_boot.append({"language": L, "pair": "labse-logprob", "point": round(auc(y, a) - auc(y, b), 4),
                          "ci_lo": round(np.percentile(diffs, 2.5), 4), "ci_hi": round(np.percentile(diffs, 97.5), 4),
                          "share_labse_wins": round((diffs > 0).mean(), 3)})

out = HERE
pd.DataFrame(rows_auc).to_csv(f"{out}/ch4_auc_replication.csv", index=False)
pd.DataFrame(rows_gap).to_csv(f"{out}/ch4_flagged_gap.csv", index=False)
pd.DataFrame(rows_boot).to_csv(f"{out}/ch4_labse_vs_logprob_bootstrap.csv", index=False)
print(pd.DataFrame(rows_auc).to_string()); print(pd.DataFrame(rows_gap).to_string()); print(pd.DataFrame(rows_boot).to_string())
