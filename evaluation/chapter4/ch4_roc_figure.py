"""Figure 4.1: ROC curves of logprob_norm and CometKiwi for the four languages
(label: bottom chrF++ quartile). Reads evaluation/data/."""
import numpy as np, pandas as pd, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

def roc(y, s):
    o = np.argsort(-np.asarray(s, float)); y = np.asarray(y)[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]
def auc(y, s):
    r = pd.Series(np.asarray(s, float)).rank().to_numpy(); y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

h = pd.read_csv(f"{DATA}/hin_dev_scores.csv")
D = {"Hindi": (h.logprob_norm, h.cometkiwi_score, h.chrf)}
base = pd.read_csv(f"{DATA}/npi_tgl_dev_scores.csv")
cand = pd.read_csv(f"{DATA}/npi_tgl_candidate_scores.csv")
for L, name in (("npi", "Nepali"), ("tgl", "Tagalog")):
    lp = pd.read_csv(f"{DATA}/logprob_{L}.csv")
    m = base[base.language == L].merge(lp[["sentence_id", "logprob_norm"]], on="sentence_id").merge(
        cand[cand.language == L][["sentence_id", "cometkiwi_score"]], on="sentence_id")
    D[name] = (m.logprob_norm, m.cometkiwi_score, m.chrf)
g = pd.read_csv(f"{DATA}/ige_val_scores.csv")
D["Igede"] = (g.logprob_norm, g.cometkiwi, g.chrfpp)

fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0))
for ax, (name, (lp, kw, ch)) in zip(axes.flat, D.items()):
    y = (ch <= ch.quantile(0.25)).astype(int).to_numpy()
    for s, lab, st, col in ((-lp, "logprob_norm", "-", "#1a1a1a"), (-kw, "CometKiwi", "--", "#888888")):
        f, t = roc(y, s); ax.plot(f, t, st, color=col, lw=1.6, label=f"{lab} ({auc(y, s):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#bbbbbb", lw=1)
    ax.set_title(f"{name}, n = {len(y)}", fontsize=10)
    ax.set_xlabel("False positive rate", fontsize=8); ax.set_ylabel("True positive rate", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(loc="lower right", fontsize=7.5)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig4_1_roc_all.png"), dpi=200); print("ok")
