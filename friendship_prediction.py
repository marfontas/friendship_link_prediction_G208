"""
Can We Predict a Friendship Before It Exists?
Link prediction with Common Neighbors and Adamic-Adar on the SNAP CollegeMsg dataset.

HOW IT WORKS
1. Load the messages (sender, receiver, timestamp) and treat each pair of users as one
   undirected "connection" that starts at their FIRST message.
2. Split time: the first 60% of messages -> the network we "know" (training graph).
3. For every pair of users who had NOT talked yet, compute:
      - Common Neighbors (CN): number of mutual contacts
      - Adamic-Adar (AA): mutual contacts, where rare contacts count more
4. Check which of those pairs really DID start talking in the remaining 40% (the future).
5. Save four poster-ready charts + a results table.

USAGE
    pip install numpy pandas matplotlib networkx scikit-learn
    python friendship_prediction.py                # downloads CollegeMsg automatically
    python friendship_prediction.py --data CollegeMsg.txt.gz
    python friendship_prediction.py --split 0.5    # try a different train/test split

Dataset: https://snap.stanford.edu/data/CollegeMsg.html
(If the download fails, download "CollegeMsg.txt.gz" manually and use --data.)
"""

import argparse
import os
import urllib.request

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

URL = "https://snap.stanford.edu/data/CollegeMsg.txt.gz"
OUT_DIR = "poster_figures"
SEED = 42

# Colours (consistent across all charts)
C_CN = "#1f77b4"      # Common Neighbors
C_AA = "#d62728"      # Adamic-Adar
C_RAND = "#7f7f7f"    # Random guessing
C_GREEN = "#2ca02c"   # Correct prediction

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
def load_data(path):
    if path is None:
        path = "CollegeMsg.txt.gz"
        if not os.path.exists(path):
            print(f"Downloading {URL} ...")
            req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r, open(path, "wb") as f:
                f.write(r.read())
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["src", "dst", "t"])
    df = df[df["src"] != df["dst"]]  # remove messages to oneself
    print(f"Loaded {len(df):,} messages between {len(set(df.src) | set(df.dst)):,} users")
    return df


def first_contacts(df):
    """One row per undirected pair, with the time of their first message."""
    a = df[["src", "dst"]].min(axis=1)
    b = df[["src", "dst"]].max(axis=1)
    pairs = (
        pd.DataFrame({"a": a, "b": b, "t": df["t"]})
        .groupby(["a", "b"], as_index=False)["t"].min()
    )
    return pairs


# --------------------------------------------------------------------------- #
# 2. Prediction scores
# --------------------------------------------------------------------------- #
def build_scores(train_pairs):
    nodes = sorted(set(train_pairs.a) | set(train_pairs.b))
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=np.float32)
    for u, v in zip(train_pairs.a, train_pairs.b):
        A[idx[u], idx[v]] = A[idx[v], idx[u]] = 1.0

    deg = A.sum(axis=1)
    w = np.zeros(n, dtype=np.float32)
    w[deg > 1] = 1.0 / np.log(deg[deg > 1])      # Adamic-Adar weight of each user

    CN = A @ A                                    # number of common neighbours
    AA = (A * w[None, :]) @ A                     # sum of 1/log(degree) over common neighbours
    return nodes, idx, A, CN, AA


def precision_at_k(scores, y, ks, rng):
    order = np.argsort(-(scores + rng.random(len(scores)) * 1e-9))  # random tie-breaking
    return {k: y[order[:k]].mean() for k in ks}


# --------------------------------------------------------------------------- #
# 3. Charts
# --------------------------------------------------------------------------- #
def chart_probability_vs_common_friends(cn, y, path):
    bins = np.minimum(cn, 4).astype(int)
    labels = ["0", "1", "2", "3", "4+"]
    rates, counts = [], []
    for b in range(5):
        m = bins == b
        counts.append(int(m.sum()))
        rates.append(100 * y[m].mean() if m.any() else 0.0)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, rates, color=[C_RAND] + [C_CN] * 4)
    base = rates[0] if rates[0] > 0 else np.nan
    for bar, r, c, i in zip(bars, rates, counts, range(5)):
        txt = f"{r:.1f}%"
        if i > 0 and not np.isnan(base):
            txt += f"\n(x{r / base:.0f})"
        ax.text(bar.get_x() + bar.get_width() / 2, r, txt, ha="center", va="bottom", fontsize=15)
    ax.set_xlabel("Number of common friends (mutual contacts)")
    ax.set_ylabel("Pairs that connected later (%)")
    ax.set_title("More mutual friends = more likely to connect")
    ax.set_ylim(0, max(rates) * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return dict(zip(labels, rates)), dict(zip(labels, counts))


def chart_roc(cn, aa, y, path):
    auc_cn, auc_aa = roc_auc_score(y, cn), roc_auc_score(y, aa)
    fig, ax = plt.subplots(figsize=(7, 7))
    for s, name, col, auc in [(cn, "Common Neighbors", C_CN, auc_cn),
                              (aa, "Adamic-Adar", C_AA, auc_aa)]:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=col, lw=3, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=C_RAND, lw=2, label="Random guessing (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("How well can we predict new connections?")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return auc_cn, auc_aa


def chart_precision_at_k(p_cn, p_aa, base_rate, ks, path):
    x = np.arange(len(ks))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w, [100 * p_cn[k] for k in ks], w, color=C_CN, label="Common Neighbors")
    ax.bar(x, [100 * p_aa[k] for k in ks], w, color=C_AA, label="Adamic-Adar")
    ax.bar(x + w, [100 * base_rate] * len(ks), w, color=C_RAND, label="Random guessing")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top {k}" for k in ks])
    ax.set_ylabel("Correct predictions (%)")
    ax.set_title("If we recommend the top-k pairs, how many are right?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_network_example(nodes, A, ii, jj, y, aa, order, path):
    """Clean example: one correctly predicted future link and their mutual friends."""
    top = order[:300]
    hits = [k for k in top if y[k]]
    if not hits:
        print("No correct predictions in the top 300; skipping network drawing.")
        return

    def common_of(k):
        return np.where((A[ii[k]] > 0) & (A[jj[k]] > 0))[0]

    # prefer an example with a readable number of mutual friends (3-6)
    nice = [k for k in hits if 3 <= len(common_of(k)) <= 6] or hits
    k = nice[0]
    common = common_of(k)

    n_total = len(common)
    shown = common[:8]                      # draw at most 8 mutual friends to stay readable
    m = len(shown)
    fig, ax = plt.subplots(figsize=(9, 6))
    xa, xb = 0.0, 2.0
    angles = np.linspace(np.pi * 0.85, np.pi * 0.15, m) if m > 1 else np.array([np.pi / 2])
    mx = 1.0 + 0.95 * np.cos(angles)
    my = 0.4 + 1.5 * np.sin(angles)
    for x, y_ in zip(mx, my):
        ax.plot([xa, x], [0, y_], color="#aaaaaa", lw=3, zorder=1)
        ax.plot([xb, x], [0, y_], color="#aaaaaa", lw=3, zorder=1)
    ax.plot([xa, xb], [0, 0], color=C_GREEN, lw=5, ls="--", zorder=2)
    ax.scatter(mx, my, s=900, color=C_CN, zorder=3)
    ax.scatter([xa, xb], [0, 0], s=1800, color="#ff7f0e", zorder=3)
    ax.text(xa, 0, "A", ha="center", va="center", fontsize=24, color="white", zorder=4)
    ax.text(xb, 0, "B", ha="center", va="center", fontsize=24, color="white", zorder=4)
    ax.text(1.0, -0.22, "Will they connect?", ha="center", va="top", fontsize=16,
            color=C_GREEN, fontweight="bold")
    suffix = "" if n_total <= 8 else f" ({n_total} in total, 8 shown)"
    ax.set_title(f"A and B shared {n_total} friends... and connected later")
    if suffix:
        ax.text(1.0, 2.35, suffix.strip(), ha="center", fontsize=13, color="#555555")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", markersize=16,
               label="Two people who had not talked yet"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CN, markersize=14,
               label="Their mutual friends"),
        Line2D([0], [0], color=C_GREEN, lw=4, ls="--", label="Connection made later"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(-0.8, 2.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="path to CollegeMsg.txt(.gz)")
    parser.add_argument("--split", type=float, default=0.6,
                        help="fraction of messages (in time) used as known network")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- data and time split
    df = load_data(args.data)
    pairs = first_contacts(df)
    cutoff = df["t"].quantile(args.split)
    train = pairs[pairs.t <= cutoff]
    future = pairs[pairs.t > cutoff]
    print(f"Known network (past): {len(train):,} connections")
    print(f"New connections (future): {len(future):,}")

    # --- scores
    nodes, idx, A, CN, AA = build_scores(train)
    n = len(nodes)
    Y = np.zeros((n, n), dtype=bool)
    n_future_used = 0
    for u, v in zip(future.a, future.b):
        if u in idx and v in idx:                 # both users already known (no cold start)
            Y[idx[u], idx[v]] = Y[idx[v], idx[u]] = True
            n_future_used += 1
    print(f"Users in known network: {n:,}")
    print(f"New connections between already-known users: {n_future_used:,}")

    iu = np.triu_indices(n, 1)
    mask = A[iu] == 0                              # only pairs that had NOT connected yet
    ii, jj = iu[0][mask], iu[1][mask]
    cn, aa, y = CN[ii, jj], AA[ii, jj], Y[ii, jj]
    base_rate = y.mean()
    print(f"Candidate pairs: {len(y):,} | of which connect later: {int(y.sum()):,} "
          f"({100 * base_rate:.3f}%)")

    # --- charts
    rates, counts = chart_probability_vs_common_friends(
        cn, y, f"{OUT_DIR}/1_probability_vs_common_friends.png")
    auc_cn, auc_aa = chart_roc(cn, aa, y, f"{OUT_DIR}/2_roc_curves.png")
    ks = [50, 100, 200, 500]
    p_cn = precision_at_k(cn, y, ks, rng)
    p_aa = precision_at_k(aa, y, ks, rng)
    chart_precision_at_k(p_cn, p_aa, base_rate, ks, f"{OUT_DIR}/3_precision_at_k.png")
    order_aa = np.argsort(-(aa + rng.random(len(aa)) * 1e-9))
    chart_network_example(nodes, A, ii, jj, y, aa, order_aa,
                          f"{OUT_DIR}/4_network_example.png")

    # --- results table
    rows = [{"method": "Random guessing", "AUC": 0.5,
             **{f"precision@{k}": base_rate for k in ks}},
            {"method": "Common Neighbors", "AUC": auc_cn,
             **{f"precision@{k}": p_cn[k] for k in ks}},
            {"method": "Adamic-Adar", "AUC": auc_aa,
             **{f"precision@{k}": p_aa[k] for k in ks}}]
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT_DIR}/results.csv", index=False)

    # --- summary for the poster
    print("\n================ RESULTS FOR THE POSTER ================")
    print(res.round(4).to_string(index=False))
    print("\nProbability of connecting later by number of common friends:")
    for k in rates:
        print(f"  {k:>2} common friends: {rates[k]:.2f}%  ({counts[k]:,} pairs)")
    if rates["0"] > 0:
        print(f"\nPairs with 4+ common friends are about x{rates['4+'] / rates['0']:.0f} "
              f"more likely to connect than pairs with none.")
    best = max(("Common Neighbors", auc_cn), ("Adamic-Adar", auc_aa), key=lambda t: t[1])
    print(f"Best method by AUC: {best[0]} ({best[1]:.3f}).")
    print(f"Random precision@100 = {100 * base_rate:.2f}% vs "
          f"Adamic-Adar precision@100 = {100 * p_aa[100]:.1f}%.")
    print(f"\nCharts and results.csv saved in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()