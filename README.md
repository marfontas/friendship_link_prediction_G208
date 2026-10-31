# Can We Predict a Friendship Before It Exists?

Link prediction in a social network using **Common Neighbors** and **Adamic-Adar**.

Project for *Technical Communication in English 2026-27*, Universitat Pompeu Fabra (UPF).

**Authors:** Mar Fontàs Sureda · Diyana Dimitrov Dimova · Mateu Soria Waha

---

## Question

Given two people who have not interacted yet, can we predict whether they will connect in the future using only the structure of the network (their mutual friends)?

## Dataset

[**CollegeMsg**](https://snap.stanford.edu/data/CollegeMsg.html) (Stanford SNAP): private messages on an online social network at the University of California, Irvine.

| | |
|---|---|
| Users | 1,899 |
| Messages (temporal edges) | 59,835 |
| Unique user pairs | 20,296 |
| Time span | 193 days |

> **Note:** a "connection" here means *started a conversation* (sent a message), not necessarily *became friends*.

The script downloads the dataset automatically. It is not included in this repository.

## Method

1. Each pair of users is one undirected connection that starts at their **first message**.
2. **Time split:** the first 60% of messages build the known network (8,840 connections, 1,399 users). The last 40% is the "future" (2,260 new connections between already-known users).
3. For every pair that had not talked yet (969,061 candidate pairs), two scores are computed:
   - **Common Neighbors (CN):** number of mutual friends, `|Γ(x) ∩ Γ(y)|`
   - **Adamic-Adar (AA):** sum of `1 / log|Γ(z)|` over mutual friends `z` (rare mutual friends count more)
4. Evaluation: **AUC** and **precision@k**, compared with random guessing (0.233%).

```python
A = adjacency_matrix(known_network)     # who already talked
CN = A @ A                              # number of mutual friends
AA = (A * (1/np.log(degree))) @ A       # rare mutual friends count more
```

## Main results

| Method | AUC | P@50 | P@100 | P@200 | P@500 |
|---|---|---|---|---|---|
| Random guessing | 0.500 | 0.23% | 0.23% | 0.23% | 0.23% |
| Common Neighbors | 0.667 | 6.0% | 3.0% | 2.5% | 2.6% |
| Adamic-Adar | 0.672 | 6.0% | 3.0% | 3.0% | 2.6% |

Probability of connecting later, by number of common friends:

| Common friends | Pairs | Connected later |
|---|---|---|
| 0 | 766,939 | 0.14% |
| 1 | 129,462 | 0.41% |
| 2 | 36,406 | 0.71% |
| 3 | 15,426 | 0.80% |
| 4 or more | 20,828 | 1.24% |

**Conclusion:** *yes, partially.* Pairs with 4+ common friends are about 9× more likely to connect than pairs with none, and the top-ranked pairs are roughly 11 to 26× better than random. Still, only 3 to 6% of the top predictions actually connected, and 55% of the new connections involved users not yet in the network (cold start).

## Figures

All charts are in [`poster_figures/`](poster_figures/):

| File | Content |
|---|---|
| `1_probability_vs_common_friends.png` | Probability of connecting vs. number of common friends |
| `2_roc_curves.png` | ROC curves for CN, AA and random guessing |
| `3_precision_at_k.png` | Precision among the top-k predictions |
| `4_network_example.png` | Example of a correctly predicted connection |
| `results.csv` | Results table |

## How to run

Requires Python 3.9+.

```bash
pip install -r requirements.txt
python friendship_prediction.py
```

Options:

```bash
python friendship_prediction.py --data CollegeMsg.txt.gz   # use a local copy of the dataset
python friendship_prediction.py --split 0.5                # use 50% of messages as the known network
```

If the automatic download fails, download `CollegeMsg.txt.gz` from the [SNAP page](https://snap.stanford.edu/data/CollegeMsg.html) and use `--data`.

The script creates a `poster_figures/` folder with the four charts and `results.csv`.

## Repository structure

```
├── friendship_prediction.py    # full experiment
├── requirements.txt            # Python dependencies
├── poster_figures/             # generated charts and results table
└── README.md
```

## Limitations

- One dataset and one time split.
- Only two simple methods; no machine learning models were compared.
- Cold start: new users cannot be scored by structure-based methods.
- A connection means "sent a message", not necessarily friendship.

## References

1. D. Liben-Nowell and J. Kleinberg, "The link-prediction problem for social networks," *JASIST*, vol. 58, no. 7, pp. 1019–1031, 2007.
2. L. A. Adamic and E. Adar, "Friends and neighbors on the Web," *Social Networks*, vol. 25, no. 3, pp. 211–230, 2003.
3. A. Grover and J. Leskovec, "node2vec: Scalable feature learning for networks," *KDD*, 2016.
4. P. Panzarasa, T. Opsahl and K. M. Carley, "Patterns and dynamics of users' behavior and interaction: Network analysis of an online community," *JASIST*, vol. 60, no. 5, pp. 911–932, 2009.
5. J. Leskovec and A. Krevl, "SNAP Datasets: Stanford large network dataset collection," 2014. http://snap.stanford.edu/data
