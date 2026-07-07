# Gaming-Player Clustering — How Much Structure Is Really There?

An unsupervised-learning case study on online-gaming behaviour.

**Author:** Blagoy Hristov

---

## What this project is

Game studios love to split players into behavioural "personas". This project
asks the harder, more honest question first: **do natural behavioural segments
actually exist in the data at all?** Using ~40k players, three clustering
algorithm families, and internal-validity indices benchmarked against a
**uniform-random null model**, the answer turns out to be *no* — and the
interesting part is *proving* it rigorously and explaining what little signal
does exist.

## Key findings

- The behavioural features are **uncorrelated and near-uniform** (max pairwise
  correlation ≈ 0.009; the PCA scree is flat at ~20% per component).
- **No natural clusters:** K-Means, GMM and DBSCAN all agree. The real-data
  silhouette lies **inside the Monte-Carlo band of a uniform-random null model**
  at every *k*; there is no elbow and no BIC minimum.
- The supplied `EngagementLevel` label is driven **predominantly by two
  features** (`SessionsPerWeek`, `AvgSessionDurationMinutes`). It is invisible to
  full-space clustering (**ARI ≈ 0.003**) but recoverable in its 2-D subspace
  (**ARI ≈ 0.21**, rising to ≈ 0.27 with an engineered feature) — a clear
  demonstration of how uncorrelated noise dimensions swamp a low-dimensional
  signal, and why feature selection matters more than the choice of algorithm.

## Repository layout

```
gaming-player-clustering/
├── gaming_player_clustering.ipynb   # main deliverable — full analysis
├── src/
│   └── clustering_utils.py          # reusable, documented helper module
├── data/
│   └── online_gaming_behavior_dataset.csv
├── requirements.txt                 # pinned dependencies
└── README.md
```

## How to run

```bash
python -m pip install -r requirements.txt
jupyter lab                # open the notebook, then:
#   Kernel ▸ Restart Kernel and Run All Cells
```

Or headless:

```bash
jupyter nbconvert --to notebook --execute gaming_player_clustering.ipynb
```

Every stochastic step is seeded with `random_state = 42`, so results are fully
reproducible.

## Data

The dataset is the public **"Predict Online Gaming Behavior"** dataset from
Kaggle (full citation in Section 12 of the notebook). It ships with this
repository under `data/` for convenience and reproducibility.
