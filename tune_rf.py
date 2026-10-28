"""Automatically find the best random forest hyper-parameters.

Searches over the number of trees and the maximum tree depth using 5-fold
cross-validation on the TRAINING data only, then evaluates the winner once on
the held-out test data and produces the usual decision-boundary plot via
exercise_rf.main().

Usage:
    python tune_rf.py                # default seed 42
    python tune_rf.py --seed 7       # different data / forest seed
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.ensemble as sk_ensemble
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

import exercise_rf

# The values we try. Every combination is tested (8 x 15 = 120 models x 5 folds).
NUM_TREES = [1, 2, 5, 10, 20, 50, 100, 200]
TREE_DEPTHS = list(range(1, 16))


def tune(seed: int, result_dir: str):
    os.makedirs(result_dir, exist_ok=True)

    # 1. Same data and same train/test split as exercise_rf.py
    features, labels = make_moons(n_samples=1500, noise=0.23, random_state=seed)
    feat_train, feat_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.5, random_state=42
    )

    # 2. Grid search with 5-fold cross-validation on the training set.
    #    The test set is never looked at while choosing hyper-parameters.
    forest = sk_ensemble.RandomForestClassifier(
        max_features=feat_train.shape[1], random_state=seed
    )
    search = GridSearchCV(
        forest,
        param_grid={"max_depth": TREE_DEPTHS, "n_estimators": NUM_TREES},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        scoring="accuracy",
        n_jobs=-1,
    )
    print(f"Testing {len(TREE_DEPTHS) * len(NUM_TREES)} combinations with 5-fold CV...")
    search.fit(feat_train, labels_train)

    best_depth = search.best_params_["max_depth"]
    best_trees = search.best_params_["n_estimators"]
    i = search.best_index_
    cv_mean = search.cv_results_["mean_test_score"][i]
    cv_std = search.cv_results_["std_test_score"][i]

    # 3. Top 5 combinations, so you can see how close the runners-up are
    print("\nTop 5 (by mean cross-validation accuracy):")
    order = np.argsort(-search.cv_results_["mean_test_score"])[:5]
    for j in order:
        p = search.cv_results_["params"][j]
        print(
            f"  trees={p['n_estimators']:>3}  depth={p['max_depth']:>2}  "
            f"CV acc={search.cv_results_['mean_test_score'][j]:.2%} "
            f"± {search.cv_results_['std_test_score'][j]:.2%}"
        )

    print(f"\nBest: num_trees={best_trees}, tree_depth={best_depth}")
    print(f"Cross-validation accuracy: {cv_mean:.2%} ± {cv_std:.2%}")

    # 4. Heatmap of CV accuracy for every combination
    scores = search.cv_results_["mean_test_score"].reshape(
        len(TREE_DEPTHS), len(NUM_TREES)
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(scores, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(NUM_TREES)), NUM_TREES)
    ax.set_yticks(range(len(TREE_DEPTHS)), TREE_DEPTHS)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Max. tree depth")
    ax.set_title("5-fold cross-validation accuracy (training data)")
    for r in range(scores.shape[0]):
        for c in range(scores.shape[1]):
            ax.text(c, r, f"{scores[r, c]:.1%}", ha="center", va="center",
                    fontsize=6, color="white" if scores[r, c] < scores.max() - 0.03 else "black")
    ax.add_patch(plt.Rectangle(
        (NUM_TREES.index(best_trees) - 0.5, TREE_DEPTHS.index(best_depth) - 0.5),
        1, 1, fill=False, edgecolor="red", linewidth=2))
    fig.colorbar(im, ax=ax, label="Accuracy")
    fig.tight_layout()
    heatmap_path = os.path.join(result_dir, "rf_grid_search_heatmap.png")
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved as {heatmap_path}")

    # 5. Train the winner on all training data, evaluate once on the test set,
    #    and make the standard decision-boundary plot.
    print("\nFinal run with the best hyper-parameters:")
    exercise_rf.main(True, result_dir, best_trees, best_depth, seed)

    return best_trees, best_depth


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    parser = argparse.ArgumentParser(description="Grid search for the random forest exercise")
    parser.add_argument("--seed", type=int, default=42, help="Seed for data and forest.")
    parser.add_argument(
        "--result_dir",
        type=str,
        default=os.path.join(script_dir, "random_forest_plots"),
        help="Directory for plots.",
    )
    args = parser.parse_args()
    tune(args.seed, args.result_dir)
