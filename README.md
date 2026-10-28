# Random Forest Exercise – Hyperparameter Search

`exercise_rf.py` trains a random forest on a 2-D two-moons point cloud and plots its decision boundary. The script searches automatically for the best number of trees and tree depth.

## Running it

```bash
pip install -r requirements.txt
python exercise_rf.py
```

The plot is saved to `random_forest_plots/`.

## What was changed

1. **Hyperparameter search.** A `GridSearchCV` replaces the fixed forest settings. It tries every combination of:
   - number of trees: 1, 5, 10, 25, 50, 100, 200
   - max tree depth: 1 to 15

   It scores each combination with 5-fold cross-validation on the **training data only**, so the test set plays no part in the choice. The best model is then refit on the full training set and used for testing and plotting. The best values and their CV accuracy are printed, and the plot shows them.
   The `--num_trees` and `--tree_depth` arguments still exist, but the search result overrides them.
2. **Import.** Added `GridSearchCV` from `sklearn.model_selection`.
3. **Bug fix.** `plt.cm.get_cmap("RdBu")` became `plt.get_cmap("RdBu")`. The old call was removed in matplotlib 3.9, the version pinned in `requirements.txt`, and the script crashed without this fix.

## Example result

On one run, the search picked **200 trees** with **max depth 7**:

| | Accuracy |
|---|---|
| Cross-validation (best) | 94.67% |
| Training | 98.13% |
| Testing | 93.73% |

For comparison, the template's default of 1 tree at depth 1 reached about 83% test accuracy.

## Why results differ between runs (and between people)

Two people running the same script with the same number of trees and depth can get different accuracies. The noise level (`noise=0.23`) is the same for everyone, but the script never fixes its random seeds, so two parts of it change on every run:

1. **The data.** `make_moons(..., random_state=None)` generates a new set of 1500 points each time, with the noise falling differently. Each run trains and tests on a different dataset. This is the main cause.
2. **The forest.** `RandomForestClassifier` has no `random_state`. Each tree trains on a random bootstrap resample of the training data, so the trees differ even when the data is the same.

The train/test split is fixed (`random_state=42`), but that doesn't help when the data going into it changes. Because the data varies, the chosen depth also varies slightly between runs, usually between 6 and 10.

### Making results reproducible

Fix both seeds:

```python
data = make_moons(n_samples=1500, noise=0.23, random_state=42)

sk_ensemble.RandomForestClassifier(max_features=feat_train.shape[1], random_state=42)
```

With these seeds, everyone gets identical results, as long as they use the same scikit-learn version.

The template leaves `random_state=None`, probably on purpose, to show how results vary from run to run. That's why the submitted code doesn't fix the seeds.
