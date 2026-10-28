# Random Forest Exercise 2

For this exercise I extended `exercise_rf.py` so it finds a good number of trees and tree depth on its own. Before, you had to pick them by hand.

## How to run

```bash
pip install -r requirements.txt
python exercise_rf.py
```

The plot gets saved to `random_forest_plots/`.

## What I changed

I tried to keep the changes small and stick to the task:

- **Added a grid search.** I wrapped the forest in `GridSearchCV` and tested 1, 5, 10, 25, 50, 100 and 200 trees against every depth from 1 to 15. Each combination is scored with 5-fold cross-validation on the training set only. I didn't want the test set involved in choosing the parameters, otherwise the test accuracy wouldn't really mean anything. The best model is then refit on the whole training set and used for the rest of the script, so the plot shows the chosen values. `--num_trees` and `--tree_depth` are still in the argparse, but the search overrides them now.
- **Fixed a crash.** `plt.cm.get_cmap("RdBu")` doesn't exist anymore in matplotlib 3.9, which is the version in `requirements.txt`, so the script wouldn't run. I changed it to `plt.get_cmap("RdBu")`.

## Results

On my run the search picked **200 trees with a max depth of 7**:

- CV accuracy: 94.67%
- Training accuracy: 98.13%
- Testing accuracy: 93.73%

For comparison, the default (1 tree, depth 1) only got about 83% on the test set and basically just drew a horizontal line through the data. Past a depth of about 5 the CV score barely changes. It sits around 93–94% and even drops slightly at the deepest settings, because deeper trees mostly start fitting the noise.

## Why results will not match others and How to get identical results 

A friend and I ran the same settings and got different numbers. It isn't the noise level itself (`noise=0.23` is the same for both of us). The issue is that nothing in the script fixes a random seed:

1. `make_moons(..., random_state=None)` creates a new random dataset every time you run it, with the noise landing differently. So we were training and testing on different data. This is the main reason.
2. `RandomForestClassifier` has no `random_state`, so the bootstrap samples for each tree are different too, even on the same data.

The train/test split has `random_state=42`, but that doesn't help when the data going into it is already different. This also means the best depth found by the search moves around a bit between runs, usually somewhere between 5 and 8. Since the scores in that range are so close, which one wins comes down to the random data.

To get identical results you'd set both seeds:

```python
data = make_moons(n_samples=1500, noise=0.23, random_state=42)

sk_ensemble.RandomForestClassifier(max_features=feat_train.shape[1], random_state=42)
```

I left the seeds unset in my submission because the template had `random_state=None`. I think that was on purpose, to show how much the results vary between runs.
