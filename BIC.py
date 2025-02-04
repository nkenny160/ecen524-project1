import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from matplotlib.patches import Ellipse
from scipy import linalg

//Reference: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html

#  Load your dataset
data = pd.read_csv("aligned_trajectory_data_z.csv")

#  Use only 'Aligned_X' and 'Aligned_Y' as features
X = data.iloc[:, :2].values  # Select first two columns

#  Define a function to compute BIC for model selection
def gmm_bic_score(estimator, X):
    return -estimator.bic(X)  # Make negative since GridSearchCV maximizes scores

#  Set range of GMM components and covariance types
param_grid = {
    "n_components": range(1, 7),  # Try between 1 and 6 components
    "covariance_type": ["spherical", "tied", "diag", "full"],
}

#  Perform model selection using BIC
grid_search = GridSearchCV(
    GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
)
grid_search.fit(X)

#  Convert GridSearch results to DataFrame
df = pd.DataFrame(grid_search.cv_results_)[
    ["param_n_components", "param_covariance_type", "mean_test_score"]
]
df["mean_test_score"] = -df["mean_test_score"]  # Convert back to positive BIC values
df = df.rename(
    columns={
        "param_n_components": "Number of components",
        "param_covariance_type": "Type of covariance",
        "mean_test_score": "BIC score",
    }
)

#  Plot BIC scores for different models
sns.catplot(
    data=df,
    kind="bar",
    x="Number of components",
    y="BIC score",
    hue="Type of covariance",
)
plt.title("GMM Model Selection using BIC")
plt.show()

#  Fit the best model and visualize clusters
best_gmm = grid_search.best_estimator_
Y_ = best_gmm.predict(X)

fig, ax = plt.subplots()
color_iter = sns.color_palette("tab10", best_gmm.n_components)[::-1]

for i, (mean, cov, color) in enumerate(
    zip(best_gmm.means_, best_gmm.covariances_, color_iter)
):
    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

    # Compute the covariance ellipse
    v, w = linalg.eigh(cov)
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180.0 * angle / np.pi
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ellipse = Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
    ellipse.set_clip_box(fig.bbox)
    ellipse.set_alpha(0.5)
    ax.add_artist(ellipse)

plt.title(
    f"Selected GMM: {grid_search.best_params_['covariance_type']} model, "
    f"{grid_search.best_params_['n_components']} components"
)
plt.axis("equal")
plt.show()
