import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from sklearn.cluster import KMeans

# ----------------------------- Rebuild everything due to kernel reset -----------------------------
np.random.seed(42)
n_per_cluster = 150
mean1, mean2, mean3 = [0, 0], [3, 3], [6, 0]
cov = [[1.2, 0.8], [0.8, 1.2]]

data = np.vstack([
    np.random.multivariate_normal(mean1, cov, n_per_cluster),
    np.random.multivariate_normal(mean2, cov, n_per_cluster),
    np.random.multivariate_normal(mean3, cov, n_per_cluster)
])

# ---------- Fuzzy-C-Means wrapper ----------
def fuzzy_cmeans(data_T, c=3, m=2, error=1e-5, maxiter=300):
    try:
        import skfuzzy as fuzz
        cntr, u, *_ = fuzz.cluster.cmeans(
            data_T, c=c, m=m, error=error, maxiter=maxiter, seed=42
        )
        return cntr, u
    except ModuleNotFoundError:
        N = data_T.shape[1]
        u = np.random.dirichlet(np.ones(c), size=N).T
        x = data_T
        for _ in range(maxiter):
            u_prev = u.copy()
            um = u ** m
            centers = (um @ x.T) / np.sum(um, axis=1, keepdims=True)
            dist = np.linalg.norm(x[None, :, :] - centers[:, :, None], axis=1)
            dist = np.fmax(dist, 1e-9)
            tmp = dist ** (2 / (m - 1))
            u = 1 / tmp
            u /= np.sum(u, axis=0, keepdims=True)
            if np.linalg.norm(u - u_prev) < error:
                break
        return centers, u

# ---------- Pastel palette ----------
hex_pastels = ["#ffb3ba", "#baffc9", "#bae1ff"]
rgb_pastels = np.array([to_rgb(h) for h in hex_pastels])

# ---------- Prepare grid for background shading ----------
xmin, ymin = data.min(axis=0) - 1
xmax, ymax = data.max(axis=0) + 1
gx, gy = np.meshgrid(
    np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300)
)
grid = np.c_[gx.ravel(), gy.ravel()]

# ---------- Setup figure ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=110)
m_values = [1, 1.5, 2]
titles = [r"$m = 1$  (K-Means)", r"$m = 1.5$", r"$m = 2$"]

for ax, m, title in zip(axes, m_values, titles):
    if m == 1:
        # K-Means
        kmeans = KMeans(n_clusters=3, random_state=42).fit(data)
        centers = kmeans.cluster_centers_
        labels = kmeans.predict(data)
        labels_grid = kmeans.predict(grid)
        
        # Sort clusters by x-coordinate for consistent coloring
        sort_indices = np.argsort(centers[:, 0])
        centers = centers[sort_indices]
        # Create a remap array for the labels
        remap = np.argsort(sort_indices)
        labels = remap[labels]
        labels_grid = remap[labels_grid]
        
        data_colors = rgb_pastels[labels]
        grid_colors = rgb_pastels[labels_grid]
    else:
        centers, u = fuzzy_cmeans(data.T, c=3, m=m)

        # Sort clusters by x-coordinate for consistent coloring
        sort_indices = np.argsort(centers[:, 0])
        centers = centers[sort_indices]
        u = u[sort_indices]
        
        dist = np.linalg.norm(grid[:, None, :] - centers[None, :, :], axis=2).T
        dist = np.fmax(dist, 1e-12)
        tmp = dist ** (2 / (m - 1))
        u_grid = 1 / tmp
        u_grid /= np.sum(u_grid, axis=0, keepdims=True)
        grid_colors = u_grid.T @ rgb_pastels
        data_colors = (u.T @ rgb_pastels)

    ax.imshow(
        grid_colors.reshape(gx.shape[0], gx.shape[1], 3),
        origin="lower", extent=(xmin, xmax, ymin, ymax)
    )
    ax.scatter(
        data[:, 0], data[:, 1], c=data_colors,
        edgecolor="k", linewidth=0.4, s=24
    )
    ax.scatter(
        centers[:, 0], centers[:, 1],
        marker="X", s=100, c="white", edgecolor="k", linewidths=1.3
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

plt.tight_layout()
plt.show()