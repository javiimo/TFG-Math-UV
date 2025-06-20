import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# --- Setup for a professional, LaTeX-like appearance ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.family": "serif",  # Use a standard serif font
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (12, 6),
})

# Create the figure and a 1x2 grid of subplots
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

# --- Plot 1: Incorrect Definitions ---
ax1.set_title('Incorrect Definitions of "Tall"')
ax1.set_xlabel('Height (m)')
ax1.set_ylabel('Membership degree $\mu(h)$')
ax1.set_xlim(1.4, 2.1)
ax1.set_ylim(-0.05, 1.1)
ax1.set_xticks(np.arange(1.4, 2.2, 0.1))
ax1.set_yticks(np.arange(0, 1.1, 0.2))

# --- Define and plot two different non-monotonic functions ---

# 1. First Non-monotonic function (single dip)
h_points1 = [1.5, 1.6, 1.75, 1.85, 2.0]
mu_points1 = [0, 0.2, 0.8, 0.55, 1.0]
h_smooth1 = np.linspace(min(h_points1), max(h_points1), 200)
spline1 = make_interp_spline(h_points1, mu_points1, k=3)
mu_non_monotonic1 = np.clip(spline1(h_smooth1), 0, 1)
ax1.plot(h_smooth1, mu_non_monotonic1, label='Non-monotonic 1', color='crimson', linewidth=2.5, linestyle='--')
# Extend lines to edges
ax1.plot([1.4, 1.5], [0,0], color='crimson', linewidth=2.5, linestyle='--')
ax1.plot([2.0, 2.1], [1,1], color='crimson', linewidth=2.5, linestyle='--')

# 2. Second Non-monotonic function (up-down-up wiggle)
h_points2 = [1.5, 1.6, 1.7, 1.85, 2.0]
mu_points2 = [0, 0.4, 0.2, 0.9, 1.0]
h_smooth2 = np.linspace(min(h_points2), max(h_points2), 200)
spline2 = make_interp_spline(h_points2, mu_points2, k=3)
mu_non_monotonic2 = np.clip(spline2(h_smooth2), 0, 1)
ax1.plot(h_smooth2, mu_non_monotonic2, label='Non-monotonic 2', color='darkorange', linewidth=2.5, linestyle='-.')
# Extend lines to edges
ax1.plot([1.4, 1.5], [0,0], color='darkorange', linewidth=2.5, linestyle='-.')
ax1.plot([2.0, 2.1], [1,1], color='darkorange', linewidth=2.5, linestyle='-.')

# 3. Add the thick "Consensus" lines on top
ax1.plot([1.4, 1.5], [0, 0], color='black', linewidth=4, zorder=10)
ax1.plot([2.0, 2.1], [1, 1], color='black', linewidth=4, zorder=10)

ax1.legend(loc='upper left')

# --- Plot 2: Plausible Definitions (Unchanged) ---
ax2.set_title('Plausible Definitions of "Tall"')
ax2.set_xlabel('Height (m)')
ax2.set_xlim(1.4, 2.1)
ax2.set_ylim(-0.05, 1.1)
ax2.set_xticks(np.arange(1.4, 2.2, 0.1))
ax2.set_yticks(np.arange(0, 1.1, 0.2))

h_range_full = np.linspace(1.4, 2.1, 500)
h_min, h_max = 1.5, 2.0
h_norm = np.clip((h_range_full - h_min) / (h_max - h_min), 0, 1)

ax2.plot(h_range_full, h_norm, label='Linear', color='darkgreen', linewidth=2.5, linestyle='-')
ax2.plot(h_range_full, h_norm**2, label='Quadratic', color='orangered', linewidth=2.5, linestyle='--')
ax2.plot(h_range_full, np.sqrt(h_norm), label='Square Root', color='purple', linewidth=2.5, linestyle=':')

h_steps = np.linspace(1.5, 2.0, 10)
mu_values = np.linspace(0, 1.0, 10)
ax2.step(h_steps, mu_values, where='post', label='Step-wise', color='saddlebrown', linewidth=2.5, linestyle='-.')

ax2.plot([1.4, 1.5], [0, 0], color='black', linewidth=4, zorder=10)
ax2.plot([2.0, 2.1], [1, 1], color='black', linewidth=4, zorder=10)

ax2.legend(loc='upper left')

# --- Final Adjustments & Save ---
# plt.savefig('fuzzy_tall_definitions_final_v3.pdf', bbox_inches='tight', dpi=300)
plt.show()