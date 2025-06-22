import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Quantifier Functions
# -------------------------------------------------------------------------
def quantifier_all(r):
    return r

def quantifier_at_least_half(r):
    return np.piecewise(r, [r < 0.5, r >= 0.5], [0, 1])

def quantifier_some(r):
    return np.sqrt(r)

def quantifier_majority(r):
    return r**2

def quantifier_window(r):
    conditions = [
        r <= 0.25,
        (r > 0.25) & (r <= 0.75),
        r > 0.75
    ]
    functions = [
        0,
        lambda x: 2*x - 0.5,
        1
    ]
    return np.piecewise(r, conditions, functions)

# -------------------------------------------------------------------------
# Generate Data
# -------------------------------------------------------------------------
r = np.linspace(0, 1, 500)

# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

# Plot each quantifier
ax.plot(r, quantifier_all(r), label='"All" ($Q(r)=r$)', linewidth=2)
ax.plot(r, quantifier_at_least_half(r), label='"At least half"', linewidth=2)
ax.plot(r, quantifier_some(r), label='"Some" ($Q(r)=\sqrt{r}$)', linewidth=2)
ax.plot(r, quantifier_majority(r), label='"The majority" ($Q(r)=r^2$)', linewidth=2)
ax.plot(r, quantifier_window(r), label='"Window-Type"', linewidth=2)

# -------------------------------------------------------------------------
# Formatting Improvements
# -------------------------------------------------------------------------
# Font sizes
title_font = {'fontsize': 18, 'fontweight': 'bold'}
label_font = {'fontsize': 14}
tick_font_size = 12
legend_fontsize = 12

# Titles and labels
ax.set_title('Linguistic Quantifier Functions for OWA Weights', **title_font)
ax.set_xlabel('Proportion of Criteria ($r$)', **label_font)
ax.set_ylabel('Degree of Satisfaction ($Q(r)$)', **label_font)

# Tick labels
ax.tick_params(axis='both', labelsize=tick_font_size)

# Extended limits with a small margin
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

# Keep aspect ratio square for shape clarity
ax.set_aspect('equal', adjustable='box')

# Legend and grid
ax.legend(loc='upper left', fontsize=legend_fontsize, frameon=True)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Layout
plt.tight_layout()

# Display
plt.show()
