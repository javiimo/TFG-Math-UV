import numpy as np
import matplotlib.pyplot as plt

# --- Plot Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.autolayout'] = True

# --- Quantifier and Parameters ---
n = 5  # Number of criteria

def Q(r):
    return r**2

# --- Data Generation ---
r_values = np.linspace(0, 1, 100)
q_values = Q(r_values)
proportions = np.arange(0, n + 1) / n
q_at_proportions = Q(proportions)
weights = q_at_proportions[1:] - q_at_proportions[:-1]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 8))

# Quantifier curve and points
ax.plot(r_values, q_values, label=r'$Q(r) = r^2$', linewidth=2.5)
ax.plot(proportions, q_at_proportions, 'o', markersize=8, zorder=5)

# Annotate all weights w1...w5
dx = 0.03  # horizontal offset for arrows/labels
for j in range(1, n + 1):
    x0, x1 = (j - 1) / n, j / n
    y0, y1 = Q(x0), Q(x1)
    mid_y = (y0 + y1) / 2
    
    # Dashed guide lines
    ax.plot([x0, x0], [0, y0], linestyle='--', linewidth=1, alpha=0.7)
    ax.plot([x1, x1], [0, y1], linestyle='--', linewidth=1)
    ax.plot([0, x1], [y1, y1], linestyle='--', linewidth=1)
    
    # Double-headed arrow for weight
    ax.annotate(
        '',
        xy=(x1 + dx, y0),
        xytext=(x1 + dx, y1),
        arrowprops=dict(arrowstyle='<->', lw=1.3, color='red')
    )
    # Label inside the plot
    ax.text(
        x1 + dx * 1.2,
        mid_y,
        f'$w_{{{j}}}=Q({j}/{n})-Q({j-1}/{n})$',
        va='center',
        fontsize=11,
        color='red'
    )

# --- Labels and Titles ---
ax.set_xlabel('Proportion of criteria, $r$', fontsize=13)
ax.set_ylabel('Quantifier value, $Q(r)$', fontsize=13)
ax.set_title('Determining OWA Weights from a Linguistic Quantifier ($n=5$)', fontsize=15, pad=15)

# Ticks and limits
ax.set_xticks(proportions)
ax.set_xticklabels([f'${j}/{n}$' for j in range(n + 1)])
ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.set_xlim(-0.02, 1.3)
ax.set_ylim(-0.02, 1.02)

# Legend, grid, and aspect
ax.legend(loc='upper left')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_aspect('equal', adjustable='box')

plt.show()

# Print calculated weights
print("Calculated Weights:")
for i, w in enumerate(weights, 1):
    print(f"w_{i} = {w:.4f}")
