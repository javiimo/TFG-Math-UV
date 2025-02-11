import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Sustainable Building Design Problem
#
# A design team wants to optimize the following two objectives:
#   - x: Natural Lighting Score (normalized; higher is better)
#   - y: Energy Efficiency Score (normalized; higher is better)
#
# Trade-off: Increasing natural lighting (via larger windows) can reduce 
# energy efficiency (since larger windows may lead to higher heat loss), while 
# smaller windows improve energy efficiency but limit natural lighting.
#
# The feasible region (light blue) represents realistic design combinations.
# The red curve is the Pareto frontier: the set of non-dominated designs.
# Additionally, 5 specific design proposals (black circles) are shown on the frontier.
# =============================================================================

# Define the lower and upper boundary functions.
def lower(x):
    """
    Lower boundary function:
    Represents the minimal achievable energy efficiency for a given natural lighting score.
    """
    return 1 + 0.2 * np.sin(3 * x)

def upper(x):
    """
    Upper boundary function:
    Represents the maximal achievable energy efficiency for a given natural lighting score.
    Modeled as a downward opening parabola with its peak at (2, 5).
    """
    return 5 - ((x - 2)**2) / 2

# Generate a dense grid for x (natural lighting score).
x_all = np.linspace(0, 6, 6000)

# Identify x values where the feasible region exists (i.e., lower(x) <= upper(x)).
mask = lower(x_all) <= upper(x_all)
x_feasible = x_all[mask]
y_lower = lower(x_feasible)
y_upper = upper(x_feasible)

plt.figure(figsize=(8, 6))

# Fill the feasible region (the realistic design choices).
plt.fill_between(x_feasible, y_lower, y_upper, color='lightblue', alpha=0.7,
                 label='Feasible Design Region')

# =============================================================================
# Pareto Frontier Identification:
#
# For each natural lighting score x, the best achievable energy efficiency is 
# given by the upper boundary. However, for x < 2, the design at (2,5) dominates 
# (since both objectives are lower there). Thus, the non-dominated designs are 
# represented by the portion of the upper boundary for x >= 2.
# =============================================================================
mask_pareto = x_feasible >= 2
x_pareto = x_feasible[mask_pareto]
y_pareto = upper(x_pareto)

plt.plot(x_pareto, y_pareto, 'r-', linewidth=2, label='Pareto Frontier')

# =============================================================================
# Proposed Design Points:
#
# The design team has identified 5 promising design options on the Pareto frontier.
# These points are not uniformly distributed along the frontier. They represent 
# specific trade-offs in the sustainable building design.
# =============================================================================
# Define non-uniform x-values for the proposed designs (ensure they lie in the region x>=2)
x_designs = np.array([2.1, 2.7, 3.4, 4.1, 4.3])
y_designs = upper(x_designs)  # These points lie exactly on the Pareto frontier

plt.plot(x_designs, y_designs, 'ko', markersize=8, label='Proposed Designs')

# Label the axes with the real-world variable meanings.
plt.xlabel('Natural Lighting Score')
plt.ylabel('Energy Efficiency Score')
plt.title('Sustainable Building Design')
plt.legend()
plt.grid(True)

# -----------------------------------------------------------------------------
# Adjust plot limits to add margins so the region does not collide with the borders.
x_margin = 0.1 * (np.max(x_feasible) - np.min(x_feasible))
y_margin = 0.1 * (np.max(y_upper) - np.min(y_lower))
plt.xlim(np.min(x_feasible) - x_margin, np.max(x_feasible) + x_margin)
plt.ylim(np.min(y_lower) - y_margin, np.max(y_upper) + y_margin)

plt.show()
