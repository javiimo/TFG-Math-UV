import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plots

# --- Define the T-norm functions ---

def product_tnorm(x, y):
    """Product t-norm: T_P(x,y) = x*y"""
    return x * y

def minimum_tnorm(x, y):
    """Minimum t-norm: T_M(x,y) = min(x,y)"""
    return np.minimum(x, y)

def lukasiewicz_tnorm(x, y):
    """Lukasiewicz t-norm: T_L(x,y) = max(0, x+y-1)"""
    return np.maximum(0, x + y - 1)

def drastic_tnorm_book(x, y):
    """Drastic t-norm (as defined in Klement et al., page 19):
       T_D(x,y) = 0 if (x,y) in [0,1[^2, else min(x,y)
    """
    # Ensure x and y are numpy arrays for element-wise operations and broadcasting
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Broadcast x and y to the same shape (e.g., when X, Y are from meshgrid)
    bx, by = np.broadcast_arrays(x_arr, y_arr)

    # Initialize with min(x,y) for the 'otherwise' case
    # This handles the cases where x=1 or y=1
    output = np.minimum(bx, by)

    # Condition for being 0: both x < 1 AND y < 1
    condition_zero = np.logical_and(bx < 1, by < 1)
    output[condition_zero] = 0

    return output

# --- Prepare data for plotting ---
N = 100  # Increased N for smoother plots, especially for Drastic T-norm visualization
x_vals = np.linspace(0, 1, N)
y_vals = np.linspace(0, 1, N)
X, Y = np.meshgrid(x_vals, y_vals)

# List of t-norms and their names
tnorms = [
    (product_tnorm, "Product T-norm ($T_P$)"),
    (minimum_tnorm, "Minimum T-norm ($T_M$)"),
    (lukasiewicz_tnorm, "Łukasiewicz T-norm ($T_L$)"),
    (drastic_tnorm_book, "Drastic T-norm ($T_D$)") # Removed (Book Def.) for brevity
]

# --- Generate 2D Colored Plots (Heatmaps/Contours) ---
fig2d, axes2d = plt.subplots(2, 2, figsize=(11, 10)) # Slightly adjusted size
axes2d = axes2d.ravel()

for i, (tnorm_func, title) in enumerate(tnorms):
    Z = tnorm_func(X, Y)
    ax = axes2d[i]

    # Heatmap
    heatmap = ax.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', vmin=0, vmax=1, aspect='auto')

    # Contour lines
    # For Drastic T-norm, only plot a contour at level 0 and a very high one if needed,
    # as other levels might not be meaningful or visually clean.
    if title == "Drastic T-norm ($T_D$)":
        levels = [0.0001, 0.5, 0.9999] # Levels to show for Drastic, avoiding exact 0 and 1 for clarity
        # Add specific contours for the edges of the zero region
        ax.contour(X, Y, Z, levels=[0.001], colors='black', linewidths=0.8, linestyles='dashed')
        ax.contour(X, Y, Z, levels=[np.finfo(float).eps], colors='black', linewidths=0.8) # Near zero
    else:
        levels = np.linspace(0.1, 0.9, 9) # 9 levels between 0.1 and 0.9
        ax.contour(X, Y, Z, levels=levels, colors='white', linewidths=0.7, alpha=0.7)

    fig2d.colorbar(heatmap, ax=ax, label='T-norm value', fraction=0.046, pad=0.04) # Adjust colorbar size
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_aspect('equal', adjustable='box')


fig2d.suptitle("2D Colored Plots with Contour Lines of Basic T-norms", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
plt.savefig("tnorms_2D_plots.png", dpi=300)
print("Saved 2D plots to tnorms_2D_plots.png")
plt.show()


# --- Generate 3D Surface Plots ---
# Make the figure taller
fig3d = plt.figure(figsize=(12, 14)) # Increased height
fig3d.suptitle("3D Surface Plots of Basic T-norms", fontsize=16)

for i, (tnorm_func, title) in enumerate(tnorms):
    Z = tnorm_func(X, Y) # Recompute Z for each t-norm
    ax = fig3d.add_subplot(2, 2, i + 1, projection='3d')

    # For Drastic t-norm, we can try to make the surface less "peaky"
    # by plotting it in parts, but this is complex and plot_surface
    # is generally used. The current definition should be okay with higher N.
    # A higher N helps the rendering engine to better capture the discontinuity.
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', vmin=0, vmax=1, rstride=2, cstride=2, antialiased=True)

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('T(x,y)')
    ax.set_zlim(0, 1)
    ax.view_init(elev=25, azim=-135) # Adjusted view angle for potentially better Drastic view

# Adjust spacing between subplots
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90, wspace=0.15, hspace=0.25)
plt.savefig("tnorms_3D_plots.png", dpi=300)
print("Saved 3D plots to tnorms_3D_plots.png")
plt.show()