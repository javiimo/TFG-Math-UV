import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # (required for older versions of matplotlib)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define a smooth bump function using the standard bump formula
def phi(t):
    """
    Standard bump: exp(-1/(1-t^2)) for |t|<1, and 0 for |t|>=1.
    """
    t = np.asarray(t)
    out = np.zeros_like(t, dtype=float)
    mask = np.abs(t) < 1
    out[mask] = np.exp(-1/(1 - t[mask]**2))
    return out

def bump(x, y):
    """
    Bump function defined on [2,3]x[3,7] with maximum 1 at (2.5,5) and zero on the boundary.
    
    The normalization divides by (phi(0))^2 so that at the center (where u=v=0) the value is 1.
    """
    # Normalize so that (x,y) = (2.5,5) becomes 0 and the endpoints correspond to |u|=1, |v|=1.
    u = (x - 2.5) / 0.5   # u in [-1,1] as x runs from 2 to 3
    v = (y - 5)   / 2.0   # v in [-1,1] as y runs from 3 to 7
    return (phi(u) * phi(v)) / (phi(0)**2)

# Create figure and 3D axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set axis ranges (same as your original code)
ax.set_xlim(0, 5)
ax.set_ylim(0, 8)
ax.set_zlim(0, 1)

# -------------------------------
# Plot the bump function as a smooth surface.
# -------------------------------
# Create a grid on [2,3]x[3,7]
x = np.linspace(2, 3, 100)
y = np.linspace(3, 7, 100)
X, Y = np.meshgrid(x, y)
Z = bump(X, Y)

# Plot the top surface (graph of the bump) with a blue, semi-transparent color.
surf = ax.plot_surface(X, Y, Z, color='blue', alpha=0.5,
                       rstride=4, cstride=4, edgecolor='none')

# -------------------------------
# Add a coarse wireframe overlay with faint red and white lines to highlight the geometry.
# -------------------------------
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=100, color='red', linewidth=0.5, alpha=0.3)
ax.plot_wireframe(X, Y, Z, rstride=100, cstride=10, color='white', linewidth=0.5, alpha=0.3)

# (Optional) Plot the base: the rectangle in the xy-plane.
base_vertices = np.array([
    [2, 3, 0],
    [3, 3, 0],
    [3, 7, 0],
    [2, 7, 0]
])
base = Poly3DCollection([base_vertices], alpha=0.5, edgecolor='black')
base.set_facecolor('blue')
ax.add_collection3d(base)

# -------------------------------
# Add dashed projection curves.
# -------------------------------
# For the xz projection (at y=0), we “project” the bump’s top by taking, for each x, 
# the maximum z (which here occurs at y=5 because the bump is separable).
x_proj = np.linspace(2, 3, 100)
z_proj_x = bump(x_proj, 5)
# Build the dashed line: start at (0,0,0), go to (2,0,0), then follow the bump projection,
# then drop to (3,0,0) and finally extend to (5,0,0).
xz_x = np.concatenate((np.array([0, 2]), x_proj, np.array([3, 5])))
xz_z = np.concatenate((np.array([0, 0]), z_proj_x, np.array([0, 0])))
ax.plot(xz_x, np.zeros_like(xz_x), xz_z, 'k--', alpha=0.7, linewidth=2)

# For the yz projection (at x=0), take for each y the maximum z (occurring at x=2.5).
y_proj = np.linspace(3, 7, 100)
z_proj_y = bump(2.5, y_proj)
yz_y = np.concatenate((np.array([0, 3]), y_proj, np.array([7, 8])))
yz_z = np.concatenate((np.array([0, 0]), z_proj_y, np.array([0, 0])))
ax.plot(np.zeros_like(yz_y), yz_y, yz_z, 'k--', alpha=0.7, linewidth=2)

# -------------------------------
# Add custom bold axis lines and labels (as in your original code)
# -------------------------------
ax.plot([0, 5], [0, 0], [0, 0], 'k-', linewidth=2)  # x-axis
ax.plot([0, 0], [0, 8], [0, 0], 'k-', linewidth=2)  # y-axis
ax.plot([0, 0], [0, 0], [0, 1], 'k-', linewidth=2)  # z-axis

# Remove the default axis labels
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')

# Add custom axis labels at the ends
ax.text(5.7, 0, 0, 'X', fontsize=20)
ax.text(0, 8.5, 0, 'Y', fontsize=20)
ax.text(0, 0, 1.1, '$\\mu_{X\\times Y}(x,y)$', fontsize=20)

# Hide the default axis lines
ax.xaxis.line.set_visible(False)
ax.yaxis.line.set_visible(False)
ax.zaxis.line.set_visible(False)

# Set tick marks and custom tick labels (same as your original code)
ax.set_xticks([2, 3])
ax.set_xticklabels(['a', 'b'])
ax.set_yticks([3, 7])
ax.set_yticklabels(['c', 'd'])
ax.set_zticks([1])
ax.set_zticklabels(['1'])
ax.tick_params(axis='z', which='major', pad=-3)

# Set the aspect ratio with a taller y-axis as before
ax.set_box_aspect([1, 1.6, 1])

# Adjust viewing angle for a better perspective
ax.view_init(elev=20, azim=45)

plt.show()
