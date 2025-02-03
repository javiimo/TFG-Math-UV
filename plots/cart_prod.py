import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# Create figure and 3D axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set axis ranges
ax.set_xlim(0, 5)
ax.set_ylim(0, 8)
ax.set_zlim(0, 1)

# Define vertices for the rectangular prism
vertices = np.array([
    # Bottom face (z=0)
    [[2, 3, 0], [2, 7, 0], [3, 7, 0], [3, 3, 0]],
    # Top face (z=1)
    [[2, 3, 1], [2, 7, 1], [3, 7, 1], [3, 3, 1]],
    # Front face (y=3)
    [[2, 3, 0], [3, 3, 0], [3, 3, 1], [2, 3, 1]],
    # Back face (y=7)
    [[2, 7, 0], [3, 7, 0], [3, 7, 1], [2, 7, 1]],
    # Left face (x=2)
    [[2, 3, 0], [2, 7, 0], [2, 7, 1], [2, 3, 1]],
    # Right face (x=3)
    [[3, 3, 0], [3, 7, 0], [3, 7, 1], [3, 3, 1]]
])

# Create polygons and add to plot
poly3d = Poly3DCollection(vertices, alpha=0.5, edgecolor='black')
poly3d.set_facecolor('blue')
ax.add_collection3d(poly3d)

# Add XZ projection (at y=0)
xz_x = np.array([0, 2, 2, 3, 3, 5])
xz_z = np.array([0, 0, 1, 1, 0, 0])
ax.plot(xz_x, [0]*len(xz_x), xz_z, 'k--', alpha=0.7, linewidth=2)

# Add YZ projection (at x=0)
yz_y = np.array([0, 3, 3, 7, 7, 8])
yz_z = np.array([0, 0, 1, 1, 0, 0])
ax.plot([0]*len(yz_y), yz_y, yz_z, 'k--', alpha=0.7, linewidth=2)

# Add custom bold axis lines
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
ax.text(0, 0, 1.1, '$\mu_{X\\times Y}(x,y)$', fontsize=20)

# Remove default axis lines
ax.xaxis.line.set_visible(False)
ax.yaxis.line.set_visible(False)
ax.zaxis.line.set_visible(False)



# Set ticks
ax.set_xticks([2,3]) 
ax.set_xticklabels(['a', 'b'])
ax.set_yticks([3,7])  
ax.set_yticklabels(['c', 'd'])
ax.set_zticks([ 1])
ax.set_zticklabels([ '1'])

# Move z ticks to front of axis
ax.tick_params(axis='z', which='major', pad=-3)

# Set the aspect ratio with a much taller z-axis
ax.set_box_aspect([1, 1.6, 1])

# Adjust viewing angle for better perspective
ax.view_init(elev=20, azim=45)

plt.show()