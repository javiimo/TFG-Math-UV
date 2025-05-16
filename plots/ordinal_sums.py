import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define basic t-norms
def t_min(x, y):
    return np.minimum(x, y)

def t_prod(x, y):
    return x * y

def t_lukas(x, y):
    return np.maximum(0, x + y - 1)

# Define the ordinal sum t-norm
def ordinal_sum_tnorm(x, y, active_regions_with_tnorms):
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Always work with arrays for assignment; if scalar, promote to 0-dim array
    scalar_input = False
    if x_arr.shape == () and y_arr.shape == ():
        scalar_input = True
        x_arr = np.array([x_arr])
        y_arr = np.array([y_arr])

    z_val = t_min(x_arr, y_arr)

    for (a_alpha, e_alpha), t_norm_alpha in active_regions_with_tnorms:
        mask = (x_arr >= a_alpha) & (x_arr <= e_alpha) & \
               (y_arr >= a_alpha) & (y_arr <= e_alpha)

        if np.any(mask):
            x_masked_elements = x_arr[mask]
            y_masked_elements = y_arr[mask]

            x_scaled = (x_masked_elements - a_alpha) / (e_alpha - a_alpha)
            y_scaled = (y_masked_elements - a_alpha) / (e_alpha - a_alpha)

            t_alpha_result = t_norm_alpha(x_scaled, y_scaled)
            rescaled_output = a_alpha + (e_alpha - a_alpha) * t_alpha_result

            z_val[mask] = rescaled_output

    if scalar_input:
        return z_val[0]
    return z_val

# --- Plotting ---
active_regions = [
    ((0.1, 0.4), t_prod),
    ((0.6, 0.9), t_lukas)
]

mesh_density = 100
x_vals_mesh = np.linspace(0, 1, mesh_density)
y_vals_mesh = np.linspace(0, 1, mesh_density)
X, Y = np.meshgrid(x_vals_mesh, y_vals_mesh)

Z = ordinal_sum_tnorm(X, Y, active_regions)

fig = plt.figure(figsize=(13, 11))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the ordinal sum t-norm
surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                       edgecolor='black', 
                       linewidth=0.15,  
                       alpha=0.85,      
                       rcount=mesh_density-1, 
                       ccount=mesh_density-1,
                       antialiased=True,
                       shade=True, zorder=1)

# 1. Outline the active squares on the XY plane (z=0)
for (a_alpha, e_alpha), t_norm_func in active_regions:
    square_x = [a_alpha, e_alpha, e_alpha, a_alpha, a_alpha]
    square_y = [a_alpha, a_alpha, e_alpha, e_alpha, a_alpha]
    ax.plot(square_x, square_y, np.zeros_like(square_x) - 0.01, color='red', linestyle='--', linewidth=2.5, zorder=5) # slightly below z=0

    # Plot contours of active regions ON THE SURFACE
    edge_density = 30
    # Edge 1: (x, a_alpha)
    w = 1
    edge_x_vals = np.linspace(a_alpha, e_alpha, edge_density)
    ax.plot(edge_x_vals, np.full_like(edge_x_vals, a_alpha), 
            ordinal_sum_tnorm(edge_x_vals, np.full_like(edge_x_vals, a_alpha), active_regions), 
            color='black', linewidth=w, zorder=6)
    # Edge 2: (e_alpha, y)
    edge_y_vals = np.linspace(a_alpha, e_alpha, edge_density)
    ax.plot(np.full_like(edge_y_vals, e_alpha), edge_y_vals, 
            ordinal_sum_tnorm(np.full_like(edge_y_vals, e_alpha), edge_y_vals, active_regions), 
            color='black', linewidth=w, zorder=6)
    # Edge 3: (x, e_alpha)
    ax.plot(edge_x_vals[::-1], np.full_like(edge_x_vals, e_alpha), 
            ordinal_sum_tnorm(edge_x_vals[::-1], np.full_like(edge_x_vals, e_alpha), active_regions), 
            color='black', linewidth=w, zorder=6)
    # Edge 4: (a_alpha, y)
    ax.plot(np.full_like(edge_y_vals, a_alpha), edge_y_vals[::-1], 
            ordinal_sum_tnorm(np.full_like(edge_y_vals, a_alpha), edge_y_vals[::-1], active_regions), 
            color='black', linewidth=w, zorder=6)


# 2. Plot the main diagonal y=x on the XY plane (z=0)
ax.plot(x_vals_mesh, x_vals_mesh, np.zeros_like(x_vals_mesh) -0.01 , color='blue', linestyle=':', linewidth=2.5, label='Diagonal y=x (base)', zorder=4) # slightly below z=0

# 3. Highlight the intervals [a_alpha, e_alpha] on this base diagonal
for i, ((a_alpha, e_alpha), _) in enumerate(active_regions):
    diag_segment_x = np.array([a_alpha, e_alpha])
    label = f'Active Interval' if i == 0 else None # Label only once for clarity
    ax.plot(diag_segment_x, diag_segment_x, np.full_like(diag_segment_x, -0.005), 
            color='magenta', linewidth=2.5, zorder=5, label=label)

# 4. Diagonal of the cube [0,1]^3 (dotted line from (0,0,0) to (1,1,1))
ax.plot([0, 1], [0, 1], [0, 1], color='dimgray', linestyle=':', linewidth=1.5, label='Cube Diagonal (x=y=z)', zorder=3)

# 5. Diagonal on the surface T(x,x) (continuous line)
x_diag_vals = np.linspace(0, 1, mesh_density) # Use the same dense x_vals for smoothness
Z_diag_on_surface = np.array([ordinal_sum_tnorm(val, val, active_regions) for val in x_diag_vals])
ax.plot(x_diag_vals, x_diag_vals, Z_diag_on_surface, color='cyan', linewidth=3, label='Surface Diagonal T(x,x)', zorder=7)


# Add text annotations for the active regions ON THE XY PLANE
for (a_alpha, e_alpha), t_norm_func in active_regions:
    center_x = (a_alpha + e_alpha) / 2
    center_y = (a_alpha + e_alpha) / 2
    
    tnorm_name = "Unknown"
    if t_norm_func == t_prod:
        tnorm_name = "Rescaled $T_P$"
    elif t_norm_func == t_lukas:
        tnorm_name = "Rescaled $T_L$"
    
    ax.text(center_x, center_y, -0.15, tnorm_name, color='black', # z value for XY plane
            horizontalalignment='center', verticalalignment='center', zorder=15, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.9))

ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_zlabel('T(x,y)', fontsize=14)
ax.set_title('Ordinal Sum of T-norms Visualization', pad=25, fontsize=18)
ax.set_zlim(0, 1.1) # Adjusted zlim to make space for text on XY plane
ax.view_init(elev=30, azim=-135) # Adjusted view angle

# Legend
handles, labels = ax.get_legend_handles_labels()
unique_labels_dict = {}
new_handles = []
new_labels = []
for handle, label in zip(handles, labels):
    if label not in unique_labels_dict:
        unique_labels_dict[label] = handle
        new_handles.append(handle)
        new_labels.append(label)
ax.legend(new_handles, new_labels, loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=10)

plt.tight_layout(rect=[0, 0, 0.95, 0.95]) # Adjust layout for title and legend
plt.show()