import numpy as np
import matplotlib.pyplot as plt

def plot_fuzzy_numbers():
    """
    Plots three fuzzy numbers in one figure:
      1. Triangular fuzzy number: A = (a, α, β)
      2. Trapezoidal fuzzy number: A = (a, b, α, β)
      3. LR fuzzy number: A = (a, b, α, β) with nonlinear L and R functions.
    
    The x–axis shows only the key labels ("a" or "a" and "b"), and horizontal double–headed
    arrows below the x–axis indicate that the left segment has length α and the right segment has
    length β. A horizontal dashed line at y=1 is drawn in every subplot.
    
    Adjustments:
      - The arrow text labels are now placed closer to the arrows.
      - The x-axis label "x" is positioned in the bottom right corner.
    """
    
    # Create figure and subplots with extra white space
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15, wspace=0.3)
    
    # Define the arrow offsets for all subplots (in axis coordinate units)
    arrow_y = -0.05  # vertical offset for arrows
    text_y  = -0.08  # vertical offset for the arrow labels (closer now)
    
    # --- 1. Triangular Fuzzy Number ---
    a_tri = 5
    alpha_tri = 3   # left width
    beta_tri  = 2   # right width
    
    x_left  = np.linspace(a_tri - alpha_tri, a_tri, 100)
    mu_left = (x_left - (a_tri - alpha_tri)) / alpha_tri
    x_right = np.linspace(a_tri, a_tri + beta_tri, 100)
    mu_right = ((a_tri + beta_tri) - x_right) / beta_tri
    x_tri = np.concatenate([x_left, x_right])
    mu_tri = np.concatenate([mu_left, mu_right])
    
    ax = axs[0]
    ax.plot(x_tri, mu_tri, 'b-', linewidth=2)
    ax.fill_between(x_tri, mu_tri, color='gray', alpha=0.5)
    ax.axvline(x=a_tri, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=1, color='k', linestyle='--', linewidth=1)
    
    # Set x-axis ticks to show only the label "a"
    ax.set_xticks([a_tri])
    ax.set_xticklabels(['a'])
    
    # Use the x-axis transform to place arrow annotations
    trans = ax.get_xaxis_transform()
    # Left arrow: from (a - α) to a, labeled with "α"
    ax.annotate("",
                xy=(a_tri, arrow_y), xytext=(a_tri - alpha_tri, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="black"),
                xycoords=('data', 'axes fraction'))
    ax.text(a_tri - alpha_tri/2, text_y, r'$\alpha$', transform=trans,
            ha='center', va='top', clip_on=False)
    
    # Right arrow: from a to (a + β), labeled with "β"
    ax.annotate("",
                xy=(a_tri + beta_tri, arrow_y), xytext=(a_tri, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="black"),
                xycoords=('data', 'axes fraction'))
    ax.text(a_tri + beta_tri/2, text_y, r'$\beta$', transform=trans,
            ha='center', va='top', clip_on=False)
    
    ax.set_xlim(a_tri - alpha_tri - 1, a_tri + beta_tri + 1)
    ax.set_ylim(0, 1.1)
    ax.set_anchor('SW')
    
    ax.set_title(r'Triangular Fuzzy Number: $A=(a,\ \alpha,\ \beta)$')
    ax.set_xlabel('x')
    ax.xaxis.set_label_coords(0.975, -0.02)  # place x-axis label in bottom right
    ax.set_ylabel(r'$\mu_A$')
    ax.yaxis.set_label_coords(-0.02, 0.975)
    
    # --- 2. Trapezoidal Fuzzy Number ---
    a_trap = 4
    b_trap = 6
    alpha_trap = 2   # left width
    beta_trap  = 3   # right width
    
    x_left  = np.linspace(a_trap - alpha_trap, a_trap, 100)
    mu_left = (x_left - (a_trap - alpha_trap)) / alpha_trap
    x_plateau = np.linspace(a_trap, b_trap, 100)
    mu_plateau = np.ones_like(x_plateau)
    x_right = np.linspace(b_trap, b_trap + beta_trap, 100)
    mu_right = ((b_trap + beta_trap) - x_right) / beta_trap
    x_trap = np.concatenate([x_left, x_plateau, x_right])
    mu_trap = np.concatenate([mu_left, mu_plateau, mu_right])
    
    ax = axs[1]
    ax.plot(x_trap, mu_trap, 'g-', linewidth=2)
    ax.fill_between(x_trap, mu_trap, color='gray', alpha=0.5)
    ax.axvline(x=a_trap, color='k', linestyle='--', linewidth=1)
    ax.axvline(x=b_trap, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=1, color='k', linestyle='--', linewidth=1)
    
    ax.set_xticks([a_trap, b_trap])
    ax.set_xticklabels(['a', 'b'])
    
    trans = ax.get_xaxis_transform()
    ax.annotate("",
                xy=(a_trap, arrow_y), xytext=(a_trap - alpha_trap, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="black"),
                xycoords=('data', 'axes fraction'))
    ax.text(a_trap - alpha_trap/2, text_y, r'$\alpha$', transform=trans,
            ha='center', va='top', clip_on=False)
    ax.annotate("",
                xy=(b_trap + beta_trap, arrow_y), xytext=(b_trap, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="black"),
                xycoords=('data', 'axes fraction'))
    ax.text(b_trap + beta_trap/2, text_y, r'$\beta$', transform=trans,
            ha='center', va='top', clip_on=False)
    
    ax.set_xlim(a_trap - alpha_trap - 1, b_trap + beta_trap + 1)
    ax.set_ylim(0, 1.1)
    ax.set_anchor('SW')
    
    ax.set_title(r'Trapezoidal Fuzzy Number: $A=(a,\ b,\ \alpha,\ \beta)$')
    ax.set_xlabel('x')
    ax.xaxis.set_label_coords(0.975, -0.02)
    ax.set_ylabel(r'$\mu_A$')
    ax.yaxis.set_label_coords(-0.02, 0.975)
    
    # --- 3. LR Fuzzy Number ---
    a_lr = 3
    b_lr = 7
    alpha_lr = 2
    beta_lr  = 4
    
    L_func = lambda t: np.maximum(1 - t**2, 0)
    R_func = lambda t: np.sqrt(np.maximum(1 - t, 0))
    
    x_left  = np.linspace(a_lr - alpha_lr, a_lr, 100)
    t_left = (a_lr - x_left) / alpha_lr
    mu_left = L_func(t_left)
    x_core = np.linspace(a_lr, b_lr, 100)
    mu_core = np.ones_like(x_core)
    x_right = np.linspace(b_lr, b_lr + beta_lr, 100)
    t_right = (x_right - b_lr) / beta_lr
    mu_right = R_func(t_right)
    x_lr = np.concatenate([x_left, x_core, x_right])
    mu_lr = np.concatenate([mu_left, mu_core, mu_right])
    
    ax = axs[2]
    ax.plot(x_lr, mu_lr, 'r-', linewidth=2, label=r'$L(t)=1-t^2,\quad R(t)=\sqrt{1-t}$')
    ax.fill_between(x_lr, mu_lr, color='gray', alpha=0.5)
    ax.axvline(x=a_lr, color='k', linestyle='--', linewidth=1)
    ax.axvline(x=b_lr, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=1, color='k', linestyle='--', linewidth=1)
    
    ax.set_xticks([a_lr, b_lr])
    ax.set_xticklabels(['a', 'b'])
    
    trans = ax.get_xaxis_transform()
    ax.annotate("",
                xy=(a_lr, arrow_y), xytext=(a_lr - alpha_lr, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="black"),
                xycoords=('data', 'axes fraction'))
    ax.text(a_lr - alpha_lr/2, text_y, r'$\alpha$', transform=trans,
            ha='center', va='top', clip_on=False)
    ax.annotate("",
                xy=(b_lr + beta_lr, arrow_y), xytext=(b_lr, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="black"),
                xycoords=('data', 'axes fraction'))
    ax.text(b_lr + beta_lr/2, text_y, r'$\beta$', transform=trans,
            ha='center', va='top', clip_on=False)
    
    ax.set_xlim(a_lr - alpha_lr - 1, b_lr + beta_lr + 1)
    ax.set_ylim(0, 1.1)
    ax.set_anchor('SW')
    
    ax.set_title(r'LR Fuzzy Number: $A=(a,\ b,\ \alpha,\ \beta)$')
    ax.set_xlabel('x')
    ax.xaxis.set_label_coords(0.975, -0.02)
    ax.set_ylabel(r'$\mu_A$')
    ax.yaxis.set_label_coords(-0.02, 0.975)
    ax.legend(loc='upper right', frameon=True)
    
    plt.show()

if __name__ == '__main__':
    plot_fuzzy_numbers()
