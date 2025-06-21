import numpy as np
import matplotlib.pyplot as plt

def ramp_function(x, start, end):
    """Computes a ramp-up membership function (S-shape)."""
    if start >= end:
        raise ValueError("start must be less than end")
    return np.clip((x - start) / (end - start), 0, 1)

def z_function(x, start, end):
    """Computes a ramp-down membership function (Z-shape)."""
    if start >= end:
        raise ValueError("start must be less than end")
    return np.clip((end - x) / (end - start), 0, 1)

# 1. Define the universe of discourse
price = np.linspace(0, 150, 500)

# 2. Define membership functions with overlapping parameters
# 'cheap': fully true < $30, fully false > $70
mu_cheap = z_function(price, 30, 70)
# 'expensive': fully false < $50, fully true > $90
mu_expensive = ramp_function(price, 50, 90)

# 3. Apply linguistic modifiers
mu_somewhat_cheap = np.sqrt(mu_cheap)
mu_very_expensive = mu_expensive ** 2

# 4. Create the plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the 'cheap' related functions
ax.plot(price, mu_cheap, label='cheap', color='cornflowerblue', linewidth=2.5)
ax.plot(price, mu_somewhat_cheap, label='somewhat cheap', color='lightskyblue', linestyle='--', linewidth=2)

# Plot the 'expensive' related functions
ax.plot(price, mu_expensive, label='expensive', color='salmon', linewidth=2.5)
ax.plot(price, mu_very_expensive, label='very expensive', color='firebrick', linestyle='--', linewidth=2)

# Add a fill to highlight the overlap region
ax.fill_between(price, 0, np.minimum(mu_cheap, mu_expensive), 
                color='grey', alpha=0.2, label='Overlap Region')

# 5. Style the plot for clarity
ax.set_title("Linguistic Variable 'Price' with Overlapping Terms", fontsize=16, pad=20)
ax.set_xlabel("Price (€)", fontsize=12)
ax.set_ylabel("Membership Degree (μ)", fontsize=12)
ax.set_xlim(0, 150)
ax.set_ylim(0, 1.05)
ax.grid(True, which='both', linestyle=':', linewidth=0.6)

# Improve legend
legend = ax.legend(fontsize=11, title="Linguistic Values", frameon=True, facecolor='white', framealpha=0.9)
legend.get_title().set_fontweight('bold')

# Clean up spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 6. Save the figure with the new name
plt.savefig('linguistic_variable_price_overlap.pdf', bbox_inches='tight', dpi=300)

print("Plot 'linguistic_variable_price_overlap.pdf' generated successfully.")
plt.show()