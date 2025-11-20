"""
Generate all figures for Week 3A slides: Dynamics on Networks — Spreading and Epidemics
Outputs PNG images in ./figures
Requires: matplotlib, networkx, numpy
"""

import os, random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import odeint

# -------------------------------------------------------------
# Setup
# -------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
os.makedirs("figures", exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}", dpi=300, bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 1. Spreading example network - 3 panels showing progression
# -------------------------------------------------------------
def create_spreading_examples():
    """Create 3-panel visualization showing progressive infection spread with proper SIR dynamics"""
    G = nx.watts_strogatz_graph(15, 4, 0.3, seed=1)
    pos = nx.spring_layout(G, seed=2)
    
    # SIR parameters - adjusted for better visualization
    infection_prob = 0.6  # Probability of infection per infected neighbor
    recovery_prob = 0.3   # Probability of recovery per time step
    
    # Initialize states - using sets for efficient operations
    infected = {0}  # Start with node 0 infected
    recovered = set()
    susceptible = set(range(1, 15))  # All other nodes start susceptible
    
    # Store states for each time step
    states = []
    
    # Simulate 3 time steps with controlled randomness for good visualization
    random.seed(123)  # Different seed for better spread visualization
    for t in range(3):
        # Store current state
        states.append({
            'infected': infected.copy(),
            'recovered': recovered.copy(),
            'susceptible': susceptible.copy()
        })
        
        if t < 2:  # Don't simulate after the last visualization
            # New infections
            new_infected = set()
            for infected_node in infected:
                for neighbor in G.neighbors(infected_node):
                    if neighbor in susceptible:
                        if random.random() < infection_prob:
                            new_infected.add(neighbor)
            
            # Remove newly infected from susceptible
            susceptible -= new_infected
            
            # Recovery process - only recover some infected nodes to maintain spread
            newly_recovered = set()
            for infected_node in list(infected):
                if random.random() < recovery_prob:
                    newly_recovered.add(infected_node)
            
            # Update states
            recovered |= newly_recovered
            infected = (infected - newly_recovered) | new_infected
            
            # Ensure we always have some infection for visualization
            if not infected and new_infected:
                infected = new_infected.copy()
                newly_recovered = set()  # Don't recover if we need to show spread
    
    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, (ax, state) in enumerate(zip(axes, states)):
        # Color nodes based on state
        colors = []
        for n in G.nodes:
            if n in state['infected']:
                colors.append("red")  # Currently infected
            elif n in state['recovered']:
                colors.append("green")  # Recovered/immune
            else:
                colors.append("lightblue")  # Susceptible
        
        # Draw network
        nx.draw(G, pos, ax=ax, node_color=colors, with_labels=False, 
                node_size=300, edge_color='gray', alpha=0.7)
        
        # Add title for each panel with state counts
        s_count = len(state['susceptible'])
        i_count = len(state['infected'])
        r_count = len(state['recovered'])
        ax.set_title(f"Time Step {i}\nS: {s_count}, I: {i_count}, R: {r_count}", 
                    fontsize=12, fontweight='bold')
        
        # Add legend only to the first panel
        if i == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightblue', label='Susceptible'),
                Patch(facecolor='red', label='Infected'),
                Patch(facecolor='green', label='Recovered')
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    plt.tight_layout()
    plt.savefig("figures/spreading_examples.png", dpi=300, bbox_inches="tight")
    plt.close()

create_spreading_examples()


# -------------------------------------------------------------
# 2. Homogeneous mixing vs structured network
# -------------------------------------------------------------
def draw_complete(n=10, fname="wellmixed.png"):
    G = nx.complete_graph(n)
    pos = nx.circular_layout(G)
    plt.figure(figsize=(3,3))
    nx.draw(G, pos, node_color="lightblue", edge_color="gray", node_size=200)
    savefig(fname)
draw_complete(10, "wellmixed.png")

def draw_random(n=12, p=0.25, fname="networked.png"):
    G = nx.erdos_renyi_graph(n, p, seed=2)
    pos = nx.spring_layout(G, seed=1)
    plt.figure(figsize=(3,3))
    nx.draw(G, pos, node_color="lightblue", edge_color="gray", node_size=250)
    savefig(fname)
draw_random()

# -------------------------------------------------------------
# 3. Epidemic threshold curve
# -------------------------------------------------------------
beta_over_mu = np.linspace(0, 3, 100)
epidemic_size = 1 / (1 + np.exp(-4*(beta_over_mu - 1)))  # logistic-like curve
plt.figure()
plt.plot(beta_over_mu, epidemic_size, color="crimson", lw=2)
plt.axvline(1, color="gray", ls="--")
plt.xlabel(r"$\beta / \mu$")
plt.ylabel("Final epidemic size")
plt.title("Epidemic threshold")
savefig("threshold_plot.png")

# -------------------------------------------------------------
# 4. Example networks (ER, WS, BA)
# -------------------------------------------------------------
# ER Network with degree-scaled nodes
G_er = nx.erdos_renyi_graph(50, 0.1, seed=1)
pos_er = nx.spring_layout(G_er, seed=2)
degrees_er = dict(G_er.degree())
node_sizes_er = [degrees_er[node] * 50 + 50 for node in G_er.nodes()]  # Scale by degree
plt.figure(figsize=(4,4))
nx.draw(G_er, pos_er, node_color="lightgray", edge_color="gray", 
        node_size=node_sizes_er, with_labels=False)
savefig("er_network.png")

# WS Network (keeping original size for comparison)
G_ws = nx.watts_strogatz_graph(15, 4, 0.3, seed=2)
pos_ws = nx.spring_layout(G_ws, seed=2)
plt.figure(figsize=(3,3))
nx.draw(G_ws, pos_ws, node_color="lightgray", edge_color="gray", node_size=250)
savefig("ws_network.png")

# BA Network with degree-scaled nodes
G_ba = nx.barabasi_albert_graph(50, 2, seed=3)
pos_ba = nx.spring_layout(G_ba, seed=2)
degrees_ba = dict(G_ba.degree())
node_sizes_ba = [degrees_ba[node] * 30 + 50 for node in G_ba.nodes()]  # Scale by degree
plt.figure(figsize=(4,4))
nx.draw(G_ba, pos_ba, node_color="lightgray", edge_color="gray", 
        node_size=node_sizes_ba, with_labels=False)
savefig("ba_network.png")

# -------------------------------------------------------------
# 5. Power-law degree distribution (scale-free)
# -------------------------------------------------------------
G = nx.barabasi_albert_graph(500, 3)
degrees = [d for _, d in G.degree()]
plt.figure()
plt.hist(degrees, bins=np.logspace(np.log10(1), np.log10(max(degrees)), 20),
         color="skyblue", edgecolor="k")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Degree (log)")
plt.ylabel("Frequency (log)")
plt.title("Power-law degree distribution")
savefig("powerlaw_degree.png")

# -------------------------------------------------------------
# 6. Immunization comparison
# -------------------------------------------------------------
def immunization_demo(G, remove_type="random", frac=0.2):
    n_remove = int(frac * G.number_of_nodes())
    nodes = list(G.nodes())
    if remove_type == "targeted":
        deg = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        removed = [n for n, _ in deg[:n_remove]]
    else:
        removed = random.sample(nodes, n_remove)
    remaining = [n for n in G if n not in removed]
    pos = nx.spring_layout(G, seed=1)
    colors = ["lightgray" if n in removed else "skyblue" for n in G]
    nx.draw(G, pos, node_color=colors, edge_color="gray", node_size=250)
    return pos, removed

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
immunization_demo(nx.barabasi_albert_graph(20, 2, seed=1), "random")
plt.title("Random removal")

plt.subplot(1, 2, 2)
immunization_demo(nx.barabasi_albert_graph(20, 2, seed=2), "targeted")
plt.title("Targeted removal")

savefig("immunization_compare.png")

# -------------------------------------------------------------
# 7. SIR time series (proper differential equation solution)
# -------------------------------------------------------------
def sir_model(y, t, beta, gamma):
    """
    SIR differential equations:
    dS/dt = -beta * S * I
    dI/dt = beta * S * I - gamma * I
    dR/dt = gamma * I
    """
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Parameters
beta = 0.5    # transmission rate
gamma = 0.1   # recovery rate
N = 1000      # total population
I0 = 1        # initial infected
S0 = N - I0   # initial susceptible
R0 = 0        # initial recovered

# Initial conditions (as fractions)
y0 = [S0/N, I0/N, R0/N]

# Time points
t = np.linspace(0, 50, 200)

# Solve ODE
sol = odeint(sir_model, y0, t, args=(beta, gamma))
S, I, R = sol.T

plt.figure()
plt.plot(t, S, label="S(t)", lw=2, color='blue')
plt.plot(t, I, label="I(t)", lw=2, color='red')
plt.plot(t, R, label="R(t)", lw=2, color='green')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Fraction of population")
plt.title("SIR epidemic curve")
plt.grid(True, alpha=0.3)
savefig("sir_curve.png")

# -------------------------------------------------------------
# 8. Epidemic size vs beta
# -------------------------------------------------------------
beta = np.linspace(0, 1, 100)
size = 1/(1 + np.exp(-20*(beta - 0.4)))
plt.figure()
plt.plot(beta, size, color="darkred", lw=2)
plt.axvline(0.4, color="gray", ls="--")
plt.xlabel(r"Transmission rate $\beta$")
plt.ylabel("Epidemic size")
plt.title("Epidemic size vs transmission rate")
savefig("epidemic_size_curve.png")

# -------------------------------------------------------------
# 9. Icon images for slide elements
# -------------------------------------------------------------
def create_node_icon():
    """Create a simple node icon"""
    fig, ax = plt.subplots(figsize=(1, 1))
    fig.patch.set_alpha(0)  # Make figure background transparent
    circle = plt.Circle((0.5, 0.5), 0.3, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("figures/node_icon.png", dpi=300, bbox_inches="tight", 
                transparent=True)
    plt.close()

def create_edge_icon():
    """Create a simple edge icon"""
    fig, ax = plt.subplots(figsize=(1, 1))
    fig.patch.set_alpha(0)  # Make figure background transparent
    ax.plot([0.2, 0.8], [0.5, 0.5], 'k-', linewidth=3)
    # Add small circles at endpoints
    ax.plot(0.2, 0.5, 'o', color='lightblue', markersize=8, markeredgecolor='black')
    ax.plot(0.8, 0.5, 'o', color='lightblue', markersize=8, markeredgecolor='black')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("figures/edge_icon.png", dpi=300, bbox_inches="tight", 
                transparent=True)
    plt.close()

def create_beta_icon():
    """Create a beta (transmission) icon"""
    fig, ax = plt.subplots(figsize=(1, 1))
    fig.patch.set_alpha(0)  # Make figure background transparent
    ax.text(0.5, 0.5, r'$\beta$', fontsize=24, ha='center', va='center', 
            color='red', weight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("figures/beta_icon.png", dpi=300, bbox_inches="tight", 
                transparent=True)
    plt.close()

def create_mu_icon():
    """Create a mu (recovery) icon"""
    fig, ax = plt.subplots(figsize=(1, 1))
    fig.patch.set_alpha(0)  # Make figure background transparent
    ax.text(0.5, 0.5, r'$\mu$', fontsize=24, ha='center', va='center', 
            color='green', weight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("figures/mu_icon.png", dpi=300, bbox_inches="tight", 
                transparent=True)
    plt.close()

# Generate all icon images
create_node_icon()
create_edge_icon()
create_beta_icon()
create_mu_icon()

# -------------------------------------------------------------
# 10. Temporal contacts visualization
# -------------------------------------------------------------
def create_temporal_contacts():
    """Create a visualization showing temporal network contacts"""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    # Create a base network
    G = nx.erdos_renyi_graph(8, 0.3, seed=1)
    pos = nx.spring_layout(G, seed=1)
    
    # Three time snapshots with different active edges
    time_labels = ['t=1', 't=2', 't=3']
    edge_probs = [0.3, 0.6, 0.2]  # Different activity levels
    
    for i, (ax, prob, label) in enumerate(zip(axes, edge_probs, time_labels)):
        # Randomly activate edges based on probability
        np.random.seed(i)
        active_edges = [e for e in G.edges() if np.random.random() < prob]
        
        # Draw all nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', 
                              node_size=200, edgecolors='black')
        
        # Draw only active edges
        nx.draw_networkx_edges(G, pos, edgelist=active_edges, ax=ax, 
                              edge_color='red', width=2)
        
        # Draw inactive edges as light gray
        inactive_edges = [e for e in G.edges() if e not in active_edges]
        nx.draw_networkx_edges(G, pos, edgelist=inactive_edges, ax=ax, 
                              edge_color='lightgray', width=0.5, alpha=0.5)
        
        ax.set_title(label, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("figures/temporal_contacts.png", dpi=300, bbox_inches="tight")
    plt.close()

create_temporal_contacts()

# -------------------------------------------------------------
# 11. Contact networks examples
# -------------------------------------------------------------
def create_contact_networks():
    """Create examples of different contact network types"""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    # Workplace network (hierarchical)
    G1 = nx.balanced_tree(2, 2)
    pos1 = nx.spring_layout(G1, seed=1)
    nx.draw(G1, pos1, ax=axes[0], node_color='lightcoral', 
            node_size=150, with_labels=False)
    axes[0].set_title('Workplace', fontsize=10)
    axes[0].axis('off')
    
    # School network (clustered)
    G2 = nx.caveman_graph(3, 3)
    pos2 = nx.spring_layout(G2, seed=2)
    nx.draw(G2, pos2, ax=axes[1], node_color='lightgreen', 
            node_size=150, with_labels=False)
    axes[1].set_title('School Classes', fontsize=10)
    axes[1].axis('off')
    
    # Social network (scale-free)
    G3 = nx.barabasi_albert_graph(12, 2, seed=3)
    pos3 = nx.spring_layout(G3, seed=3)
    degrees3 = dict(G3.degree())
    node_sizes3 = [degrees3[node] * 30 + 50 for node in G3.nodes()]
    nx.draw(G3, pos3, ax=axes[2], node_color='lightblue', 
            node_size=node_sizes3, with_labels=False)
    axes[2].set_title('Social Media', fontsize=10)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("figures/contact_networks.png", dpi=300, bbox_inches="tight")
    plt.close()

create_contact_networks()

# -------------------------------------------------------------
# 12. Different types of contagion
# -------------------------------------------------------------
def create_contagion_types():
    """Create visualization showing different contagion types"""
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    # Base network for all examples
    G = nx.watts_strogatz_graph(12, 3, 0.3, seed=1)
    pos = nx.spring_layout(G, seed=2)
    
    # Disease spreading
    infected_nodes = [0, 1, 5]
    node_colors1 = ['red' if n in infected_nodes else 'lightgray' for n in G.nodes()]
    nx.draw(G, pos, ax=axes[0,0], node_color=node_colors1, node_size=200, 
            with_labels=False, edge_color='gray')
    axes[0,0].set_title('Disease Spreading', fontsize=10)
    axes[0,0].axis('off')
    
    # Information diffusion
    informed_nodes = [0, 2, 3, 7]
    node_colors2 = ['blue' if n in informed_nodes else 'lightgray' for n in G.nodes()]
    nx.draw(G, pos, ax=axes[0,1], node_color=node_colors2, node_size=200, 
            with_labels=False, edge_color='gray')
    axes[0,1].set_title('Information Diffusion', fontsize=10)
    axes[0,1].axis('off')
    
    # Innovation adoption
    adopter_nodes = [0, 1, 2]
    node_colors3 = ['green' if n in adopter_nodes else 'lightgray' for n in G.nodes()]
    nx.draw(G, pos, ax=axes[1,0], node_color=node_colors3, node_size=200, 
            with_labels=False, edge_color='gray')
    axes[1,0].set_title('Innovation Adoption', fontsize=10)
    axes[1,0].axis('off')
    
    # Emotional contagion
    emotional_nodes = [0, 4, 6, 8]
    node_colors4 = ['orange' if n in emotional_nodes else 'lightgray' for n in G.nodes()]
    nx.draw(G, pos, ax=axes[1,1], node_color=node_colors4, node_size=200, 
            with_labels=False, edge_color='gray')
    axes[1,1].set_title('Emotional Contagion', fontsize=10)
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig("figures/contagion_types.png", dpi=300, bbox_inches="tight")
    plt.close()

create_contagion_types()

print("✅ All figures generated in ./figures")