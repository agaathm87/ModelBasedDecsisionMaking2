# ==============================================================================
# ASSIGNMENT 2: DYNAMICS ON NETWORKS - THRESHOLDS AND SPREADING
# Unified Simulation Framework
#
# Course: Model Based Decision-making (5404MBDM6Y)
# Student: 10205071
# Date: November 20, 2025
#
# DESCRIPTION:
# This script integrates graph ingestion (YouTube), Hub-and-Spoke sampling,
# Linear Threshold Model (LTM) simulation with 5 threshold distributions,
# and high-fidelity visualization into a single pipeline.
#
# Dependencies: networkx, matplotlib, pandas, numpy, scipy, gzip
# ==========================================
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
import random
import os
import sys
import time
import gzip
import seaborn as sns
import matplotlib.colors as mcolors
from collections import defaultdict

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
# I/O Settings
INPUT_FILE = "com-youtube.ungraph.txt.gz"
OUTPUT_DIR = "simulation_outputs"
RESULTS_CSV = "monte_carlo_results.csv"
BC_FILE = "sampled_bc_exact.csv"

# Simulation Parameters
# N_TARGET_SUBGRAPH can be:
#  - int >=1            -> absolute node count, e.g. 200
#  - float in (0,1]     -> fraction of full graph, e.g. 0.2 for 20%
#  - str like "20%"     -> percent string parsed as fraction
# Default reduced sample for faster centrality runs
N_TARGET_SUBGRAPH = 0.05    # default: 5% of the full graph

SAMPLE_SEED_FRACTIONS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]
NUM_SIMULATIONS = 10        # Pilot runs; adaptive controller may request more
MAX_STEPS = 50              # Guard clause for convergence

# CONFIG additions: choose one fixed value, and a sweep of constant values to vary
DEFAULT_FIXED_THRESHOLD = 0.2
CONSTANT_SWEEP = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]   # 'Constant' scenarios will iterate these values

# Statistical Constants for Rigor Check
CONFIDENCE_LEVEL = 0.99
Z_SCORE = st.norm.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2)
PRECISION_EPSILON = 0.01

# Reproducibility
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# Module 1: Graph Ingestion & Sampling
# ==========================================
def load_network_data(filepath="datasets\\com-youtube.ungraph.txt.gz",
                      data_dir=".",
                      youtube_url="https://snap.stanford.edu/data/com-Youtube.ungraph.txt.gz",
                      force_download=False,
                      max_rows=None,
                      dtype_nodes=np.int32):
    """
    Robust loader: try local (data_dir, datasets/), optional download, parse with pandas,
    clean (drop NA/self-loops/dupes), build NetworkX graph and return LCC.
    Falls back to a Barabasi-Albert surrogate graph on any failure.
    """
    from io import BytesIO
    try:
        import requests
    except Exception:
        requests = None

    start = time.time()
    candidates = [
        os.path.join(data_dir, filepath),
        os.path.join(data_dir, os.path.basename(filepath)),
        os.path.join("datasets", filepath),
        os.path.join("datasets", os.path.basename(filepath)),
        filepath,
        os.path.basename(filepath),
    ]
    # if user passed .gz, also consider uncompressed
    if filepath.endswith(".gz"):
        ungz = filepath[:-3]
        candidates.extend([os.path.join(data_dir, ungz), os.path.join("datasets", ungz), ungz, os.path.basename(ungz)])
    # keep order, remove duplicates
    seen = set(); candidates = [p for p in candidates if not (p in seen or seen.add(p))]

    found = None
    for p in candidates:
        if os.path.exists(p) and not force_download:
            found = p; break

    df = None
    read_kwargs = dict(sep="\t", comment="#", names=["start_node", "end_node"],
                       dtype={"start_node": dtype_nodes, "end_node": dtype_nodes},
                       header=None, engine="c")

    if found:
        print(f"[INFO] Loading local file: {found}")
        try:
            if str(found).endswith(".gz"):
                df = pd.read_csv(found, compression="gzip", **read_kwargs, nrows=max_rows)
            else:
                df = pd.read_csv(found, **read_kwargs, nrows=max_rows)
        except Exception as e:
            print(f"[WARN] Failed to parse local file ({e}). Will fallback to surrogate or download.")
            df = None

    if df is None and requests is not None and youtube_url:
        try:
            print(f"[INFO] Attempting download from {youtube_url} ...")
            r = requests.get(youtube_url, timeout=90)
            r.raise_for_status()
            buf = BytesIO(r.content)
            df = pd.read_csv(buf, compression="gzip", **read_kwargs, nrows=max_rows)
            print("[INFO] Downloaded and parsed dataset.")
        except Exception as e:
            print(f"[WARN] Download/parse failed: {e}")
            df = None

    if df is None or df.shape[0] == 0:
        print("[WARN] Edge-list not available or empty. Generating Barabasi-Albert surrogate graph.")
        G = nx.barabasi_albert_graph(n=5000, m=3, seed=GLOBAL_SEED)
        elapsed = time.time() - start
        print(f"[INFO] Surrogate BA graph created in {elapsed:.2f}s: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    # defensive cleaning
    try:
        df = df.dropna()
        df = df[df.start_node != df.end_node]
        df = df.drop_duplicates()
        df.start_node = df.start_node.astype(dtype_nodes, copy=False)
        df.end_node = df.end_node.astype(dtype_nodes, copy=False)
    except Exception as e:
        print(f"[WARN] Cleaning failed ({e}). Proceeding with best-effort df.")

    if df.shape[0] == 0:
        print("[WARN] No valid edges after cleaning. Generating Barabasi-Albert surrogate.")
        G = nx.barabasi_albert_graph(n=5000, m=3, seed=GLOBAL_SEED)
        return G

    try:
        G = nx.from_pandas_edgelist(df, "start_node", "end_node", create_using=nx.Graph())
    except Exception as e:
        print(f"[WARN] Failed to build graph from dataframe ({e}). Generating surrogate BA graph.")
        return nx.barabasi_albert_graph(n=5000, m=3, seed=GLOBAL_SEED)

    # extract LCC
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    elapsed = time.time() - start
    print(f"[INFO] LCC Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (load time {elapsed:.2f}s)")
    return G

def hub_and_spoke_sampling(G_full, N_target=None):
    """
    Hub-and-Spoke sampling. N_target (optional) may be:
      - None -> uses global N_TARGET_SUBGRAPH
      - int >=1 : absolute node count
      - float in (0,1] : fraction of G_full nodes
      - str like '20%' : percent string parsed as fraction
    Returns sampled subgraph with node attribute 'role' in {'Hub','Spoke'}.
    """
    if N_target is None:
        N_target = globals().get('N_TARGET_SUBGRAPH', 0.2)

    # parse percent string
    if isinstance(N_target, str) and N_target.strip().endswith('%'):
        try:
            pct = float(N_target.strip().rstrip('%')) / 100.0
            N_target_val = max(10, int(len(G_full) * pct))
        except Exception:
            N_target_val = max(10, int(len(G_full) * 0.2))
    elif isinstance(N_target, float) and 0.0 < N_target <= 1.0:
        N_target_val = max(10, int(len(G_full) * N_target))
    else:
        try:
            N_target_val = int(N_target)
        except Exception:
            N_target_val = max(10, int(len(G_full) * 0.2))

    # clamp to graph size and minimum
    N_target_val = min(len(G_full), max(10, N_target_val))

    if G_full.number_of_nodes() <= N_target_val:
        return G_full

    print(f"[INFO] Executing Hub-and-Spoke Sampling (Target N={N_target_val} / {len(G_full)} nodes requested)...")

    degree_map = dict(G_full.degree())
    sorted_hubs = sorted(degree_map.keys(), key=degree_map.get, reverse=True)

    # Heuristic: select top 5% as initial hubs (at least 10)
    k_H = max(10, int(len(G_full) * 0.05))
    hubs = set(sorted_hubs[:k_H])

    # collect hubs + neighbors (spokes)
    nodes_to_keep = set(hubs)
    for node in hubs:
        nodes_to_keep.update(G_full.neighbors(node))
        if len(nodes_to_keep) >= N_target_val * 1.5:
            break

    # trim if too many
    if len(nodes_to_keep) > N_target_val:
        subgraph_degrees = {n: degree_map[n] for n in nodes_to_keep}
        sorted_selection = sorted(subgraph_degrees, key=subgraph_degrees.get, reverse=True)
        nodes_to_keep = set(sorted_selection[:N_target_val])

    G_sampled = G_full.subgraph(nodes_to_keep).copy()

    # tag role attribute
    nx.set_node_attributes(G_sampled, {n: 'Hub' if n in hubs else 'Spoke' for n in G_sampled.nodes()}, 'role')

    print(f"[INFO] Sample Generated: {G_sampled.number_of_nodes()} nodes, {G_sampled.number_of_edges()} edges.")
    return G_sampled

# ==========================================
# Module 2: Centrality & Seeding
# ==========================================
def precompute_centrality(G, approx_k=None):
    """
    Calculates Betweenness Centrality and saves to CSV.
    - If approx_k (int) is provided and < len(G), uses NetworkX approximation (k sources).
    - If approx_k is None, chooses a sensible default (min(200, max(10, 5% of nodes))).
    Returns node->betweenness dict and writes CSV to OUTPUT_DIR/BC_FILE.
    """
    csv_path = os.path.join(OUTPUT_DIR, BC_FILE)
    print(f"[INFO] Calculating Betweenness Centrality (approx_k={approx_k}) ...")
    start_t = time.time()

    # choose default k relative to graph size when not provided
    if approx_k is None:
        approx_k = min(200, max(10, int(len(G) * 0.05)))

    try:
        if approx_k and approx_k < len(G):
            # approximate Brandes (samples 'k' source nodes)
            bc_map = nx.betweenness_centrality(G, k=approx_k, normalized=True, seed=GLOBAL_SEED)
        else:
            # full exact computation
            bc_map = nx.betweenness_centrality(G, normalized=True, seed=GLOBAL_SEED)
    except Exception as e:
        print(f"[WARN] Betweenness computation failed ({e}). Falling back to exact on top-degree subset.")
        # Fallback: compute exact on top-degree induced subgraph up to 1000 nodes, map back
        deg = dict(G.degree())
        top_nodes = sorted(deg, key=deg.get, reverse=True)[:1000]
        G_sub = G.subgraph(top_nodes).copy()
        try:
            bc_sub = nx.betweenness_centrality(G_sub, normalized=True, seed=GLOBAL_SEED)
            bc_map = {n: bc_sub.get(n, 0.0) for n in G.nodes()}
        except Exception as e2:
            print(f"[ERROR] Fallback betweenness also failed ({e2}). Assigning zero scores.")
            bc_map = {n: 0.0 for n in G.nodes()}

    # Save to CSV
    df = pd.DataFrame(bc_map.items(), columns=['Node', 'Betweenness'])
    df = df.sort_values(by='Betweenness', ascending=False)
    df.to_csv(csv_path, index=False)

    print(f"[INFO] Centrality calculated in {time.time()-start_t:.2f}s. Saved to {csv_path}")
    return bc_map

def get_seeds(G, strategy, fraction, bc_map=None):
    """
    Selects seed nodes according to strategy.
    Parameters:
      - G: NetworkX Graph (sample)
      - strategy: 'Random' | 'Degree' | 'Betweenness'
      - fraction: fraction of nodes to seed (0-1)
      - bc_map: optional precomputed betweenness map (for 'Betweenness' strategy)
    Returns:
      list of node IDs selected as seeds
    """
    N = G.number_of_nodes()
    num_seeds = max(1, int(N * fraction))

    nodes = list(G.nodes())

    if strategy == 'Random':
        # uniformly random seed selection
        return random.sample(nodes, num_seeds)

    elif strategy == 'Degree':
        # choose nodes with highest degree in the sampled subgraph
        degree_map = dict(G.degree())
        sorted_nodes = sorted(degree_map.keys(), key=degree_map.get, reverse=True)
        return sorted_nodes[:num_seeds]

    elif strategy == 'Betweenness':
        # require bc_map; if not provided compute fallback (costly)
        if bc_map is None:
            bc_map = nx.betweenness_centrality(G, normalized=True)
        sorted_nodes = sorted(bc_map.keys(), key=lambda x: bc_map.get(x, 0), reverse=True)
        return sorted_nodes[:num_seeds]

    # fallback empty
    return []

# ==========================================
# Module 3: Threshold Dynamics (LTM)
# ==========================================
def get_thresholds(G, scenario_key):
    """
    Generates threshold values based on distributions.
    Supported keys:
      - 'Fixed' or 'Fixed_0.2'    : all thresholds = chosen fixed value (DEFAULT_FIXED_THRESHOLD if unspecified)
      - 'Constant_0.3'            : all thresholds = 0.3  (used when sweeping constants)
      - 'Uniform'                 : Uniform(0,1)
      - 'Beta_Low' / 'Beta_Symmetric' / 'Beta_High'
      - 'Normal'                  : Normal(phi_mean, phi_std) clipped to [0,1]
    Unknown keys -> fallback to DEFAULT_FIXED_THRESHOLD.
    Returns: dict node -> threshold float
    """
    phi_mean = globals().get('PHI_MEAN', 0.3)
    phi_std = globals().get('PHI_STD', 0.1)

    # Helper to parse suffix value if present (e.g., 'Fixed_0.2' or 'Constant_0.3')
    def _parse_value_from_key(key, default):
        parts = str(key).split('_')
        if len(parts) > 1:
            try:
                return float(parts[1])
            except Exception:
                return default
        return default

    if scenario_key.startswith('Fixed'):
        val = _parse_value_from_key(scenario_key, globals().get('DEFAULT_FIXED_THRESHOLD', 0.2))
        return {n: float(val) for n in G.nodes()}

    if scenario_key.startswith('Constant'):
        val = _parse_value_from_key(scenario_key, globals().get('DEFAULT_FIXED_THRESHOLD', 0.2))
        return {n: float(val) for n in G.nodes()}

    if scenario_key == 'Uniform':
        return {n: float(np.random.uniform(0, 1)) for n in G.nodes()}

    if scenario_key == 'Beta_Low':
        return {n: float(np.random.beta(2, 5)) for n in G.nodes()}

    if scenario_key == 'Beta_Symmetric':
        return {n: float(np.random.beta(2, 2)) for n in G.nodes()}

    if scenario_key == 'Beta_High':
        return {n: float(np.random.beta(5, 2)) for n in G.nodes()}

    if scenario_key in ('Normal', 'normal'):
        vals = np.clip(np.random.normal(phi_mean, phi_std, len(G)), 0.0, 1.0)
        return {n: float(vals[i]) for i, n in enumerate(G.nodes())}

    # fallback to the default fixed
    return {n: float(globals().get('DEFAULT_FIXED_THRESHOLD', 0.2)) for n in G.nodes()}

def run_threshold_simulation(G, seeds, thresholds):
    """
    Executes a synchronous Linear Threshold Model (Granovetter LTM) run.
    Returns history list (adoption fraction per step) and final active count.
    """
    active_set = set(seeds)
    history = [len(active_set) / len(G)]

    # Pre-compute neighbors to speed up inner loop
    adj = {n: set(G.neighbors(n)) for n in G.nodes()}
    degrees = {n: len(adj[n]) for n in G.nodes()}

    for step in range(MAX_STEPS):
        newly_active = set()

        # iterate inactive nodes
        inactive_nodes = [n for n in G.nodes() if n not in active_set]

        if not inactive_nodes:
            break

        for node in inactive_nodes:
            deg = degrees[node]
            if deg == 0: continue

            # Count active neighbors
            active_neighbors = len([nbr for nbr in adj[node] if nbr in active_set])
            influence = active_neighbors / deg

            if influence >= thresholds[node]:
                newly_active.add(node)

        if newly_active:
            active_set.update(newly_active)
            history.append(len(active_set) / len(G))
        else:
            history.extend([history[-1]] * (MAX_STEPS - step - 1))
            break

    return history, len(active_set)

# ==========================================
# Module 4: Execution & Statistics
# ==========================================
def calculate_statistical_rigor(pilot_variances):
    """
    Calculates R_req for statistical rigor based on pilot data variance.
    Formula: R_req >= (Z * s0 / epsilon)^2
    Returns the maximum recommended runs across scenarios.
    """
    print("\n" + "="*40)
    print("STATISTICAL RIGOR CHECK")
    print("="*40)

    max_r_req = 0
    per_scenario = {}

    for scenario, s0 in pilot_variances.items():
        if s0 == 0:
            per_scenario[scenario] = 0.0
            continue
        r_req = (Z_SCORE * s0 / PRECISION_EPSILON) ** 2
        per_scenario[scenario] = r_req
        print(f"Scenario [{scenario}]: StdDev={s0:.4f} -> R_req >= {r_req:.1f}")
        max_r_req = max(max_r_req, r_req)

    print(f"\n[DECISION] Recommended Runs for {CONFIDENCE_LEVEL*100}% Confidence: {int(np.ceil(max_r_req))}")
    print(f"[ACTION] Using configured NUM_SIMULATIONS = {NUM_SIMULATIONS}")
    return max_r_req, per_scenario

def adaptive_execute_experiment_suite(G, bc_map, pilot_runs=10, max_total_runs=500, max_wall_time_seconds=600, seed_fractions=None):
    """
    Adaptive experiment controller (pilot -> variance -> additional runs with caps).
    Returns combined timeseries DataFrame and saves teacher-style summaries.
    """
    strategies = ['Random', 'Degree', 'Betweenness']

    # dynamic list: fixed single value + standard distributions + sweep of constants
    base_scenarios = ['Uniform', 'Beta_Low', 'Beta_Symmetric', 'Beta_High', 'Normal']
    fixed_val = globals().get('DEFAULT_FIXED_THRESHOLD', 0.2)
    sweep = globals().get('CONSTANT_SWEEP', [0.1, 0.2, 0.3])
    threshold_scenarios = [f'Fixed_{fixed_val}'] + base_scenarios + [f'Constant_{v}' for v in sweep]
    if seed_fractions is None:
        seed_fractions = globals().get('SAMPLE_SEED_FRACTIONS', [globals().get('SEED_FRACTION', 0.01)])

    B = max(1, int(G.number_of_nodes() * min(seed_fractions)))  # fallback for aggregation

    results_data = []
    pilot_tracker = defaultdict(list)

    total_pilot_units = pilot_runs * len(threshold_scenarios) * len(strategies) * len(seed_fractions)
    print(f"\n[INFO] Starting PILOT: {pilot_runs} runs per setting -> {total_pilot_units} simulation units...")
    pilot_start = time.time()
    unit_counter = 0

    # --- PILOT PHASE ---
    for thresh_key in threshold_scenarios:
        for seed_frac in seed_fractions:
            for trial in range(pilot_runs):
                trial_seed = GLOBAL_SEED + (abs(hash(thresh_key)) % 1000) + int(seed_frac * 1000) + trial * 1009
                random.seed(trial_seed); np.random.seed(trial_seed)
                thresholds = get_thresholds(G, thresh_key)

                for strat in strategies:
                    seeds = get_seeds(G, strat, seed_frac, bc_map)
                    curve, _ = run_threshold_simulation(G, seeds, thresholds)
                    final_frac = float(curve[-1])
                    pilot_tracker[f"{strat}_{thresh_key}_{seed_frac}"].append(final_frac)

                    arr = np.array(curve)
                    changes = np.where(np.diff(arr) != 0)[0]
                    time_to_converge = int(changes[-1] + 1) if len(changes) else 0

                    for t, val in enumerate(curve):
                        results_data.append({
                            'Trial': trial,
                            'Strategy': strat,
                            'Threshold_Dist': thresh_key,
                            'Seed_Fraction': seed_frac,
                            'Step': int(t),
                            'Adoption_Fraction': float(val),
                            'Final_Fraction': final_frac,
                            'TimeToConverge': time_to_converge,
                            'SeedsCount': len(seeds)
                        })

                    unit_counter += 1
                    if unit_counter % 50 == 0:
                        print(f" -> Completed {unit_counter}/{total_pilot_units} pilot simulation units...")

    pilot_elapsed = time.time() - pilot_start
    time_per_unit = pilot_elapsed / max(1, total_pilot_units)
    cores = os.cpu_count() or 1

    # Compute pilot variances and recommended runs
    variances = {k: np.std(v, ddof=0) for k, v in pilot_tracker.items()}
    max_r_req, per_scenario = calculate_statistical_rigor(variances)

    # Decide target trials-per-setting (R_req rounded up), but respect max_total_runs cap
    recommended_total_per_setting = int(np.ceil(max_r_req))
    target_total_per_setting = max(pilot_runs, min(recommended_total_per_setting, max_total_runs))

    # Time-based cap: how many additional simulation-units can we do within max_wall_time_seconds
    max_additional_units_time = int((max_wall_time_seconds * cores) / max(1e-9, time_per_unit))
    units_per_setting = len(threshold_scenarios) * len(strategies) * len(seed_fractions)
    additional_runs_time_cap = max(0, max_additional_units_time // units_per_setting - pilot_runs)
    additional_runs_requested = max(0, target_total_per_setting - pilot_runs)
    additional_runs = min(additional_runs_requested, additional_runs_time_cap)

    print(f"\n[INFO] Pilot elapsed {pilot_elapsed:.1f}s -> ~{time_per_unit:.3f}s per simulation unit (cores={cores})")
    print(f"[INFO] Recommended trials-per-setting = {recommended_total_per_setting}, target (capped) = {target_total_per_setting}")
    print(f"[INFO] Time-budget allows ~{additional_runs_time_cap} additional trials-per-setting; executing {additional_runs} additional trials-per-setting.")

    if additional_runs <= 0:
        print("[INFO] No additional runs scheduled (time/cap reached). Proceeding to aggregation and return pilot results.")
        df_ts = pd.DataFrame(results_data)
        # perform aggregation & pivots (teacher-style) and save
        final_per_trial = df_ts.groupby(['Trial', 'Strategy', 'Threshold_Dist', 'Seed_Fraction'], as_index=False).agg(
            Final_Fraction=('Final_Fraction', 'max'),
            TimeToConverge=('TimeToConverge', 'max'),
            SeedsCount=('SeedsCount', 'max')
        )

        agg = final_per_trial.groupby(['Strategy', 'Threshold_Dist', 'Seed_Fraction'], as_index=False).agg(
            n_runs=('Final_Fraction', 'count'),
            FinalAdoption=('Final_Fraction', 'mean'),
            Final_STD=('Final_Fraction', 'std'),
            CascadeProb=('Final_Fraction', lambda x: np.mean(np.array(x) >= 0.5)),
            Time50=('TimeToConverge', 'mean'),
            Seeds=('SeedsCount', lambda x: int(x.mode().iloc[0]) if len(x) > 0 else B)
        )
        agg['Efficiency'] = agg['FinalAdoption'] / agg['Seeds'].replace(0, 1)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        agg_path = os.path.join(OUTPUT_DIR, "aggregated_summary_by_strategy_threshold_teacherstyle.csv")
        agg.to_csv(agg_path, index=False)
        print(f"[INFO] Saved aggregated teacher-style summary to {agg_path}")

        # pivots
        try:
            pivot_final = agg.pivot_table(index='Strategy', columns='Threshold_Dist', values='FinalAdoption', aggfunc='mean')
            pivot_final.to_csv(os.path.join(OUTPUT_DIR, "pivot_final_adoption.csv"))
            print(f"[INFO] Saved pivot_final_adoption.csv")
        except Exception:
            pass

        try:
            pivot_cascade = agg.pivot_table(index='Strategy', columns='Threshold_Dist', values='CascadeProb', aggfunc='mean')
            pivot_cascade.to_csv(os.path.join(OUTPUT_DIR, "pivot_cascade_prob.csv"))
            print(f"[INFO] Saved pivot_cascade_prob.csv")
        except Exception:
            pass

        try:
            pivot_eff = agg.pivot_table(index='Strategy', columns='Threshold_Dist', values='Efficiency', aggfunc='mean')
            pivot_eff.to_csv(os.path.join(OUTPUT_DIR, "pivot_efficiency.csv"))
            print(f"[INFO] Saved pivot_efficiency.csv")
        except Exception:
            pass

        det_path = os.path.join(OUTPUT_DIR, "detailed_per_trial_summary.csv")
        final_per_trial.to_csv(det_path, index=False)
        print(f"[INFO] Saved detailed per-trial summary to {det_path}")

        return df_ts

    # --- ADDITIONAL RUNS PHASE ---
    print(f"\n[INFO] Running additional {additional_runs} runs per setting (may take time)...")
    pbar2 = 0
    for extra in range(additional_runs):
        trial = pilot_runs + extra
        for thresh_key in threshold_scenarios:
            for seed_frac in seed_fractions:
                trial_seed = GLOBAL_SEED + (abs(hash(thresh_key)) % 1000) + int(seed_frac * 1000) + trial * 1009
                random.seed(trial_seed); np.random.seed(trial_seed)
                thresholds = get_thresholds(G, thresh_key)

                for strat in strategies:
                    seeds = get_seeds(G, strat, seed_frac, bc_map)
                    curve, _ = run_threshold_simulation(G, seeds, thresholds)
                    final_frac = float(curve[-1])

                    arr = np.array(curve)
                    changes = np.where(np.diff(arr) != 0)[0]
                    time_to_converge = int(changes[-1] + 1) if len(changes) else 0

                    for t, val in enumerate(curve):
                        results_data.append({
                            'Trial': trial,
                            'Strategy': strat,
                            'Threshold_Dist': thresh_key,
                            'Seed_Fraction': seed_frac,
                            'Step': int(t),
                            'Adoption_Fraction': float(val),
                            'Final_Fraction': final_frac,
                            'TimeToConverge': time_to_converge,
                            'SeedsCount': len(seeds)
                        })

                    pilot_tracker[f"{strat}_{thresh_key}_{seed_frac}"].append(final_frac)

                    pbar2 += 1
                    if pbar2 % 50 == 0:
                        total_add_units = additional_runs * len(threshold_scenarios) * len(strategies) * len(seed_fractions)
                        print(f" -> Completed {pbar2}/{total_add_units} additional simulation units...")

    # Final aggregation & save pivot tables
    df_ts = pd.DataFrame(results_data)

    final_per_trial = df_ts.groupby(['Trial', 'Strategy', 'Threshold_Dist', 'Seed_Fraction'], as_index=False).agg(
        Final_Fraction=('Final_Fraction', 'max'),
        TimeToConverge=('TimeToConverge', 'max'),
        SeedsCount=('SeedsCount', 'max')
    )

    agg = final_per_trial.groupby(['Strategy', 'Threshold_Dist', 'Seed_Fraction'], as_index=False).agg(
        n_runs=('Final_Fraction', 'count'),
        FinalAdoption=('Final_Fraction', 'mean'),
        Final_STD=('Final_Fraction', 'std'),
        CascadeProb=('Final_Fraction', lambda x: np.mean(np.array(x) >= 0.5)),
        Time50=('TimeToConverge', 'mean'),
        Seeds=('SeedsCount', lambda x: int(x.mode().iloc[0]) if len(x) > 0 else B)
    )
    agg['Efficiency'] = agg['FinalAdoption'] / agg['Seeds'].replace(0, 1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    agg_path = os.path.join(OUTPUT_DIR, "aggregated_summary_by_strategy_threshold_teacherstyle.csv")
    agg.to_csv(agg_path, index=False)
    print(f"[INFO] Saved aggregated teacher-style summary to {agg_path}")

    # Pivot tables (Strategy x Threshold)
    try:
        pivot_final = agg.pivot_table(index='Strategy', columns='Threshold_Dist', values='FinalAdoption', aggfunc='mean')
        pivot_final.to_csv(os.path.join(OUTPUT_DIR, "pivot_final_adoption.csv"))
        print(f"[INFO] Saved pivot_final_adoption.csv")
    except Exception:
        pass

    det_path = os.path.join(OUTPUT_DIR, "detailed_per_trial_summary.csv")
    final_per_trial.to_csv(det_path, index=False)
    print(f"[INFO] Saved detailed per-trial summary to {det_path}")

    # Final variance report
    final_variances = {k: np.std(v, ddof=0) for k, v in pilot_tracker.items()}
    print("\n[INFO] Final variance estimates after adaptive runs:")
    for k, v in final_variances.items():
        print(f"  {k}: std={v:.4f}")
    calculate_statistical_rigor(final_variances)

    return df_ts

# Thin wrapper to preserve call sites and provide automatic time-based capping
def execute_experiment_suite(G, bc_map, B=None):
    """
    Backwards-compatible entrypoint. Runs adaptive scheme:
      - pilot of NUM_SIMULATIONS (default 10)
      - compute required R_req and run additional trials up to time/cap limits
    Returns DataFrame of per-step timeseries (same as previous usage).
    """
    # Tunable caps (trials per setting)
    MAX_TOTAL_TRIALS_PER_SETTING = 500
    # Wall-time budget in seconds used to estimate feasible additional runs (default 10 minutes)
    MAX_WALL_TIME_SECONDS = 600

    return adaptive_execute_experiment_suite(G, bc_map,
                                            pilot_runs=NUM_SIMULATIONS,
                                            max_total_runs=MAX_TOTAL_TRIALS_PER_SETTING,
                                            max_wall_time_seconds=MAX_WALL_TIME_SECONDS)

# ==========================================
# Module 5: Visualization
# ==========================================
def setup_plot_style():
    """
    Central plotting style: consistent colorscheme, fonts, grid/legend/title behavior.
    Call this at the top of every plot_* / visualize_* function (minimal intrusion).
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.dpi": 100,
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "lines.linewidth": 1.8,
        "axes.prop_cycle": plt.cycler("color", sns.color_palette("tab10")),
    })
    sns.set_palette("tab10")

def visualize_network_structure(G):
    """
    Visualizes the Hub-and-Spoke topology (full graph; Kamada-Kawai primary).
    """
    setup_plot_style()
    if len(G) == 0:
        print("[WARN] Empty graph given; skipping plot.")
        return
    if len(G) > 3000:
        print("[WARN] Graph large; plotting full graph may be slow but will proceed.")
    print("\n[INFO] Generating Network Topology Plot (Kamada-Kawai primary, spring fallback)...")
    fig, ax = plt.subplots(figsize=(12, 12))
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception as e:
        print(f"[WARN] Kamada-Kawai layout failed ({e}). Falling back to spring_layout.")
        pos = nx.spring_layout(G, seed=GLOBAL_SEED)

    roles = nx.get_node_attributes(G, 'role')
    degrees = dict(G.degree())

    if not roles:
        deg_sorted = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        k = max(1, int(len(G) * 0.05))
        hubs = [n for n, _ in deg_sorted[:k]]
        spokes = [n for n in G.nodes() if n not in hubs]
    else:
        hubs = [n for n, r in roles.items() if r == 'Hub']
        spokes = [n for n in G.nodes() if n not in hubs]

    max_deg = max(degrees.values()) if degrees else 1
    min_size, max_size = 40, 420
    hub_sizes = [min_size + (np.sqrt(degrees.get(n, 0)) / np.sqrt(max_deg)) * (max_size - min_size) for n in hubs]
    spoke_size = 28

    import matplotlib as mpl
    cmap = mpl.colormaps.get('YlOrRd')  # warmer, more visually distinct for hubs
    norm = mpl.colors.Normalize(vmin=min([degrees[n] for n in hubs]) if hubs else 0, vmax=max_deg)
    hub_colors = [cmap(norm(degrees.get(n, 0))) for n in hubs]
    spoke_color = "#4C72B0"  # muted blue for spokes

    nx.draw_networkx_edges(G, pos, alpha=0.06, edge_color="#888888", ax=ax, width=0.6)

    spokes_coll = nx.draw_networkx_nodes(
        G, pos,
        nodelist=spokes,
        node_size=spoke_size,
        node_color=spoke_color,
        alpha=0.55,
        linewidths=0.0,
        ax=ax,
        label="Spokes",
    )
    if hasattr(spokes_coll, "set_zorder"):
        spokes_coll.set_zorder(1)

    hubs_coll = None
    if hubs:
        hubs_coll = nx.draw_networkx_nodes(
            G, pos,
            nodelist=hubs,
            node_size=hub_sizes,
            node_color=hub_colors,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
            ax=ax,
            label="Hubs"
        )
        if hasattr(hubs_coll, "set_zorder"):
            hubs_coll.set_zorder(2)

    from matplotlib.patches import Patch, Circle
    handles = [Patch(facecolor=spoke_color, label="Spokes (many)")]
    if hubs:
        rep_color = cmap(0.8)
        rep_handle = Circle((0, 0), radius=6, facecolor=rep_color, edgecolor="black", label="Hubs (size ∝ √degree)")
        handles.append(rep_handle)

    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.9)
    ax.set_title(f"YouTube Hub-and-Spoke Sample (N={len(G)})", fontsize=16, fontweight="semibold")
    ax.set_axis_off()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "network_topology.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"[INFO] Saved topology plot to {out_path}")

def plot_comparative_curves(df):
    """
    Robust plotting: mean±CI time series and violin/box diagnostics.
    """
    setup_plot_style()
    print("[INFO] Generating Comparative Adoption Curves and Diagnostics (combined)...")

    df = df.copy()
    df['Step'] = df['Step'].astype(int)

    scenarios = sorted(df['Threshold_Dist'].unique())
    strategies = ['Random', 'Degree', 'Betweenness']

    if len(scenarios) == 0:
        print("[WARN] No scenarios found in df; aborting comparative curves.")
        return

    steps = list(range(MAX_STEPS))
    z = st.norm.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2)
    grouped = df.groupby(['Strategy', 'Threshold_Dist', 'Step'])['Adoption_Fraction'].agg(['mean', 'std', 'count']).reset_index()

    # choose grid
    n = len(scenarios)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    # Create subplots and normalize axes to 1D flattened list
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()

    # If for any reason axes are fewer than scenarios, recreate with single-column layout
    if axes_flat.size < n:
        ncols = 1
        nrows = n
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), sharey=True)
        axes_flat = np.atleast_1d(axes).flatten()

    styles = {'Random': ':', 'Degree': '--', 'Betweenness': '-'}
    palette_map = dict(zip(strategies, sns.color_palette("tab10", len(strategies))))

    for idx, scenario in enumerate(scenarios):
        ax = axes_flat[idx]
        for strat in strategies:
            sub = grouped[(grouped['Threshold_Dist'] == scenario) & (grouped['Strategy'] == strat)].set_index('Step')
            means, lo_ci, hi_ci = [], [], []
            prev_mean = 0.0
            for s in steps:
                if s in sub.index:
                    row = sub.loc[s]
                    mean = float(row['mean'])
                    ncnt = int(row['count'])
                    std = float(row['std']) if ncnt > 1 and not np.isnan(row['std']) else 0.0
                    sem = std / np.sqrt(ncnt) if ncnt > 0 else 0.0
                    ci_half = z * sem
                    lo = mean - ci_half
                    hi = mean + ci_half
                else:
                    mean = prev_mean
                    lo = prev_mean
                    hi = prev_mean
                means.append(float(np.clip(mean, 0.0, 1.0)))
                lo_ci.append(float(np.clip(lo, 0.0, 1.0)))
                hi_ci.append(float(np.clip(hi, 0.0, 1.0)))
                prev_mean = mean

            ax.plot(steps, means, label=strat, linestyle=styles.get(strat, '-'),
                    color=palette_map[strat], lw=2)
            ax.fill_between(steps, lo_ci, hi_ci, color=palette_map[strat], alpha=0.15)

        ax.set_title(f"Scenario: {scenario}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time Step")
        if idx % ncols == 0:
            ax.set_ylabel("Adoption Fraction")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        if idx == 0:
            ax.legend(title="Seeding Strategy", loc='lower right')

    # hide any extra axes (if subplot grid larger than needed)
    for j in range(n, axes_flat.size):
        try:
            axes_flat[j].axis('off')
        except Exception:
            pass

    plt.suptitle("Dynamics of Contagion: Strategy vs. Threshold Heterogeneity (mean ± CI)", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(OUTPUT_DIR, "adoption_curves_comparison_fixed.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"[INFO] Saved mean+CI curves to {out_path}")

    # Prepare final_df (one final value per trial)
    if 'Final_Fraction' in df.columns:
        final_df = df[['Trial', 'Strategy', 'Threshold_Dist', 'Final_Fraction', 'TimeToConverge']].drop_duplicates()
    else:
        final_df = df.groupby(['Trial', 'Strategy', 'Threshold_Dist'], as_index=False)['Adoption_Fraction'].last().rename(
            columns={'Adoption_Fraction': 'Final_Fraction'}
        )
        if 'TimeToConverge' in df.columns:
            ttc = df.groupby(['Trial', 'Strategy', 'Threshold_Dist'], as_index=False)['TimeToConverge'].max()
            final_df = final_df.merge(ttc, on=['Trial', 'Strategy', 'Threshold_Dist'], how='left')

    if final_df.empty:
        print("[WARN] No per-trial final values available for box/violin plots - skipping.")
        return

    # Violin + Box plots for Final_Fraction
    try:
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=final_df, x='Threshold_Dist', y='Final_Fraction', hue='Strategy',
                       order=scenarios, hue_order=strategies, inner='quartile', palette="Set2", split=False)
        plt.title("Final Adoption Fraction by Threshold Scenario and Seeding Strategy (violin)")
        plt.xlabel("Threshold Distribution"); plt.ylabel("Final Adoption Fraction")
        plt.legend(loc='upper right')
        out_path = os.path.join(OUTPUT_DIR, "violin_final_fraction.png")
        plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.show()
        print(f"[INFO] Saved violin final fractions to {out_path}")
    except Exception as e:
        print(f"[WARN] violin final fraction failed: {e}")

    try:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=final_df, x='Threshold_Dist', y='Final_Fraction', hue='Strategy',
                    order=scenarios, hue_order=strategies, palette="Set2")
        plt.title("Final Adoption Fraction by Threshold Scenario and Seeding Strategy (boxplot)")
        plt.xlabel("Threshold Distribution"); plt.ylabel("Final Adoption Fraction")
        plt.legend(loc='upper right')
        out_path = os.path.join(OUTPUT_DIR, "boxplot_final_fraction.png")
        plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.show()
        print(f"[INFO] Saved boxplot final fractions to {out_path}")
    except Exception as e:
        print(f"[WARN] boxplot final fraction failed: {e}")

    # TimeToConverge diagnostics
    if 'TimeToConverge' in final_df.columns and not final_df['TimeToConverge'].isna().all():
        try:
            plt.figure(figsize=(12, 6))
            sns.violinplot(data=final_df, x='Threshold_Dist', y='TimeToConverge', hue='Strategy',
                           order=scenarios, hue_order=strategies, inner='quartile', palette="Set3", split=False)
            plt.title("Time to Convergence by Threshold Scenario & Strategy (violin)")
            plt.xlabel("Threshold Distribution"); plt.ylabel("Time to Convergence (steps)")
            plt.legend(loc='upper right')
            out_path = os.path.join(OUTPUT_DIR, "violin_time_to_converge.png")
            plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.show()
            print(f"[INFO] Saved violin time-to-convergence to {out_path}")
        except Exception as e:
            print(f"[WARN] violin time-to-converge failed: {e}")

        try:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=final_df, x='Threshold_Dist', y='TimeToConverge', hue='Strategy',
                        order=scenarios, hue_order=strategies, palette="Set3")
            plt.title("Time to Convergence by Threshold Scenario & Strategy (boxplot)")
            plt.xlabel("Threshold Distribution"); plt.ylabel("Time to Convergence (steps)")
            plt.legend(loc='upper right')
            out_path = os.path.join(OUTPUT_DIR, "boxplot_time_to_converge.png")
            plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.show()
            print(f"[INFO] Saved boxplot time-to-convergence to {out_path}")
        except Exception as e:
            print(f"[WARN] boxplot time-to-converge failed: {e}")
    else:
        print("[WARN] 'TimeToConverge' missing or empty: skipping time-to-convergence plots")

    # Summary table (aggregated stats) saved
    value_col = 'Final_Fraction' if 'Final_Fraction' in df.columns else 'Adoption_Fraction'
    time_col = 'TimeToConverge' if 'TimeToConverge' in df.columns else None

    agg_dict = {
        'Final_Mean': (value_col, 'mean'),
        'Final_STD': (value_col, 'std'),
    }
    if time_col:
        agg_dict.update({'TimeToConv_Mean': (time_col, 'mean'), 'TimeToConv_STD': (time_col, 'std')})
    else:
        agg_dict.update({'TimeToConv_Mean': ('Step', 'max'), 'TimeToConv_STD': ('Step', 'max')})

    summary_table = df.groupby(['Strategy', 'Threshold_Dist']).agg(**agg_dict).reset_index()
    summary_csv = os.path.join(OUTPUT_DIR, "summary_stats_by_strategy_threshold.csv")
    summary_table.to_csv(summary_csv, index=False)
    print(f"[INFO] Saved summary table to {summary_csv}")

def create_summary_tables(df, out_dir=OUTPUT_DIR):
    """
    Create CSV tables useful for the report:
      - aggregated summary by Strategy x Threshold_Dist (mean, std, median, q25, q75, n)
      - ranking per Threshold_Dist by mean final adoption
      - per-trial detailed summary (Trial, Strategy, Threshold_Dist, Final_Fraction, TimeToConverge)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Ensure we have Final_Fraction per trial
    if 'Final_Fraction' not in df.columns:
        final = df.groupby(['Trial', 'Strategy', 'Threshold_Dist'], as_index=False)['Adoption_Fraction'].last()
        final = final.rename(columns={'Adoption_Fraction': 'Final_Fraction'})
    else:
        final = df[['Trial', 'Strategy', 'Threshold_Dist', 'Final_Fraction', 'TimeToConverge']].drop_duplicates()

    det_path = os.path.join(out_dir, "detailed_per_trial_summary.csv")
    final.to_csv(det_path, index=False)

    # Aggregated summary with descriptive stats
    agg = final.groupby(['Strategy', 'Threshold_Dist']).agg(
        n_runs=('Final_Fraction', 'count'),
        mean_final=('Final_Fraction', 'mean'),
        std_final=('Final_Fraction', 'std'),
        median_final=('Final_Fraction', 'median'),
        q25_final=('Final_Fraction', lambda x: np.percentile(x, 25)),
        q75_final=('Final_Fraction', lambda x: np.percentile(x, 75)),
        mean_time_to_conv=('TimeToConverge', 'mean'),
        std_time_to_conv=('TimeToConverge', 'std')
    ).reset_index()

    agg_path = os.path.join(out_dir, "aggregated_summary_by_strategy_threshold.csv")
    agg.to_csv(agg_path, index=False)

    # Ranking per threshold scenario
    ranks = agg.sort_values(['Threshold_Dist', 'mean_final'], ascending=[True, False])
    rank_path = os.path.join(out_dir, "ranking_by_threshold.csv")
    ranks.to_csv(rank_path, index=False)

    print(f"[INFO] Saved detailed ({det_path}), aggregated ({agg_path}), ranking ({rank_path})")
    return {'detailed': det_path, 'aggregated': agg_path, 'ranking': rank_path}

def plot_degree_histogram(G, bins='auto', log_scale=True, out_name="degree_hist.png"):
    setup_plot_style()
    degs = np.array([d for _, d in G.degree()])
    plt.figure(figsize=(8,4))
    sns.histplot(degs, bins=bins, kde=False, color="#4C72B0")
    plt.xlabel("Degree"); plt.ylabel("Count"); plt.title("Degree distribution")
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
    print(f"[INFO] Saved degree histogram to {out_path}")

def plot_degree_ccdf(G, out_name="degree_ccdf.png"):
    setup_plot_style()
    degs = np.array([d for _, d in G.degree()])
    vals, counts = np.unique(degs, return_counts=True)
    pdf = counts / counts.sum()
    ccdf = 1 - np.cumsum(pdf) + pdf  # complementary cumulative
    plt.figure(figsize=(6,5))
    plt.loglog(vals[vals>0], ccdf[vals>0], marker='o', linestyle='none')
    plt.xlabel("Degree (log)"); plt.ylabel("CCDF (log)"); plt.title("Degree CCDF (log-log)")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
    print(f"[INFO] Saved degree CCDF to {out_path}")

def plot_thresholds_grid(df=None, G=None, scenarios=None, cols=3, bins=30, kde=True, out_name="thresholds_grid.png"):
    setup_plot_style()
    if G is None:
        raise ValueError("Graph G must be provided to sample thresholds")

    if scenarios is None:
        if isinstance(df, pd.DataFrame) and 'Threshold_Dist' in df.columns:
            scenarios = sorted(df['Threshold_Dist'].unique())
        else:
            fixed_val = globals().get('DEFAULT_FIXED_THRESHOLD', 0.2)
            sweep_vals = globals().get('CONSTANT_SWEEP', [0.4])
            scenarios = [
                f"Fixed_{fixed_val}",
                "Uniform",
                "Beta_Low",
                "Beta_Symmetric",
                "Beta_High",
                "Normal"
            ] + [f"Constant_{v}" for v in sweep_vals]

    if len(scenarios) == 0:
        print("[WARN] No threshold scenarios found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n = len(scenarios)
    ncols = max(1, int(cols))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)

    for idx, scenario in enumerate(scenarios):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        np.random.seed(GLOBAL_SEED + idx)
        try:
            thr = get_thresholds(G, scenario)
            vals = np.array(list(thr.values()))
        except Exception:
            vals = np.array([])

        if vals.size:
            sns.histplot(vals, bins=bins, kde=kde, ax=ax, color="#2CA02C")
            ax.set_xlim(0, 1)
        else:
            ax.text(0.5, 0.5, "no thresholds", ha='center', va='center')

        ax.set_title(scenario, fontsize=10)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Count")

    total_axes = nrows * ncols
    for j in range(n, total_axes):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis('off')

    plt.suptitle("Threshold distributions — one subplot per scenario", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"[INFO] Saved thresholds grid to {out_path}")

def plot_final_fraction_grid(df_results, out_name="final_fraction_grid.png", cols=5, kind=None):
    """
    Grid: rows = Strategy, cols = Threshold scenario.
    If kind is 'violin' or 'box' draw only that; otherwise draw both violin and box versions.
    """
    setup_plot_style()

    if df_results is None or df_results.empty:
        print("[WARN] Empty df_results for final fraction grid"); return

    # ensure final fraction per trial
    if 'Final_Fraction' not in df_results.columns:
        final_df = df_results.groupby(['Trial', 'Strategy', 'Threshold_Dist'], as_index=False)['Adoption_Fraction'] \
                    .last().rename(columns={'Adoption_Fraction': 'Final_Fraction'})
    else:
        final_df = df_results[['Trial', 'Strategy', 'Threshold_Dist', 'Final_Fraction']].drop_duplicates()

    strategies = sorted(final_df['Strategy'].unique())
    scenarios = sorted(final_df['Threshold_Dist'].unique())
    if not strategies or not scenarios:
        print("[WARN] No strategies or scenarios found for final fraction grid"); return

    # layout params
    ncols = min(len(scenarios), int(cols)) if cols > 0 else len(scenarios)
    ncols = max(1, ncols)
    nrows = max(1, len(strategies))

    # palette consistent across both figures
    palette = dict(zip(strategies, sns.color_palette("tab10", n_colors=max(3, len(strategies)))))

    from matplotlib.patches import Patch

    def _draw_grid(draw_kind, fig_title, save_path):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows),
                                 squeeze=False, sharey='row')
        for i, strat in enumerate(strategies):
            for j in range(ncols):
                scen = scenarios[j] if j < len(scenarios) else None
                ax = axes[i][j]
                if scen is None:
                    ax.axis('off'); continue
                sub = final_df[(final_df['Strategy'] == strat) & (final_df['Threshold_Dist'] == scen)]
                if sub.empty:
                    ax.text(0.5, 0.5, "no data", ha='center', va='center', fontsize=9)
                    ax.set_xticks([])
                else:
                    color = palette.get(strat, "#777777")
                    if draw_kind == 'violin':
                        sns.violinplot(data=sub, y='Final_Fraction', color=color, inner='quartile', ax=ax)
                    else:  # 'box'
                        sns.boxplot(data=sub, y='Final_Fraction', color=color, ax=ax)
                    ax.set_ylim(0, 1)
                    ax.grid(True, alpha=0.22, linestyle='--')

                if i == 0:
                    ax.set_title(scen, fontsize=10, fontweight='semibold')
                if j == 0:
                    ax.set_ylabel(strat, fontsize=9, fontweight='semibold')
                else:
                    ax.set_ylabel("")
                ax.set_xlabel("")

        fig.suptitle(fig_title, fontsize=13, y=0.98)
        legend_handles = [Patch(facecolor=palette[s], label=s) for s in strategies]
        fig.legend(handles=legend_handles, title="Seeding Strategy", loc='upper right', frameon=True, framealpha=0.9)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f"[INFO] Saved {draw_kind} final-fraction grid to {save_path}")

    base, ext = os.path.splitext(out_name)
    violin_path = f"{base}_violin{ext}"
    box_path = f"{base}_box{ext}"

    if kind is None:
        _draw_grid('violin', "Final adoption (violin) — rows: Strategy, cols: Threshold scenario", violin_path)
        _draw_grid('box', "Final adoption (boxplots) — rows: Strategy, cols: Threshold scenario", box_path)
    elif kind == 'violin':
        _draw_grid('violin', "Final adoption (violin) — rows: Strategy, cols: Threshold scenario", out_name)
    elif kind == 'box':
        _draw_grid('box', "Final adoption (boxplots) — rows: Strategy, cols: Threshold scenario", out_name)
    else:
        print(f"[WARN] Unknown kind={kind}; expected 'violin' or 'box'")

def plot_time_to_converge_grid(df_results, out_name="time_to_converge_grid.png", cols=5, bins=12):
    setup_plot_style()

    if df_results is None or df_results.empty:
        print("[WARN] Empty df_results for time-to-converge grid"); return

    if 'TimeToConverge' not in df_results.columns:
        print("[WARN] TimeToConverge missing in df_results; skipping time-to-converge grid."); return

    tdf = df_results.groupby(['Trial','Strategy','Threshold_Dist'], as_index=False)['TimeToConverge'].max()

    strategies = sorted(tdf['Strategy'].unique())
    scenarios = sorted(tdf['Threshold_Dist'].unique())
    if not strategies or not scenarios:
        print("[WARN] No strategies or scenarios found for time-to-converge grid"); return

    ncols = min(len(scenarios), int(cols))
    ncols = max(1, ncols)
    nrows = len(strategies)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4*ncols, 2.6*nrows), squeeze=False)

    for i, strat in enumerate(strategies):
        for j in range(ncols):
            scen = scenarios[j] if j < len(scenarios) else None
            ax = axes[i][j]
            if scen is None:
                ax.axis('off'); continue
            sub = tdf[(tdf['Strategy']==strat) & (tdf['Threshold_Dist']==scen)]
            if sub.empty:
                ax.text(0.5, 0.5, "no data", ha='center', va='center', fontsize=9)
                ax.set_xticks([])
            else:
                sns.histplot(sub['TimeToConverge'], bins=bins, kde=False, color="#9467BD", ax=ax)
                ax.grid(True, alpha=0.2)
            if i == 0:
                ax.set_title(scen, fontsize=9, fontweight='semibold')
            if j == 0:
                ax.set_ylabel(strat, fontsize=9, fontweight='semibold')
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")

    plt.suptitle("Time to converge — rows: Strategy, cols: Threshold scenario", fontsize=12, y=0.98)
    plt.tight_layout(rect=[0,0,1,0.95])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.show(); plt.close(fig)
    print(f"[INFO] Saved time-to-converge grid to {out_path}")

def plot_degree_vs_betweenness(G, bc_map, out_name="deg_vs_bc.png"):
    setup_plot_style()
    deg = dict(G.degree())
    nodes = [n for n in G.nodes() if n in bc_map]
    x = [deg[n] for n in nodes]; y = [bc_map[n] for n in nodes]
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Degree (log)"); plt.ylabel("Betweenness (log)"); plt.title("Degree vs Betweenness")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
    print(f"[INFO] Saved degree vs betweenness scatter to {out_path}")

def plot_pivot_heatmap(df_results, value_col='Final_Fraction', aggfunc='mean', out_name="pivot_heatmap.png"):
    setup_plot_style()
    if value_col not in df_results.columns:
        final_df = df_results.groupby(['Trial','Strategy','Threshold_Dist'], as_index=False)['Adoption_Fraction'].last().rename(columns={'Adoption_Fraction':'Final_Fraction'})
    else:
        final_df = df_results[['Trial','Strategy','Threshold_Dist',value_col]].drop_duplicates()
    pivot = final_df.pivot_table(index='Strategy', columns='Threshold_Dist', values=value_col, aggfunc=aggfunc)
    plt.figure(figsize=(8,4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"Pivot heatmap ({value_col})")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
    print(f"[INFO] Saved pivot heatmap to {out_path}")

# ==========================================
# Main Controller (script entry)
# ==========================================
def execute_pipeline(sample_arg=None, run_plots=True):
    """
    Convenience wrapper to run full pipeline:
      1) load data (or surrogate)
      2) sample hub-and-spoke subgraph
      3) precompute centrality on sample
      4) visualize topology
      5) run adaptive experiment suite and save results
      6) produce summary tables and plots
    Parameter:
      - sample_arg: overrides N_TARGET_SUBGRAPH (int, float, or percent string)
      - run_plots: whether to generate and save plots
    Returns:
      - sampled graph (for interactive inspection)
    """
    G_raw = load_network_data(INPUT_FILE)
    G_sample = hub_and_spoke_sampling(G_raw, sample_arg if sample_arg is not None else N_TARGET_SUBGRAPH)
    # Use approximate centrality by default (faster)
    bc_map = precompute_centrality(G_sample, approx_k=None)
    visualize_network_structure(G_sample)
    df_results = adaptive_execute_experiment_suite(G_sample, bc_map, pilot_runs=NUM_SIMULATIONS)
    if isinstance(df_results, pd.DataFrame) and not df_results.empty:
        csv_path = os.path.join(OUTPUT_DIR, RESULTS_CSV)
        df_results.to_csv(csv_path, index=False)
        print(f"[INFO] Raw data saved to {csv_path}")
        try:
            create_summary_tables(df_results, out_dir=OUTPUT_DIR)
        except Exception as e:
            print(f"[WARN] create_summary_tables failed: {e}")
        if run_plots:
            try:
                plot_comparative_curves(df_results)
            except Exception as e:
                print(f"[WARN] plot_comparative_curves failed: {e}")
            try:
                plot_thresholds_grid(df_results, G_sample, cols=3, out_name="thresholds_grid_pretty.png")
            except Exception:
                pass
            try:
                plot_final_fraction_grid(df_results, out_name="final_fraction_grid.png", cols=4)
            except Exception:
                pass
            try:
                plot_time_to_converge_grid(df_results, out_name="time_to_converge_grid.png", cols=4)
            except Exception:
                pass
            try:
                plot_pivot_heatmap(df_results, value_col='Final_Fraction', out_name="pivot_heatmap_pretty.png")
            except Exception:
                pass
            try:
                plot_degree_vs_betweenness(G_sample, bc_map, out_name="deg_vs_bc_pretty.png")
            except Exception:
                pass
    else:
        print("[WARN] No results returned from execute_experiment_suite; skipping save/plots.")
    return G_sample

if __name__ == "__main__":
    # Optional CLI: allow specifying N_TARGET_SUBGRAPH via first arg (int, float, or percent-string)
    sample_arg = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        try:
            if isinstance(arg, str) and arg.endswith('%'):
                sample_arg = arg
            elif '.' in arg:
                sample_arg = float(arg)
            else:
                sample_arg = int(arg)
        except Exception:
            sample_arg = None
    # Execute full pipeline and generate outputs in simulation_outputs/
    execute_pipeline(sample_arg=sample_arg, run_plots=True)
