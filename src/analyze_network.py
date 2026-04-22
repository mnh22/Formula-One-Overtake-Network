import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from networkx.algorithms import community
from scipy.stats import linregress
from build_network import build_overtake_network, G_2025

drivers = pd.read_csv('data/original_datasets/drivers.csv')
ot = pd.read_csv('data/analysis_datasets/overtakes.csv')



# prep driver information to add for easy review
drivers_copy = drivers.copy()
drivers_copy['full_name'] = drivers_copy['forename'] + ' ' + drivers_copy['surname']

# use driver surname each node label
drivers_copy['label'] = drivers_copy['surname']

# compute the basic network details
def compute_basic_stats(G):
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "average_out_degree": np.mean([d for n, d in G.out_degree()]),
        "average_in_degree": np.mean([d for n, d in G.in_degree()]),
        "reciprocity": nx.reciprocity(G),
    }
    
    # try to compute clustering and average path length (may fail if not connected)
    try:
        undirected_G = G.to_undirected()
        stats["avg_clustering"] = nx.average_clustering(undirected_G)
    except:
        stats["avg_clustering"] = None
    
    try:
        # only compute if strongly connected, otherwise use largest component
        if nx.is_strongly_connected(G):
            stats["avg_path_length"] = nx.average_shortest_path_length(G)
        else:
            largest_cc = max(nx.strongly_connected_components(G), key=len)
            subG = G.subgraph(largest_cc)
            stats["avg_path_length"] = nx.average_shortest_path_length(subG)
            stats["largest_component_size"] = len(largest_cc)
    except:
        stats["avg_path_length"] = None
        stats["largest_component_size"] = None
    
    return stats

# compute and plot degree distributions (for both out and in degrees)
def compute_and_plot_degree_dist(G, title_prefix="", save_path="results", use_weight=True, log_log=False, bins=20):
    
    if use_weight:
        degrees_out = [d for n, d in G.out_degree(weight='weight')]
        degrees_in = [d for n, d in G.in_degree(weight='weight')]
    else:
        degrees_out = [d for n, d in G.out_degree()]
        degrees_in = [d for n, d in G.in_degree()]

    # binning - use density=False and normalize manually for true probability
    counts_out, bin_edges_out = np.histogram(degrees_out, bins=bins, density=False)
    counts_in, bin_edges_in = np.histogram(degrees_in, bins=bins, density=False)
    
    # normalize to get probability: P(k) = count / total_nodes
    counts_out = counts_out / len(degrees_out)
    counts_in = counts_in / len(degrees_in)

    bin_centers_out = (bin_edges_out[:-1] + bin_edges_out[1:]) / 2
    bin_centers_in = (bin_edges_in[:-1] + bin_edges_in[1:]) / 2

    # filter out zero values for log-log fitting
    if log_log:
        mask_out = (bin_centers_out > 0) & (counts_out > 0)
        mask_in = (bin_centers_in > 0) & (counts_in > 0)
        
        # fit power law in log space for out-degree
        if mask_out.sum() > 1:
            log_x_out = np.log10(bin_centers_out[mask_out])
            log_y_out = np.log10(counts_out[mask_out])
            slope_out, intercept_out, r_out, _, _ = linregress(log_x_out, log_y_out)
            fit_out = 10**(intercept_out) * bin_centers_out[mask_out]**slope_out
        else:
            slope_out, r_out, fit_out = None, None, None
            
        # fit power law in log space for in-degree  
        if mask_in.sum() > 1:
            log_x_in = np.log10(bin_centers_in[mask_in])
            log_y_in = np.log10(counts_in[mask_in])
            slope_in, intercept_in, r_in, _, _ = linregress(log_x_in, log_y_in)
            fit_in = 10**(intercept_in) * bin_centers_in[mask_in]**slope_in
        else:
            slope_in, r_in, fit_in = None, None, None

    else:
        fit_out=None


    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # out-degree scatter
    axes[0].scatter(bin_centers_out, counts_out, alpha=0.7, color='tab:blue')

    if log_log and fit_out is not None:
        axes[0].plot(bin_centers_out[mask_out], fit_out, 'r--', alpha=0.8, linewidth=2,
                    label=f'Fit: γ={slope_out:.2f}, R²={r_out**2:.3f}')

    axes[0].set_xlabel("Out-degree")
    axes[0].set_ylabel("Probability P(k)")
    axes[0].set_title(f"{title_prefix} Out-degree Distribution{' (log-log)' if log_log else ''}")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    if log_log:
        axes[0].legend()
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')


    # in-degree scatter
    axes[1].scatter(bin_centers_in, counts_in, alpha=0.7, color='tab:orange')
    if log_log and fit_in is not None:
        axes[1].plot(bin_centers_in[mask_in], fit_in, 'r--', alpha=0.8, linewidth=2,
                    label=f'Fit: γ={slope_in:.2f}, R²={r_in**2:.3f}')
    axes[1].set_xlabel("In-degree")
    axes[1].set_ylabel("Probability P(k)")
    axes[1].set_title(f"{title_prefix} In-degree Distribution{' (log-log)' if log_log else ''}")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    if log_log:
        axes[1].legend()
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')

    plt.suptitle(f"{title_prefix} Degree Probability Distributions{' (log-log)' if log_log else ''}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f"{save_path}/degree_probability_scatter_{'Weighted' if use_weight else 'Unweighted'}{'_log_log' if log_log else ''}.png"
    plt.savefig(fname)
    plt.close()

    return degrees_out, degrees_in


def compute_and_plot_friendship_paradox(G, title="", use_weight=True, log_log=False, save_path="results", axis_limit=None):

    # compute out-degree (weighted or unweighted)
    if use_weight:
        out_deg = dict(G.out_degree(weight='weight'))
    else:
        out_deg = dict(G.out_degree())

    # compute neighbor average out-degree for each node
    neighbor_out_deg = {}
    for node in G.nodes():
        neighbors = list(G.successors(node))
        if neighbors:
            neighbor_avg = np.mean([out_deg[n] for n in neighbors])
        else:
            neighbor_avg = 0
        neighbor_out_deg[node] = neighbor_avg

    # aggregate by node degree to compute knn(k)
    degree_to_neighbors = {}
    for node, deg in out_deg.items():
        if deg not in degree_to_neighbors:
            degree_to_neighbors[deg] = []
        degree_to_neighbors[deg].append(neighbor_out_deg[node])

    # compute k and knn(k) for plotting
    k_vals = []
    knn_vals = []
    for k in sorted(degree_to_neighbors.keys()):
        k_vals.append(k)
        knn_vals.append(np.mean(degree_to_neighbors[k]))

    k_vals = np.array(k_vals)
    knn_vals = np.array(knn_vals)

    # overall averages and ratio
    avg_out = np.mean(list(out_deg.values()))
    avg_neighbor_out = np.mean(list(neighbor_out_deg.values()))
    ratio = avg_neighbor_out / avg_out
    difference = avg_neighbor_out - avg_out

    # create figure with two subplots presented side by side for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # left plot: k vs knn(k) scatter
    ax1.scatter(k_vals, knn_vals, alpha=0.7, s=50, color='tab:purple')
    ax1.axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Node Degree (k)")
    ax1.set_ylabel("Avg Neighbor Degree knn(k)")
    ax1.set_title(f"{title} Friendship Paradox: k vs knn(k)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    if log_log:
        ax1.legend()
        ax1.set_xscale('log')
        ax1.set_yscale('log')

    # right plot: binned histogram comparison (similar to degree distribution plot)
    max_val = int(max(k_vals.max(), knn_vals.max(), 1)) + 1
    bins = np.linspace(0, max_val, min(30, max_val))

    # histogram for k (node degrees)
    counts_k, edges_k = np.histogram(list(out_deg.values()), bins=bins, density=True)
    bin_centers_k = (edges_k[:-1] + edges_k[1:]) / 2
    
    # histogram for knn (neighbor degrees)
    neighbor_deg_list = list(neighbor_out_deg.values())
    counts_knn, edges_knn = np.histogram(neighbor_deg_list, bins=bins, density=True)
    bin_centers_knn = (edges_knn[:-1] + edges_knn[1:]) / 2

    ax2.plot(bin_centers_k, counts_k, 'o-', label='Node Degree', color='tab:blue', alpha=0.7)
    ax2.plot(bin_centers_knn, counts_knn, 's-', label='Avg Neighbor Degree', color='tab:green', alpha=0.7)
    
    ax2.axvline(avg_out, color='tab:blue', linestyle=':', alpha=0.5, label=f'Avg k={avg_out:.1f}')
    ax2.axvline(avg_neighbor_out, color='tab:green', linestyle=':', alpha=0.5, label=f'Avg knn={avg_neighbor_out:.1f}')
    
    ax2.set_xlabel("Degree Value")
    ax2.set_ylabel("Probability P(k)")
    ax2.set_title(f"{title} Degree Distribution Comparison")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_ylim(bottom=0)  # Bound y-axis at 0

    if log_log:
        ax2.legend(fontsize=10)
        ax2.set_xscale('log')
        ax2.set_yscale('log')

    if axis_limit:
        ax2.set_xlim(axis_limit)

    plt.suptitle(f"{title} Friendship Paradox Analysis ({'Weighted' if use_weight else 'Unweighted'})")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fname = f"{save_path}/friendship_paradox_{'weighted' if use_weight else 'unweighted'}{'_log_log' if log_log else ''}.png"
    plt.savefig(fname)
    plt.close()

    return {"avg_out": avg_out, "avg_neighbor_out": avg_neighbor_out, "ratio": ratio, "difference": difference}


def detect_communities(G):
    # convert to undirected for community detection
    undirected_G = G.to_undirected()
    
    # use Louvain method (compare with Gephi results later)
    communities = community.louvain_communities(undirected_G, seed=42)  # seed for reproducibility
    modularity = community.modularity(undirected_G, communities)
    
    return {"communities": communities, "modularity": modularity}


def compute_out_in_assortativity(G):
    r = nx.degree_pearson_correlation_coefficient(G, x='out', y='in')
    return r

def compute_out_out_assortativity(G):
    r = nx.degree_pearson_correlation_coefficient(G, x='out', y='out')
    return r

def compute_in_in_assortativity(G):
    r = nx.degree_pearson_correlation_coefficient(G, x='in', y='in')
    return r

def compute_in_out_assortativity(G):
    r = nx.degree_pearson_correlation_coefficient(G, x='in', y='out')
    return r

# definition to do complete analysis for each G network that is passed into the function
# create summaries for each individual network
def analyze_network(G, title, save_path, use_weight=True, log_log=False, fp_axis_limit=None):
    summary = {}

    # basic stats
    summary["basic_stats"] = compute_basic_stats(G)

    # degree distributions
    summary["degree_distributions"] = compute_and_plot_degree_dist(
        G, title_prefix=title, save_path=save_path, use_weight=use_weight, log_log=log_log
    )
    
    # friendship paradox weighted
    summary["friendship_paradox_weighted"] = compute_and_plot_friendship_paradox(
        G, title, use_weight=True, save_path=save_path
    )

    # friendship paradox unweighted
    summary["friendship_paradox_unweighted"] = compute_and_plot_friendship_paradox(
        G, title, use_weight=False, save_path=save_path, axis_limit=fp_axis_limit
    )
    # communities and modularity
    summary["communities"] = detect_communities(G)

    # assortativity
    summary["out_in_assortativity"] = compute_out_in_assortativity(G)
    summary["out_out_assortativity"] = compute_out_out_assortativity(G)
    summary["in_in_assortativity"] = compute_in_in_assortativity(G)
    summary["in_out_assortativity"] = compute_in_out_assortativity(G)

    return summary


def print_summary(summary, title="Network Summary"):
    print(f"\n=== {title} ===")
    stats = summary["basic_stats"]
    print(f"Nodes: {stats['nodes']}, Edges: {stats['edges']}, Density: {stats['density']:.4f}")
    print(f"Average out-degree: {stats['average_out_degree']:.2f}, Average in-degree: {stats['average_in_degree']:.2f}")
    print(f"Reciprocity: {stats['reciprocity']:.3f}")
    
    if stats['avg_clustering'] is not None:
        print(f"Average clustering: {stats['avg_clustering']:.3f}")
    if stats['avg_path_length'] is not None:
        print(f"Average path length: {stats['avg_path_length']:.2f}")
        if stats.get('largest_component_size'):
            print(f"(computed on largest strongly connected component: {stats['largest_component_size']} nodes)")
    
    fp_w = summary["friendship_paradox_weighted"]
    fp_u = summary["friendship_paradox_unweighted"]
    print(f"Friendship Paradox (Weighted): Avg Out {fp_w['avg_out']:.2f}, Avg Neighbor Out {fp_w['avg_neighbor_out']:.2f}, Ratio {fp_w['ratio']:.2f}, Difference {fp_w['difference']:.2f}")
    print(f"Friendship Paradox (Unweighted): Avg Out {fp_u['avg_out']:.2f}, Avg Neighbor Out {fp_u['avg_neighbor_out']:.2f}, Ratio {fp_u['ratio']:.2f}, Difference {fp_u['difference']:.2f}")

    # assortativity
    out_in_assortativity = summary["out_in_assortativity"]
    print(f"Out→In Assortativity: {out_in_assortativity:.3f}")
    out_out_assortativity = summary["out_out_assortativity"]
    print(f"Out→Out Assortativity: {out_out_assortativity:.3f}")
    in_in_assortativity = summary["in_in_assortativity"]
    print(f"In→In Assortativity: {in_in_assortativity:.3f}")
    in_out_assortativity = summary["in_out_assortativity"]
    print(f"In→Out Assortativity: {in_out_assortativity:.3f}")

    modularity = summary["communities"]["modularity"]
    print(f"Number of communities: {len(summary['communities']['communities'])}, Modularity: {modularity:.3f}")



def export_summary_to_csv(summary, save_path="results"):
    """Exports network summary stats and community list to CSV files."""

    # flatten the summary into a 1-row table
    stats = summary["basic_stats"]
    fp_w = summary["friendship_paradox_weighted"]
    fp_u = summary["friendship_paradox_unweighted"]
    comms = summary["communities"]

    flat_summary = {
        "nodes": stats["nodes"],
        "edges": stats["edges"],
        "density": stats["density"],
        "average_out_degree": stats["average_out_degree"],
        "average_in_degree": stats["average_in_degree"],
        "reciprocity": stats["reciprocity"],
        "avg_clustering": stats.get("avg_clustering"),
        "avg_path_length": stats.get("avg_path_length"),
        "largest_component_size": stats.get("largest_component_size"),

        "fp_weighted_avg_out": fp_w["avg_out"],
        "fp_weighted_avg_neighbor_out": fp_w["avg_neighbor_out"],
        "fp_weighted_difference": fp_w["difference"],
        "fp_weighted_ratio": fp_w["ratio"],

        "fp_unweighted_avg_out": fp_u["avg_out"],
        "fp_unweighted_avg_neighbor_out": fp_u["avg_neighbor_out"],
        "fp_unweighted_difference": fp_u["difference"],
        "fp_unweighted_ratio": fp_u["ratio"],

        "modularity": comms["modularity"],
        "num_communities": len(comms["communities"]),

        "out_in_assortativity": summary["out_in_assortativity"],
        "out_out_assortativity": summary["out_out_assortativity"],
        "in_in_assortativity": summary["in_in_assortativity"],
        "in_out_assortativity": summary["in_out_assortativity"]
    }

    df_summary = pd.DataFrame([flat_summary])
    df_summary.to_csv(f"{save_path}/summary_table.csv", index=False)

    # export communities (long-form) - NOW WITH SURNAMES
    community_rows = []
    for i, comm in enumerate(comms["communities"]):
        for node in comm:
            # look up driver surname
            driver_row = drivers_copy[drivers_copy['driverId'] == node]
            surname = driver_row['surname'].values[0] if not driver_row.empty else "Unknown"
            
            community_rows.append({
                "community_id": i,
                "driver_id": node,
                "surname": surname
            })

    df_comms = pd.DataFrame(community_rows)
    df_comms.to_csv(f"{save_path}/communities.csv", index=False)


def export_degree_summary(G, drivers_df, save_path="results"):
    """Exports degree summary with driver ID, surname, in-degree, out-degree, and total degree."""
    
    degree_data = []
    
    for node in G.nodes():
        # get degrees
        in_deg = G.in_degree(node, weight='weight')
        out_deg = G.out_degree(node, weight='weight')
        total_deg = in_deg + out_deg
        
        # get driver surname
        driver_row = drivers_df[drivers_df['driverId'] == node]
        surname = driver_row['surname'].values[0] if not driver_row.empty else "Unknown"
        
        degree_data.append({
            "driver_id": node,
            "surname": surname,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "total_degree": total_deg
        })
    
    df_degrees = pd.DataFrame(degree_data)
    # sort by total degree descending
    df_degrees = df_degrees.sort_values('total_degree', ascending=False)
    df_degrees.to_csv(f"{save_path}/degree_summary.csv", index=False)
    print(f"Exported degree summary to {save_path}/degree_summary.csv")





results_2025 = analyze_network(G_2025, "2025 Season", save_path="results/results_2025", use_weight=True, log_log=True)
print_summary(results_2025, "2025 Season")
export_summary_to_csv(results_2025, "results/results_2025")
export_degree_summary(G_2025, drivers_copy, "results/results_2025")