import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from networkx.algorithms import community

# ── Load data and rebuild network ──────────────────────────────────────────────
ot = pd.read_csv('data/analysis_datasets/overtakes.csv')
drivers = pd.read_csv('data/original_datasets/drivers.csv')

drivers_copy = drivers.copy()
drivers_copy['full_name'] = drivers_copy['forename'] + ' ' + drivers_copy['surname']
drivers_copy['label'] = drivers_copy['surname']


G = nx.DiGraph()

involved_drivers = pd.unique(pd.concat([ot['overtakerId'], ot['overtakenId']]))
for driver_id in involved_drivers:
    row = drivers_copy.loc[drivers_copy['driverId'] == driver_id]
    label = row['label'].values[0] if not row.empty else str(driver_id)
    G.add_node(driver_id, label=label)

pair_points = (
    ot.groupby(['overtakerId', 'overtakenId'])['point_value']
    .sum().reset_index(name='total_points')
)
pair_counts = (
    ot.groupby(['overtakerId', 'overtakenId'])['net_positions_gained']
    .sum().reset_index(name='total_count')
)
pair_summary = pair_points.merge(pair_counts, on=['overtakerId', 'overtakenId'])

processed_pairs = set()
for _, row in pair_summary.iterrows():
    a = int(row['overtakerId'])
    b = int(row['overtakenId'])
    pair = tuple(sorted([a, b]))
    if pair in processed_pairs:
        continue
    processed_pairs.add(pair)

    a_pts = pair_summary[(pair_summary['overtakerId'] == a) & (pair_summary['overtakenId'] == b)]['total_points'].sum()
    b_pts = pair_summary[(pair_summary['overtakerId'] == b) & (pair_summary['overtakenId'] == a)]['total_points'].sum()
    a_cnt = pair_summary[(pair_summary['overtakerId'] == a) & (pair_summary['overtakenId'] == b)]['total_count'].sum()
    b_cnt = pair_summary[(pair_summary['overtakerId'] == b) & (pair_summary['overtakenId'] == a)]['total_count'].sum()

    net_pts = a_pts - b_pts
    net_cnt = abs(a_cnt - b_cnt)

    if net_pts > 0:
        G.add_edge(a, b, weight=net_pts, count=net_cnt)
    elif net_pts < 0:
        G.add_edge(b, a, weight=abs(net_pts), count=net_cnt)



# ── Community detection ────────────────────────────────────────────────────────
undirected_G = G.to_undirected()
communities_list = list(community.louvain_communities(undirected_G, seed=42))
modularity = community.modularity(undirected_G, communities_list)
print(f"Communities: {len(communities_list)}, Modularity: {modularity:.3f}")

# assign community id to each node
node_community = {}
for i, comm in enumerate(communities_list):
    for node in comm:
        node_community[node] = i



# use weight to attract nodes that interact more heavily
# use 200 iterations to ensure layout settles
# set k=7.5 to keep nodes from touching or being too close
pos = nx.spring_layout(G, weight='weight', seed=42, k=7.5, iterations=200)



COMMUNITY_COLORS = ["#3E7EEEEC", "#32BEA280", "#EA797995"]  # blue, blue, teal 
BACKGROUND = "#ffffffff"
NODE_EDGE_COLOR ='#1a1a2e'
LABEL_COLOR = '#1a1a2e'
ARROW_COLOR_BASE = "#1a1a2e24"


# node size: scaled by 2025 championship points
CHAMPIONSHIP_PTS = {
    'Norris':     423, 'Verstappen': 421, 'Piastri':    410,
    'Russell':    319, 'Leclerc':    242, 'Hamilton':   156,
    'Antonelli':  150, 'Albon':       73, 'Sainz':       64,
    'Alonso':      56, 'Hulkenberg':  51, 'Hadjar':      51,
    'Bearman':     41, 'Lawson':      38, 'Ocon':        38,
    'Stroll':      33, 'Tsunoda':     33, 'Gasly':       22,
    'Bortoleto':   19, 'Colapinto':    0,
}
min_pts = min(CHAMPIONSHIP_PTS.values())
max_pts = max(CHAMPIONSHIP_PTS.values())
node_sizes = {
    n: 2500 + 4000 * (CHAMPIONSHIP_PTS.get(G.nodes[n]['label'], 0) - min_pts) / (max_pts - min_pts + 1e-9)
    for n in G.nodes()
}




# edge width: scaled net point value (weight)
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
max_weight = max(edge_weights)
edge_widths = [0.3 + 3.5 * (w / max_weight) for w in edge_weights]

# edge opacity: scaled raw count frequency
edge_counts = [G[u][v]['count'] for u, v in G.edges()]
max_count = max(edge_counts)
edge_alphas = [0.08 + 0.72 * (c / max_count) for c in edge_counts]

# node colors by community
node_colors = [COMMUNITY_COLORS[node_community[n]] for n in G.nodes()]
node_size_list = [node_sizes[n] for n in G.nodes()]




fig, ax = plt.subplots(figsize=(16, 12))
fig.patch.set_facecolor(BACKGROUND)
ax.set_facecolor(BACKGROUND)

# draw edges individually to apply per-edge alpha
for (u, v), width, alpha in zip(G.edges(), edge_widths, edge_alphas):
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)], ax=ax,
        width=width,
        alpha=alpha,
        edge_color=ARROW_COLOR_BASE,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15,
        connectionstyle='arc3,rad=0.08',
        min_source_margin=18,
        min_target_margin=27,
    )

# draw nodes
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_size=node_size_list,
    node_color=node_colors,
    edgecolors=NODE_EDGE_COLOR,
    linewidths=0.8,
    alpha=0.92,
)

# draw labels with surname only
labels = nx.get_node_attributes(G, 'label')
label_pos = {}
for node, (x, y) in pos.items():
    pts = CHAMPIONSHIP_PTS.get(G.nodes[node]['label'], 0)
    size_factor = 0.06 + 0.02 * (pts - min_pts) / (max_pts - min_pts + 1e-9)
    label_pos[node] = (x, y + size_factor)
nx.draw_networkx_labels(
    G, label_pos, labels=labels, ax=ax,
    font_size=16,
    font_color='#1a1a2e',
    font_weight='bold',
    font_family='monospace',
)




community_sizes = [len(c) for c in communities_list]
community_patches = [
    mpatches.Patch(color=COMMUNITY_COLORS[i], label=f'Community {i+1}  ({community_sizes[i]} drivers)')
    for i in range(len(communities_list))
]


leg1 = ax.legend(handles=community_patches, loc='upper right',
                 framealpha=0.8, facecolor='#eeeeee',
                 edgecolor='#aaaaaa', labelcolor='black',
                 fontsize=18, title='Communities', title_fontsize=21)

leg1.get_title().set_color('black')
ax.add_artist(leg1)

ax.text(0.03, 0.97,
        'Node size = championship points\nEdge width = points gained from passing\nEdge opacity = frequency of passing',
        transform=ax.transAxes, ha='left', va='top',
        fontsize=16, color='black', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#eeeeee',
                  edgecolor='#aaaaaa', alpha=0.8))






stats_text = (
    f'Nodes: {G.number_of_nodes()}   |   Edges: {G.number_of_edges()}   |   '
    f'Communities: {len(communities_list)}   |   Modularity: {modularity:.3f}'
)
ax.text(0.5, -0.02, stats_text, transform=ax.transAxes,
        ha='center', va='top', fontsize=18, color="#030101ff", fontfamily='monospace')

ax.axis('off')
plt.tight_layout()
plt.savefig('results/network_visualization.png', dpi=300, bbox_inches='tight',
            facecolor=BACKGROUND)
plt.show()
print("Saved to results/network_visualization.png")