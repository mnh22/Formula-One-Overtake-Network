import pandas as pd
import networkx as nx
import numpy as np

ot = pd.read_csv('data/analysis_datasets/overtakes.csv')
drivers = pd.read_csv('data/original_datasets/drivers.csv')

# prep driver information to add for easy review later on
drivers_copy = drivers.copy()
drivers_copy['full_name'] = drivers_copy['forename'] + ' ' + drivers_copy['surname']

# use driver surname each node label
drivers_copy['label'] = drivers_copy['surname']

# construct network (general definition to be able to easily make networks for diffent time period inputs)

# nodes: each of the drivers
# edges: net overtake interactions determines direction to show relationshop between overtaker and overtaken for all drivers


def build_overtake_network(overtakes_df, drivers_df):
    
    G = nx.DiGraph()
    
    # only include unique drivers who were involved in overtakes for specified time range
    
    involved_drivers = pd.unique(
        pd.concat([overtakes_df['overtakerId'], overtakes_df['overtakenId']])
    )

    # add nodes for each of the relevant drivers
    for driver_id in involved_drivers:
        row = drivers_df.loc[drivers_df['driverId'] == driver_id]
        if not row.empty:
            label = row['label'].values[0]
            
        G.add_node(driver_id, label=label)

    
    # compute net overtakes between each pair of drivers
    # count overtakes in each direction between pairs
    # aggregate point values in each direction for every ordered pair
    pair_points = (
        overtakes_df
        .groupby(['overtakerId', 'overtakenId'])['point_value']
        .sum()
        .reset_index(name='total_points')
    )


    
    # calculate net overtakes for each unique pair
    processed_pairs = set()

    for _, row in pair_points.iterrows():
        overtaker = int(row['overtakerId'])
        overtaken = int(row['overtakenId'])
        
        # skip if processed this pair already
        pair = tuple(sorted([overtaker, overtaken]))
        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)
        
        # aggregate point value for overtaker overtakes of overtaken
        a_to_b = pair_points[
            (pair_points['overtakerId'] == overtaker) & 
            (pair_points['overtakenId'] == overtaken)
        ]['total_points'].sum()
        
        # aggregate point value for overtaken overtakes of overtaker
        b_to_a = pair_points[
            (pair_points['overtakerId'] == overtaken) & 
            (pair_points['overtakenId'] == overtaker)
        ]['total_points'].sum()
        
        # calculate net and add edge from overall overtaker to overall overtaken
        net = a_to_b - b_to_a
        
        if net > 0:
            G.add_edge(overtaker, overtaken, weight=net)
        elif net < 0:
            G.add_edge(overtaken, overtaker, weight=abs(net))
        # if net = 0, no edge is added or necessary
    
    print(f"Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")
    return G


# build 2025 network
G_2025 = build_overtake_network(ot, drivers_copy)