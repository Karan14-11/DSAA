import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import os

def process_citation_hepph(edge_path, date_path):
    """Merges HepPh edges with publication dates."""
    print("   -> Merging Citation Dates...")
    # Load Dates
    dates = pd.read_csv(date_path, sep='\t', header=None, comment='#', names=['node_id', 'date'])
    dates['date'] = pd.to_datetime(dates['date'], errors='coerce')
    dates = dates.dropna()
    node_to_time = pd.Series(dates['date'].values, index=dates['node_id']).to_dict()
    
    # Load Edges
    edges = pd.read_csv(edge_path, sep='\t', header=None, comment='#', names=['u', 'v'])
    
    # Map Edge Time = Publication Date of Source Paper (u)
    edges['time'] = edges['u'].map(node_to_time)
    edges = edges.dropna(subset=['time'])
    return edges

def process_social_reddit(file_path):
    """Loads Reddit TSV."""
    print("   -> Loading Reddit TSV...")
    df = pd.read_csv(file_path, sep=' ', header=0, names=['u', 'v', 'time'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def generate_snapshots(name, df, output_dir, num_snapshots=50, max_nodes=22000, mode='window'):
    print(f"\n--- Processing {name} (N={max_nodes}, T={num_snapshots}, Mode={mode}) ---")
    
    # Sort & Filter Top Nodes
    df = df.sort_values('time')
    all_nodes = pd.concat([df['u'], df['v']])
    top_nodes = all_nodes.value_counts().head(max_nodes).index
    node_map = {name: i for i, name in enumerate(top_nodes)}
    
    # Keep only edges between top nodes
    df = df[df['u'].isin(top_nodes) & df['v'].isin(top_nodes)].copy()
    df['u'] = df['u'].map(node_map)
    df['v'] = df['v'].map(node_map)
    
    print(f"   -> Filtered to {len(df)} edges among top {max_nodes} nodes.")

    # Define Time Bins
    times = df['time'].values.astype(np.int64) // 10**9 # Unix timestamp
    t_start, t_end = times.min(), times.max()
    time_step = (t_end - t_start) / num_snapshots
    
    # Distribute Edges into Buckets
    snapshot_edges = [[] for _ in range(num_snapshots)]
    
    print("   -> Binning edges...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        u, v, t = int(row['u']), int(row['v']), row['time'].timestamp()
        
        # Calculate Index
        idx = int((t - t_start) / time_step)
        if idx >= num_snapshots: idx = num_snapshots - 1
        
        # PHYSICS LOGIC
        if mode == 'window':
            # Social: Edge exists ONLY in this bucket
            snapshot_edges[idx].append([u, v])
        elif mode == 'cumulative':
            # Citation: Edge exists in this bucket AND ALL FUTURE buckets
            for future_idx in range(idx, num_snapshots):
                snapshot_edges[future_idx].append([u, v])

    # Save to Disk
    save_path = os.path.join(output_dir, name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"   -> Saving to {save_path}...")
    for i in range(num_snapshots):
        edges = snapshot_edges[i]
        
        # Sparse Conversion
        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            # Make Undirected
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            # Remove duplicates
            edge_index = torch.unique(edge_index, dim=1)
            
        # Add Class Label (y=0 for Citation, y=1 for Social)
        # We also add 'num_nodes' to help PyG
        y_label = torch.tensor([0 if name == 'citation' else 1], dtype=torch.long)
        x_dummy = torch.ones((max_nodes, 1)) # Placeholder features
        
        data = Data(x=x_dummy, edge_index=edge_index, y=y_label, num_nodes=max_nodes)
        torch.save(data, f"{save_path}/snap_{i}.pt")
    
    print(f"   -> Done! Saved {num_snapshots} snapshots.")


raw_files = ['./raw/sx-stackoverflow-a2q.txt','./raw/sx-askubuntu-a2q.txt',
             './raw/sx-mathoverflow-a2q.txt','./raw/sx-superuser-a2q.txt',
             './raw/wiki-talk-temporal.txt',]
names = [ 'stackoverflow', 'askubuntu', 'mathoverflow', 'superuser','wikitalk']
nodes = [2464607,137517,21688,167981,1140149]

if __name__ == "__main__":
    # 1. Process 
    # REDDIT
    for raw_file, name in zip(raw_files[-1:], names[-1:]):
        if os.path.exists(raw_file):
            df_citation = process_social_reddit(raw_file)
            print(df_citation.head())
            generate_snapshots(name, df_citation, 'processed', 
                               num_snapshots=50, max_nodes=2464607, mode='window')
        else:
            print(f"Error: Citation file for {name} not found in /raw/")
    

