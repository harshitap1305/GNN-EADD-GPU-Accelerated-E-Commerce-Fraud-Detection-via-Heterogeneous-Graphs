import gzip
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- 1. DATA STREAMING ---
def parse(path):
    with gzip.open(path, 'rb') as g:
        for l in g:
            yield json.loads(l)

def load_full_data(meta_path, core_path):
    print("Step 1/4: Building Product Mapping from Metadata...")
    asin_map = {}
    meta_info = {} 
    
    # Process Product Metadata 
    for d in tqdm(parse(meta_path), desc="Parsing Meta"):
        asin = d.get('asin')
        if asin:
            if asin not in asin_map:
                asin_map[asin] = len(asin_map)
            
            # Feature extraction (Price and Categories)
            price = d.get('price', 0)
            cats = d.get('categories', [])
            cat_id = hash(str(cats[0][0])) % 1000 if (cats and cats[0]) else 0
            meta_info[asin] = [price, cat_id]

    inv_asin_map = {v: k for k, v in asin_map.items()}
    num_nodes = len(asin_map)

    print(f"Step 2/4: Building Structural Edges (also_buy/viewed)...")
    edge_list = []
    for d in tqdm(parse(meta_path), desc="Building Edges"):
        src_asin = d.get('asin')
        if src_asin in asin_map:
            src_idx = asin_map[src_asin]
            # Modeling relationships as edges
            related = d.get('also_buy', []) + d.get('also_viewed', [])
            for rel_asin in related:
                if rel_asin in asin_map:
                    edge_list.append([src_idx, asin_map[rel_asin]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    print("Step 3/4: Scaling Product Features...")
    feature_matrix = np.zeros((num_nodes, 2))
    for asin, idx in asin_map.items():
        if asin in meta_info:
            p_str = str(meta_info[asin][0]).replace('$', '').replace(',', '')
            try:
                price = float(p_str)
            except:
                price = 0.0
            feature_matrix[idx] = [price, meta_info[asin][1]]
    
    x = torch.tensor(StandardScaler().fit_transform(feature_matrix), dtype=torch.float)

    print("Step 4/4: Mapping Interaction Data (5-Core)...")
    core_set = set()
    for d in tqdm(parse(core_path), desc="Parsing 5-Core"):
        core_set.add(d.get('asin'))

    return Data(x=x, edge_index=edge_index), inv_asin_map, core_set, feature_matrix

# --- 2. GRAPHSAGE ARCHITECTURE ---
class GraphSAGEBaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # SAGEConv aggregates neighbor features
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        # Linear decoder for unsupervised reconstruction
        self.decoder = torch.nn.Linear(out_channels, in_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        return self.decoder(h)

# --- 3. TRAINING & TARGETED EXTRACTION ---
def run_baseline():
    start_time = time.time()
    device = torch.device('cpu') 
    
    data, inv_map, core_set, raw_features = load_full_data('meta_Electronics.json.gz', 'Electronics_5.json.gz')
    
    model = GraphSAGEBaseline(data.num_node_features, 64, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\nTraining Unsupervised Encoder-Decoder...")
    model.train()
    for epoch in range(31):
        optimizer.zero_grad()
        recon = model(data.x, data.edge_index)
        # Minimizing reconstruction loss
        loss = F.mse_loss(recon, data.x)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        recon = model(data.x, data.edge_index)
        # L2 norm for anomaly scoring
        scores = torch.norm(recon - data.x, dim=1).numpy()

    # --- TARGETING 4000-5000 ANOMALIES ---
    # Calculating the specific percentile to hit the target range
    target_count = 4500 
    percentile_rank = 100 * (1 - target_count / len(scores))
    threshold = np.percentile(scores, percentile_rank)
    
    anomaly_indices = np.where(scores >= threshold)[0]
    
    meta_anomalies = []
    core_anomalies = []

    for idx in anomaly_indices:
        asin = inv_map[idx]
        item = {'asin': asin, 'score': float(scores[idx]), 
                'price': float(raw_features[idx][0]), 'cat_id': int(raw_features[idx][1])}
        
        meta_anomalies.append(item)
        if asin in core_set:
            core_anomalies.append(item)

    # --- EXPORTING 4 FILES ---
    # Metadata Related Outputs (Products)
    pd.DataFrame(meta_anomalies).to_csv('meta_anomalies.csv', index=False)
    with open('meta_asins.txt', 'w') as f:
        for item in meta_anomalies: f.write(f"{item['asin']}\n")
    
    # 5-Core Related Outputs (Interaction/Review subsets)
    pd.DataFrame(core_anomalies).to_csv('five_core_anomalies.csv', index=False)
    with open('five_core_asins.txt', 'w') as f:
        for item in core_anomalies: f.write(f"{item['asin']}\n")

    print("\n" + "="*50)
    print("TARGETED BASELINE RESULTS")
    print("="*50)
    print(f"Total Processing Time:      {time.time() - start_time:.2f}s")
    print(f"Product Anomalies (Meta):   {len(meta_anomalies)}")
    print(f"Review Anomalies (5-Core):  {len(core_anomalies)}")
    print("="*50)

if __name__ == "__main__":
    run_baseline()
