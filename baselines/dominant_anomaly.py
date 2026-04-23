import gzip, json, torch, re, random, time
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# --- HELPER FUNCTIONS ---
def parse(path):
    with gzip.open(path, 'rb') as g:
        for l in g:
            yield json.loads(l)

def clean_price_string(price_str):
    if not price_str: return 0.0
    if isinstance(price_str, (int, float)): return float(price_str)
    found = re.findall(r"[-+]?\d*\.\d+|\d+", str(price_str))
    return float(found[0]) if found else 0.0

# --- MODEL DEFINITION ---
class HeteroDOMINANT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # Dual-stage logic: Shared encoder for heterogeneous nodes
        self.encoder = HeteroConv({
            ('buyer', 'reviews', 'product'): SAGEConv((-1, -1), hidden_channels),
            ('seller', 'sells', 'product'): SAGEConv((-1, -1), hidden_channels),
            ('product', 'rev_reviews', 'buyer'): SAGEConv((-1, -1), hidden_channels),
            ('product', 'rev_sells', 'seller'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')
        self.bn_map = torch.nn.ModuleDict({
            'product': torch.nn.BatchNorm1d(hidden_channels),
            'buyer': torch.nn.BatchNorm1d(hidden_channels),
            'seller': torch.nn.BatchNorm1d(hidden_channels)
        })
        self.proj = torch.nn.Linear(hidden_channels, out_channels)
        self.attr_decoder = torch.nn.Linear(out_channels, 2)

    def forward(self, x_dict, edge_index_dict):
        h_dict = self.encoder(x_dict, edge_index_dict)
        h_dict = {k: self.bn_map[k](F.leaky_relu(v)) for k, v in h_dict.items() if k in self.bn_map}
        z_dict = {k: self.proj(v) for k, v in h_dict.items()}
        # Attribute reconstruction for Stage 1
        x_hat = torch.sigmoid(self.attr_decoder(z_dict['product']))
        return z_dict, x_hat

# --- MAIN EXECUTION ---
def run_dominant():
    start_time = time.time()
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1: Meta Parsing (Products & Sellers)
    products, brand_map = [], {}
    for d in tqdm(parse('meta_Electronics.json.gz'), desc="Parsing Meta"):
        asin = d.get('asin')
        if not asin: continue
        
        # Capture 'brand' for Seller nodes
        brand = str(d.get('brand', 'Unknown'))
        
        products.append({
            'asin': asin, 
            'price': clean_price_string(d.get('price')),
            'brand': brand,
            'cat_id': hash(str(d.get('categories', [['None']])[0][0])) % 1000
        })
        brand_map[asin] = brand

    df_p = pd.DataFrame(products).drop_duplicates('asin').reset_index(drop=True)
    asin_to_idx = {asin: i for i, asin in enumerate(df_p['asin'])}
    brand_to_idx = {b: i for i, b in enumerate(df_p['brand'].unique())}

    # Step 2: 5-Core Parsing (Buyers)
    reviews, buyer_set = [], set()
    for d in tqdm(parse('Electronics_5.json.gz'), desc="Parsing 5-Core"):
        uid, asin = d.get('reviewerID'), d.get('asin')
        if uid and asin in asin_to_idx:
            buyer_set.add(uid); reviews.append((uid, asin))

    buyer_list = list(buyer_set)
    buyer_to_idx = {uid: i for i, uid in enumerate(buyer_list)}

    # HeteroData Construction
    data = HeteroData()
    scaler = MinMaxScaler() # Normalization
    data['product'].x = torch.tensor(scaler.fit_transform(df_p[['price', 'cat_id']].values), dtype=torch.float)
    data['buyer'].x, data['seller'].x = torch.ones((len(buyer_list), 1)), torch.ones((len(brand_to_idx), 1))
    
    # Define edges
    b_p = torch.tensor([[buyer_to_idx[r[0]], asin_to_idx[r[1]]] for r in reviews], dtype=torch.long).t()
    data['buyer', 'reviews', 'product'].edge_index = b_p
    data['product', 'rev_reviews', 'buyer'].edge_index = b_p[[1, 0]]
    
    s_p = torch.tensor([[brand_to_idx[brand_map[a]], asin_to_idx[a]] for a in df_p['asin']], dtype=torch.long).t()
    data['seller', 'sells', 'product'].edge_index = s_p
    data['product', 'rev_sells', 'seller'].edge_index = s_p[[1, 0]]

    # Stage 1: Unsupervised Learning
    model = HeteroDOMINANT(64, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    data = data.to(device)

    print("\nTraining Unsupervised Encoder-Decoder...")
    for epoch in range(251):
        optimizer.zero_grad()
        z_dict, x_hat = model(data.x_dict, data.edge_index_dict)
        loss = F.mse_loss(x_hat, data['product'].x)
        loss.backward(); optimizer.step()

    # Stage 2: Inference & Scoring 
    model.eval()
    with torch.no_grad():
        z_dict, x_hat = model(data.x_dict, data.edge_index_dict)
        prod_scores = torch.norm(x_hat - data['product'].x, dim=1).cpu().numpy()
        
        # Aggregate scores for Buyers
        b_ei = data['buyer', 'reviews', 'product'].edge_index.cpu().numpy()
        buyer_scores = np.zeros(data['buyer'].num_nodes)
        b_counts = np.zeros(data['buyer'].num_nodes)
        for b_idx, p_idx in zip(b_ei[0], b_ei[1]):
            buyer_scores[b_idx] += prod_scores[p_idx]
            b_counts[b_idx] += 1
        buyer_scores = buyer_scores / np.maximum(b_counts, 1)

    # Thresholds
   
    thresh_p = np.mean(prod_scores) + 2.2 * np.std(prod_scores)
  
    thresh_b = np.mean(buyer_scores) + 4.0 * np.std(buyer_scores)
    
    anom_prod = np.where(prod_scores > thresh_p)[0]
    anom_buyer = np.where(buyer_scores > thresh_b)[0]

    

    # Initialize global scores with zeros (or a very low value)
    total_nodes = data['buyer'].num_nodes + data['product'].num_nodes + data['seller'].num_nodes
    global_scores = np.zeros(total_nodes)

    # 1. Fill User scores (Indices 0 to N_u-1)
    # Note: Ensure the index order matches your mapping.json/node_counts.json
    global_scores[:data['buyer'].num_nodes] = buyer_scores

    # 2. Fill Product scores (Indices N_u to N_u + N_p - 1)
    start_p = data['buyer'].num_nodes
    end_p = start_p + data['product'].num_nodes
    global_scores[start_p:end_p] = prod_scores

    # 3. Seller scores (Indices end_p to end_p + N_s - 1) 
    # Usually left as 0 if not evaluated, but ensures array size is correct
    
    # Save the file for use in the Performance Evaluation Script
    np.save('dominant_anomalies.npy', global_scores)
    print("\n" + "="*50)
    print("TARGETED BASELINE RESULTS")
    print("="*50)
    print(f"Total Processing Time:      {time.time() - start_time:.2f}s")
    print(f"Product Anomalies (Meta):   {len(anom_prod)}")
    print(f"Review Anomalies (5-Core):  {len(anom_buyer)}")
    print("="*50)
    print(f"Successfully saved global scores to 'dominant_anomalies.npy' (Shape: {global_scores.shape})")
    
if __name__ == "__main__":
    run_dominant()
