import gzip
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import random
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
from sklearn.preprocessing import MinMaxScaler


def parse(path):
    with gzip.open(path, 'rb') as g:
        for l in g:
            yield json.loads(l)


def clean_price_string(price_str):
    if not price_str:
        return 0.0
    if isinstance(price_str, (int, float)):
        return float(price_str)
    found = re.findall(r"[-+]?\d*\.\d+|\d+", str(price_str))
    return float(found[0]) if found else 0.0


def load_hetero_data(meta_path, review_path):
    print("Step 1: Processing Metadata (Products & Sellers)...")
    products = []
    brand_map_raw = {}

    for d in parse(meta_path):
        asin = d.get('asin')
        if not asin:
            continue
        brand = str(d.get('brand', 'Unknown') or 'Unknown')
        price = clean_price_string(d.get('price'))

        try:
            cat_top = str(d.get('categories', [['None']])[0][0])
        except (IndexError, TypeError):
            cat_top = 'None'

        products.append({'asin': asin, 'price': price, 'cat_id': hash(cat_top) % 1000})
        brand_map_raw[asin] = brand

    df_prod = pd.DataFrame(products).drop_duplicates('asin').reset_index(drop=True)
    asin_to_idx = {asin: i for i, asin in enumerate(df_prod['asin'])}
    unique_brands = list(set(brand_map_raw.values()))
    brand_to_idx = {brand: i for i, brand in enumerate(unique_brands)}

    print("Step 2: Processing 5-Core (Buyers & Reviews)...")
    reviews = []
    buyer_set = set()

    for d in parse(review_path):
        uid = d.get('reviewerID')
        asin = d.get('asin')
        if uid and asin and asin in asin_to_idx:
            buyer_set.add(uid)
            reviews.append((uid, asin))

    buyer_list = list(buyer_set)
    buyer_to_idx = {uid: i for i, uid in enumerate(buyer_list)}

    print(f"  Products: {len(df_prod)} | Buyers: {len(buyer_list)} "
          f"| Sellers: {len(unique_brands)} | Reviews: {len(reviews)}")

    data = HeteroData()
    scaler = MinMaxScaler()
    data['product'].x = torch.tensor(
        scaler.fit_transform(df_prod[['price', 'cat_id']].values.astype(float)),
        dtype=torch.float)
    data['buyer'].x = torch.ones((len(buyer_list), 1))
    data['seller'].x = torch.ones((len(unique_brands), 1))

    buyer_src = [buyer_to_idx[r[0]] for r in reviews]
    prod_dst_b = [asin_to_idx[r[1]] for r in reviews]
    data['buyer', 'reviews', 'product'].edge_index = torch.tensor(
        [buyer_src, prod_dst_b], dtype=torch.long)

    seller_src = [brand_to_idx[brand_map_raw.get(a, 'Unknown')] for a in df_prod['asin']]
    prod_dst_s = list(range(len(df_prod)))
    data['seller', 'sells', 'product'].edge_index = torch.tensor(
        [seller_src, prod_dst_s], dtype=torch.long)

    data['product', 'rev_reviews', 'buyer'].edge_index = torch.tensor(
        [prod_dst_b, buyer_src], dtype=torch.long)
    data['product', 'rev_sells', 'seller'].edge_index = torch.tensor(
        [prod_dst_s, seller_src], dtype=torch.long)

    idx_to_asin = {i: asin for asin, i in asin_to_idx.items()}
    idx_to_buyer = {i: uid for uid, i in buyer_to_idx.items()}

    return data, asin_to_idx, buyer_to_idx, idx_to_asin, idx_to_buyer


class HeteroDOMINANT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.encoder = HeteroConv({
            ('buyer', 'reviews', 'product'): SAGEConv((-1, -1), hidden_channels),
            ('seller', 'sells', 'product'): SAGEConv((-1, -1), hidden_channels),
            ('product', 'rev_reviews', 'buyer'): SAGEConv((-1, -1), hidden_channels),
            ('product', 'rev_sells', 'seller'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        self.bn_product = torch.nn.BatchNorm1d(hidden_channels)
        self.bn_buyer = torch.nn.BatchNorm1d(hidden_channels)
        self.bn_seller = torch.nn.BatchNorm1d(hidden_channels)
        self.bn_map = {'product': self.bn_product,
                       'buyer': self.bn_buyer,
                       'seller': self.bn_seller}

        self.proj = torch.nn.Linear(hidden_channels, out_channels)
        self.attr_decoder = torch.nn.Linear(out_channels, 2)

    def forward(self, x_dict, edge_index_dict):
        h_dict = self.encoder(x_dict, edge_index_dict)
        h_dict = {k: self.bn_map[k](F.leaky_relu(v))
                  for k, v in h_dict.items() if k in self.bn_map}
        z_dict = {k: self.proj(v) for k, v in h_dict.items()}
        x_hat = torch.sigmoid(self.attr_decoder(z_dict['product']))
        return z_dict, x_hat


def run_dominant():
    # --- FIXED SEEDING FOR REPRODUCIBILITY ---
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing DOMINANT on: {device} (Seed: {seed})")

    (data, asin_map, buyer_map,
     idx_to_asin, idx_to_buyer) = load_hetero_data(
        'meta_Electronics.json.gz', 'Electronics_5.json.gz')
    data = data.to(device)

    # --- PARAMETER CHANGES ---
    model = HeteroDOMINANT(64, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5) # Slower LR + Weight Decay

    print("Training DOMINANT Baseline...")
    alpha = 0.8 # Higher focus on attribute stability
    ei = data['buyer', 'reviews', 'product'].edge_index

    model.train()
    # Increased Epochs for convergence
    for epoch in range(251): 
        optimizer.zero_grad()
        z_dict, x_hat = model(data.x_dict, data.edge_index_dict)

        loss_a = F.mse_loss(x_hat, data['product'].x)

        # Increased sample size for structural stability
        n_sample = min(2048, ei.size(1)) 
        perm = torch.randperm(ei.size(1), device=device)[:n_sample]
        pos_b, pos_p = ei[0, perm], ei[1, perm]
        s_pos = (z_dict['buyer'][pos_b] * z_dict['product'][pos_p]).sum(dim=1)
        loss_pos = F.binary_cross_entropy_with_logits(s_pos, torch.ones_like(s_pos))

        neg_b = torch.randint(0, data['buyer'].num_nodes, (n_sample,), device=device)
        neg_p = torch.randint(0, data['product'].num_nodes, (n_sample,), device=device)
        s_neg = (z_dict['buyer'][neg_b] * z_dict['product'][neg_p]).sum(dim=1)
        loss_neg = F.binary_cross_entropy_with_logits(s_neg, torch.zeros_like(s_neg))

        loss_s = (loss_pos + loss_neg) / 2.0
        loss = alpha * loss_a + (1 - alpha) * loss_s
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f} "
                  f"(attr={loss_a.item():.6f}, struct={loss_s.item():.6f})")

    model.eval()
    with torch.no_grad():
        z_dict, x_hat = model(data.x_dict, data.edge_index_dict)
        prod_scores = torch.norm(x_hat - data['product'].x, dim=1).cpu().numpy()

        buyer_ei_cpu = data['buyer', 'reviews', 'product'].edge_index.cpu().numpy()
        buyer_scores = np.zeros(data['buyer'].num_nodes)
        buyer_count = np.zeros(data['buyer'].num_nodes)
        for b_idx, p_idx in zip(buyer_ei_cpu[0], buyer_ei_cpu[1]):
            buyer_scores[b_idx] += prod_scores[p_idx]
            buyer_count[b_idx] += 1
        buyer_count = np.maximum(buyer_count, 1)
        buyer_scores = buyer_scores / buyer_count

    thresh_prod = np.mean(prod_scores) + 2.5 * np.std(prod_scores)
    thresh_buyer = np.mean(buyer_scores) + 3 * np.std(buyer_scores)
    anom_prod_idx = np.where(prod_scores > thresh_prod)[0]
    anom_buyer_idx = np.where(buyer_scores > thresh_buyer)[0]

    # Save outputs (keeping your original logic)
    meta_rows = [{'asin': idx_to_asin[int(i)], 'anomaly_score': round(float(prod_scores[i]), 6)} for i in anom_prod_idx]
    df_meta = pd.DataFrame(meta_rows).sort_values('anomaly_score', ascending=False).reset_index(drop=True)
    df_meta.to_csv('dominant_anomalies_metadata.csv', index=False)
    with open('dominant_anomalous_asins_metadata.txt', 'w') as f:
        f.write('\n'.join(df_meta['asin'].tolist()))

    review_rows = [{'reviewerID': idx_to_buyer[int(i)], 'anomaly_score': round(float(buyer_scores[i]), 6)} for i in anom_buyer_idx]
    df_buyers = pd.DataFrame(review_rows).sort_values('anomaly_score', ascending=False).reset_index(drop=True)
    df_buyers.to_csv('dominant_anomalies_5core_buyers.csv', index=False)

    anom_buyer_set = set(anom_buyer_idx.tolist())
    anom_review_asins = set()
    for b_idx, p_idx in zip(buyer_ei_cpu[0], buyer_ei_cpu[1]):
        if b_idx in anom_buyer_set:
            anom_review_asins.add(idx_to_asin[int(p_idx)])

    with open('dominant_anomalous_asins_5core.txt', 'w') as f:
        f.write('\n'.join(sorted(anom_review_asins)))

    print("\n" + "=" * 58)
    print("DOMINANT HETERO RESULTS")
    print("=" * 58)
    print(f"Total Products : {data['product'].num_nodes:>10,}")
    print(f"Total Buyers   : {data['buyer'].num_nodes:>10,}")
    print()
    print(f"[Metadata]  threshold  : {thresh_prod:.6f}")
    print(f"[Metadata]  anomalies  : {len(anom_prod_idx):,}  "
          f"({100*len(anom_prod_idx)/data['product'].num_nodes:.3f}%)")
    print()
    print(f"[5-Core]    threshold  : {thresh_buyer:.6f}")
    print(f"[5-Core]    anom buyers: {len(anom_buyer_idx):,}  "
          f"({100*len(anom_buyer_idx)/data['buyer'].num_nodes:.3f}%)")
    print(f"[5-Core]    anom ASINs : {len(anom_review_asins):,}")
    print("=" * 58)


if __name__ == "__main__":
    run_dominant()
