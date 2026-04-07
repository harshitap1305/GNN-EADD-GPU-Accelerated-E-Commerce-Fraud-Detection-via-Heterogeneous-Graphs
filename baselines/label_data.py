import pandas as pd
import numpy as np
import gzip
import json
import networkx as nx
from tqdm import tqdm

def run_pipeline(review_path, meta_path):
    # --- PHASE 1: LOAD METADATA (Product Nodes) ---
    print("Step 1/6: Loading Product Metadata...")
    meta_data = []
    with gzip.open(meta_path, 'rt') as f:
        for line in tqdm(f, desc="Metadata"):
            try:
                d = json.loads(line)
                cat_list = d.get('categories', [['Unknown']])
                cat = cat_list[0][-1] if cat_list and cat_list[0] else 'Unknown'
                meta_data.append({
                    'asin': d.get('asin'),
                    'brand': str(d.get('brand', 'Unknown')),
                    'title': str(d.get('title', '')),
                    'price': d.get('price'),
                    'category': cat,
                    'also_buy_count': len(d.get('related', {}).get('also_buy', []))
                })
            except:
                continue
    df_meta = pd.DataFrame(meta_data)
    df_meta['price'] = pd.to_numeric(df_meta['price'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

    # --- PHASE 2: LOAD REVIEWS (User-Product Interactions) ---
    print("\nStep 2/6: Loading Reviews...")
    rev_data = []
    with gzip.open(review_path, 'rt') as f:
        for line in tqdm(f, desc="Reviews"):
            try:
                d = json.loads(line)
                rev_data.append({
                    'reviewerID': d.get('reviewerID'), 
                    'asin': d.get('asin'), 
                    'overall': d.get('overall'),
                    'verified': d.get('verified', False),
                    'unixReviewTime': d.get('unixReviewTime')
                })
            except:
                continue
    df_rev = pd.DataFrame(rev_data)

    # --- PHASE 3: STRUCTURAL ANALYSIS ---
    print("\nStep 3/6: Computing Graph Core Numbers...")
    G = nx.Graph()
    G.add_edges_from(zip(df_rev['reviewerID'], df_rev['asin']))
    core_map = nx.core_number(G)
    
    # Map core numbers back to dataframes
    df_meta['core_num'] = df_meta['asin'].map(core_map).fillna(0)

    # --- PHASE 4: PRODUCT ANOMALIES (Unified Goal Flagging) ---
    print("\nStep 4/6: Flagging Outlier Products...")
    
    # 1. Prepare attributes for Goal Summary
    medians = df_meta.groupby('category')['price'].transform('median')
    ratings = df_rev.groupby('asin')['overall'].mean()
    df_meta['avg_rating'] = df_meta['asin'].map(ratings).fillna(0)
    
    v_ratio = df_rev.groupby('asin')['verified'].mean()
    df_meta['verified_ratio'] = df_meta['asin'].map(v_ratio).fillna(1.0)

    # FIX: Create the mismatch column BEFORE using it in the flag logic
    df_meta['brand_mismatch'] = df_meta.apply(
        lambda x: x['brand'].lower() not in x['title'].lower() if x['brand'] != 'Unknown' else False, axis=1
    )
    
    # 2. Define Goal Flags with Tuned Thresholds for 4000-5000 range
    # Target top 0.6% structurally
    k_limit_p = df_meta['core_num'].quantile(0.994)
    
    df_meta['is_fake_product'] = (
        (df_meta['avg_rating'] >= 4.8) & 
        (df_meta['price'] < 0.15 * medians) & 
        (df_meta['brand_mismatch'] == True)
    ).astype(int)

    df_meta['is_fake_seller'] = (
        (df_meta['also_buy_count'] > 80) & 
        (df_meta['verified_ratio'] < 0.25)
    ).astype(int)

    df_meta['is_kcore_anomaly'] = (df_meta['core_num'] >= k_limit_p).astype(int)

    # UNIFIED PRODUCT FLAG
    df_meta['is_anomaly'] = df_meta[['is_fake_product', 'is_fake_seller', 'is_kcore_anomaly']].max(axis=1).astype(np.uint8)

    # --- PHASE 5: USER ANOMALIES (Unified Goal Flagging) ---
    print("\nStep 5/6: Flagging Outlier Users...")
    df_users = pd.DataFrame({'reviewerID': df_rev['reviewerID'].unique()})
    df_users['core_num'] = df_users['reviewerID'].map(core_map).fillna(0)
    
    # Goal: Synchronized unixReviewTime bursts (> 35 reviews at same second)
    # Stricter threshold to pull count toward 5000
    burst_users = df_rev.groupby(['reviewerID', 'unixReviewTime']).filter(lambda x: len(x) > 35)['reviewerID'].unique()
    
    # Goal: is_kcore_anomaly - Target top 0.6% (1 - 0.994)
    k_limit_u = df_users['core_num'].quantile(0.994)
    
    # UNIFIED USER FLAG
    df_users['is_anomaly'] = (
        (df_users['core_num'] >= k_limit_u) | 
        (df_users['reviewerID'].isin(burst_users))
    ).astype(np.uint8)

    # --- PHASE 6: EXPORT (Updated for specific filenames and formats) ---
    print("\nStep 6/6: Saving Specific Output Files...")
    
    # 1. CSV files for both datasets (All details for anomalous nodes only)
    df_meta[df_meta['is_anomaly'] == 1].to_csv('labelling_meta.csv', index=False)
    df_users[df_users['is_anomaly'] == 1].to_csv('labelling_5core.csv', index=False)
    
    # 2. TXT files containing only the IDs (ASIN/ReviewerID) of anomalous nodes
    # For meta dataset
    anom_asins = df_meta[df_meta['is_anomaly'] == 1]['asin'].astype(str)
    with open('labelling_asin_meta.txt', 'w') as f:
        f.write('\n'.join(anom_asins))
        
    # For 5-core review dataset
    anom_users = df_users[df_users['is_anomaly'] == 1]['reviewerID'].astype(str)
    with open('labelling_asin_5_core.txt', 'w') as f:
        f.write('\n'.join(anom_users))
    
    print("-" * 30)
    print(f"Anomalous Products found: {len(anom_asins)}")
    print(f"Anomalous Users found:    {len(anom_users)}")
    print("-" * 30)
    print("Files created successfully:")
    print("- labelling_meta.csv")
    print("- labelling_5core.csv")
    print("- labelling_asin_meta.txt")
    print("- labelling_asin_5_core.txt")

if __name__ == "__main__":
    run_pipeline('Electronics_5.json.gz', 'meta_Electronics.json.gz')
