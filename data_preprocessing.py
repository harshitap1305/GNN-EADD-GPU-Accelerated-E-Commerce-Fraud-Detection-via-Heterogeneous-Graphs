"""
GNN-EADD Stage 1 Preprocessing Pipeline (Ultimate Edition - 16GB RAM Safe)
==========================================================================
Combines SSD-streaming memory efficiency with strict GCN mathematical correctness.
Eliminates Python string memory bloat via chunked on-the-fly NLP encoding.

Outputs:
  - CSR Topology Binaries: epu_*, eps_*, euu_* (Includes Transposes & Self-Loops)
  - Memory-mapped features: V_p, V_u, V_s, and X_combined.memmap
  - ID Mappings & Counts: node_counts.json, node_id_mappings.json
"""

import gzip
import json
import time
import random
import gc
import numpy as np
import nltk
from collections import defaultdict
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

REVIEWS_FILE  = "AMAZON_FASHION_5.json"
METADATA_FILE = "meta_AMAZON_FASHION.json"

TEXT_DIM       = 96
PROD_CAT_DIM   = 24
PROD_STAT_DIM  = 8
USER_CAT_DIM   = 112
USER_STAT_DIM  = 16
SELLER_CAT_DIM = 112
SELLER_STAT_DIM= 16
FEAT_DIM       = 128

ENCODE_BATCH = 256
CHUNK_SIZE   = 10000 # How many texts to hold in RAM before encoding
PCA_BATCH    = 2048
MAX_USERS_PER_PROD_EUU = 25  # Prevents edge explosion for popular items
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

INVALID_BRANDS = {
    'generic', 'unknown', 'unbranded', 'n/a', 'none', 'na',
    'no brand', 'no name', 'amazon', 'amazon.com', '', ' ',
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def parse_file(path):
    """Line-by-line generator handling both plain .json and .gz files."""
    is_gz = path.endswith('.gz')
    open_func = gzip.open if is_gz else open
    mode = 'rt' if is_gz else 'r'

    with open_func(path, mode, encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError: continue

def _clean_brand(raw) -> str | None:
    if not raw: return None
    s = str(raw).strip().lower()
    if not s or s.startswith('<') or s in INVALID_BRANDS: return None
    return s

def _parse_price(raw) -> float:
    if raw is None: return 0.0
    try: return float(str(raw).replace('$', '').replace(',', '').strip())
    except (ValueError, AttributeError): return 0.0

def safe_pca_transform(data, target_dim: int) -> np.ndarray:
    """Safely applies PCA and pads with zeros to prevent crashes on smaller subsets."""
    n_samples, n_features = data.shape
    if n_samples == 0 or n_features == 0:
        return np.zeros((n_samples, target_dim), dtype=np.float32)

    n_comp = min(target_dim, n_samples, n_features)
    batch_size = max(n_comp, min(PCA_BATCH, n_samples))

    pca = IncrementalPCA(n_components=n_comp, batch_size=batch_size)
    transformed = pca.fit_transform(data)

    if n_comp < target_dim:
        padding = np.zeros((n_samples, target_dim - n_comp), dtype=np.float32)
        transformed = np.hstack((transformed, padding))
    return transformed

def _build_csr_with_self_loops(src_ids, dst_ids, n_src, src_offset, prefix):
    """Builds CSR arrays with self loops and saves them directly as CUDA binaries."""
    adj = defaultdict(set)
    for s, d in zip(src_ids, dst_ids):
        adj[s - src_offset].add(d)

    # GCN Requirement: Self-loops via global ID
    for i in range(n_src):
        adj[i].add(i + src_offset)

    row_ptr = np.zeros(n_src + 1, dtype=np.int32)
    cols_list = []
    for i in range(n_src):
        nbrs = sorted(adj[i])
        cols_list.extend(nbrs)
        row_ptr[i + 1] = len(cols_list)

    col_idx = np.array(cols_list, dtype=np.int32)

    row_ptr.tofile(f"{prefix}_row_ptr.bin")
    col_idx.tofile(f"{prefix}_col_idx.bin")
    _log(f"  -> Saved {prefix}_row_ptr.bin / {prefix}_col_idx.bin")
    return len(cols_list)

def save_memmap(matrix: np.ndarray, filename: str) -> None:
    _log(f"  Saving {filename} shape={matrix.shape} ({matrix.nbytes / 1e6:.1f} MB)…")
    if matrix.shape[0] == 0 or matrix.size == 0:
        open(filename, 'wb').close()
        return
    fp = np.memmap(filename, dtype='float16', mode='w+', shape=matrix.shape)
    fp[:] = matrix[:]
    fp.flush()
    del fp

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    _log("=" * 62)
    _log("GNN-EADD Stage 1 Preprocessing Pipeline (16GB RAM Ultimate Edition)")
    _log("=" * 62)

    # ─── PASS 1: Build Unified ID Space ──────────────────────────────────────
    _log("Pass 1 – Scanning Metadata for valid items and sellers...")
    valid_items = set()
    item_to_seller = {}

    count = 0
    for meta in parse_file(METADATA_FILE):
        count += 1
        if count % 500000 == 0: _log(f"  ... scanned {count:,} metadata lines ...")

        asin = meta.get('asin')
        brand = _clean_brand(meta.get('brand'))
        if asin:
            valid_items.add(asin)
            if brand: item_to_seller[asin] = brand

    _log("Pass 1 – Scanning Reviews for valid users...")
    valid_users = set()
    item_buyers = defaultdict(list)

    count = 0
    for rev in parse_file(REVIEWS_FILE):
        count += 1
        if count % 1000000 == 0: _log(f"  ... scanned {count:,} review lines ...")

        uid = rev.get('reviewerID')
        asin = rev.get('asin')
        if uid and asin in valid_items:
            valid_users.add(uid)
            item_buyers[asin].append(uid)

    all_users = sorted(valid_users)
    all_items = sorted(valid_items)
    all_sellers = sorted(set(item_to_seller.values()))

    N_u, N_p, N_s = len(all_users), len(all_items), len(all_sellers)

    user_to_id   = {uid: i for i, uid in enumerate(all_users)}
    product_to_id = {asin: i + N_u for i, asin in enumerate(all_items)}
    seller_to_id  = {brand: i + N_u + N_p for i, brand in enumerate(all_sellers)}

    _log(f"  Users: {N_u:,} | Products: {N_p:,} | Sellers: {N_s:,}")

    # ─── PASS 2: Extract NLP Signatures ──────────────────────────────────────
    _log("Pass 2 – Extracting Review Fraud Signatures (NLP Streaming)...")
    sia = SentimentIntensityAnalyzer()

    item_stats = defaultdict(lambda: {'c':0, 'r_sum':0, 'r_sq':0, 't_min':1e18, 't_max':0, 'v_sum':0, 'wc_sum':0, 'lex_sum':0, 'mis_sum':0})
    user_stats = defaultdict(lambda: {'c':0, 'r_sum':0, 'r_sq':0, 't_min':1e18, 't_max':0, 'v_sum':0, 'wc_sum':0, 'lex_sum':0, 'mis_sum':0, 'pos':0, 'neg':0})

    count = 0
    for rev in parse_file(REVIEWS_FILE):
        count += 1
        if count % 1000000 == 0: _log(f"  ... processed {count:,} reviews for NLP ...")

        u, i = rev.get('reviewerID'), rev.get('asin')
        if u not in user_to_id or i not in product_to_id: continue

        rating = float(rev.get('overall', 3.0))
        ts = float(rev.get('unixReviewTime', 0))
        text = rev.get('reviewText', '') or ''

        vote_raw = rev.get('vote', 0)
        try: vote = float(str(vote_raw).replace(',', ''))
        except: vote = 0.0

        words = text.split()
        wc = len(words)
        lex_div = len(set(words)) / wc if wc > 0 else 0.0
        sentiment = sia.polarity_scores(text)['compound'] if text else 0.0
        mismatch = abs(sentiment - ((rating - 3.0) / 2.0))

        s, su = item_stats[i], user_stats[u]
        s['c'] += 1; su['c'] += 1
        s['r_sum'] += rating; su['r_sum'] += rating
        s['r_sq'] += rating**2; su['r_sq'] += rating**2
        s['t_min'] = min(s['t_min'], ts); su['t_min'] = min(su['t_min'], ts)
        s['t_max'] = max(s['t_max'], ts); su['t_max'] = max(su['t_max'], ts)
        s['v_sum'] += vote; su['v_sum'] += vote
        s['wc_sum'] += wc; su['wc_sum'] += wc
        s['lex_sum'] += lex_div; su['lex_sum'] += lex_div
        s['mis_sum'] += mismatch; su['mis_sum'] += mismatch
        su['pos'] += 1 if rating >= 4 else 0
        su['neg'] += 1 if rating <= 2 else 0

    # ─── PASS 3: Generate Feature Matrices & Chunked Encoding ────────────────
    _log("Pass 3 – Caching metadata and Chunk-Encoding text (RAM Safe)...")

    # 768MB numpy array for raw embeddings, perfectly safe for 16GB RAM
    raw_embs = np.zeros((N_p, 384), dtype=np.float32)
    meta_cache = {}
    item_cats_map = {}
    brand_cat_cnt = defaultdict(int)
    brand_prices = defaultdict(list)

    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    text_batch = []
    idx_batch = []

    count = 0
    for meta in parse_file(METADATA_FILE):
        count += 1
        if count % 500000 == 0: _log(f"  ... encoded text from {count:,} metadata lines ...")

        asin = meta.get('asin')
        if asin in product_to_id:
            local_idx = product_to_id[asin] - N_u

            # Cache non-text data
            price = _parse_price(meta.get('price'))
            raw_cats = meta.get('categories', []) or []
            flat_cats = [c for sub in raw_cats if isinstance(sub, list) for c in sub]
            if not flat_cats: flat_cats = ['Unknown']

            meta_cache[asin] = {
                'price': price,
                'cats': flat_cats,
            } # NOTICE: 'text' is strictly omitted from caching here
            item_cats_map[asin] = flat_cats

            brand = item_to_seller.get(asin)
            if brand:
                brand_cat_cnt[brand] += 1
                brand_prices[brand].append(price)

            # Extract text and batch for immediate processing
            raw_desc = meta.get('description', '') or ''
            desc = ' '.join(raw_desc) if isinstance(raw_desc, list) else raw_desc
            title = meta.get('title', '') or ''

            text_batch.append(title + ' ' + desc)
            idx_batch.append(local_idx)

            # Encode and flush batch to keep RAM low
            if len(text_batch) >= CHUNK_SIZE:
                embs = encoder.encode(text_batch, batch_size=ENCODE_BATCH, show_progress_bar=False, convert_to_numpy=True)
                for i, l_idx in enumerate(idx_batch):
                    raw_embs[l_idx] = embs[i]
                text_batch.clear()
                idx_batch.clear()

    # Flush remaining text batch
    if text_batch:
        embs = encoder.encode(text_batch, batch_size=ENCODE_BATCH, show_progress_bar=False, convert_to_numpy=True)
        for i, l_idx in enumerate(idx_batch):
            raw_embs[l_idx] = embs[i]
        text_batch.clear()
        idx_batch.clear()

    _log("  -> Applying PCA to the 384-D text embeddings...")
    text_96d = safe_pca_transform(raw_embs, TEXT_DIM)

    # Immediately clear the 384-D matrix to free 768MB RAM
    del raw_embs
    gc.collect()

    _log("Pass 3A – Building final V_p (product nodes, 128-D)...")
    V_p = np.zeros((N_p, FEAT_DIM), dtype=np.float32)

    cats_list, raw_stats = [], []
    brand_by_prodidx = {}

    for local_idx, asin in enumerate(all_items):
        data = meta_cache.get(asin, {'price':0.0, 'cats':['Unknown']})
        cats_list.append(data['cats'])

        if asin in item_to_seller:
            brand_by_prodidx[local_idx] = item_to_seller[asin]

        s = item_stats.get(asin, {})
        cnt = s.get('c', 0) or 1
        avg_r = s.get('r_sum', 0) / cnt
        raw_stats.append([
            data['price'], float(s.get('c', 0)), avg_r,
            max(0, (s.get('r_sq', 0) / cnt) - avg_r**2), s.get('t_max', 0) - s.get('t_min', 0),
            s.get('v_sum', 0) / cnt, s.get('wc_sum', 0) / cnt, s.get('mis_sum', 0) / cnt
        ])

    mlb = MultiLabelBinarizer(sparse_output=False)
    cat_multihot = mlb.fit_transform(cats_list) if cats_list else np.zeros((0,1))
    cat_24d = safe_pca_transform(cat_multihot, PROD_CAT_DIM)

    stats_arr = np.array(raw_stats, dtype=np.float32)
    stats_8d = MinMaxScaler().fit_transform(stats_arr) if len(stats_arr) > 0 else np.zeros((0, PROD_STAT_DIM))

    V_p[:, :TEXT_DIM] = text_96d
    V_p[:, TEXT_DIM:TEXT_DIM+PROD_CAT_DIM] = cat_24d
    V_p[:, TEXT_DIM+PROD_CAT_DIM:] = stats_8d

    _log("Pass 3B – Building V_u (user nodes, 128-D)...")
    V_u = np.zeros((N_u, FEAT_DIM), dtype=np.float32)

    user_cat_sets = defaultdict(set)
    for asin, uids in item_buyers.items():
        for uid in uids:
            user_cat_sets[uid].update(item_cats_map.get(asin, ['Unknown']))

    user_cats_list, raw_u_stats = [], []
    for local_idx, uid in enumerate(all_users):
        user_cats_list.append(list(user_cat_sets.get(uid, ['Unknown'])))
        s = user_stats.get(uid, {})
        cnt = s.get('c', 0) or 1
        avg_r = s.get('r_sum', 0) / cnt
        raw_u_stats.append([
            float(s.get('c', 0)), avg_r, max(0, (s.get('r_sq', 0) / cnt) - avg_r**2),
            s.get('pos', 0) / cnt, s.get('neg', 0) / cnt, s.get('wc_sum', 0) / cnt,
            s.get('lex_sum', 0) / cnt, s.get('mis_sum', 0) / cnt, s.get('v_sum', 0) / cnt,
            (s.get('t_max', 0) - s.get('t_min', 0)) / cnt,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])

    user_multihot = mlb.transform(user_cats_list) if user_cats_list else np.zeros((0,1))
    cat_112d = safe_pca_transform(user_multihot, USER_CAT_DIM)

    stats_u_arr = np.array(raw_u_stats, dtype=np.float32)
    stats_16d = MinMaxScaler().fit_transform(stats_u_arr) if len(stats_u_arr) > 0 else np.zeros((0, USER_STAT_DIM))

    V_u[:, :USER_CAT_DIM] = cat_112d
    V_u[:, USER_CAT_DIM:] = stats_16d

    _log("Pass 3C – Building V_s (seller nodes, 128-D)...")
    V_s = np.zeros((N_s, FEAT_DIM), dtype=np.float32)

    catalog_vecs = V_p[:, :120].astype(np.float32)
    seller_prod_indices = defaultdict(list)
    for prod_idx, brand in brand_by_prodidx.items():
        seller_prod_indices[brand].append(prod_idx)

    mean_vecs = np.zeros((N_s, 120), dtype=np.float32)
    raw_s_stats = []

    for local_idx, brand in enumerate(all_sellers):
        p_idxs = seller_prod_indices.get(brand, [])
        if p_idxs: mean_vecs[local_idx] = np.mean(catalog_vecs[p_idxs], axis=0)

        prices = brand_prices.get(brand, [0.0])
        tot_rev = sum([item_stats.get(all_items[idx], {}).get('c', 0) for idx in p_idxs])
        rep_sum = sum([item_stats.get(all_items[idx], {}).get('r_sum', 0) for idx in p_idxs])

        raw_s_stats.append([
            float(brand_cat_cnt.get(brand, 0)), np.mean(prices), np.var(prices) if len(prices) > 1 else 0.0,
            float(tot_rev), rep_sum / max(1, tot_rev),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])

    cat_s_112d = safe_pca_transform(mean_vecs, SELLER_CAT_DIM)
    stats_s_arr = np.array(raw_s_stats, dtype=np.float32)
    stats_s_16d = MinMaxScaler().fit_transform(stats_s_arr) if len(stats_s_arr) > 0 else np.zeros((0, SELLER_STAT_DIM))

    V_s[:, :SELLER_CAT_DIM] = cat_s_112d
    V_s[:, SELLER_CAT_DIM:] = stats_s_16d

    # ─── PASS 4: Build CSR Adjacency Topology ────────────────────────────────
    _log("Pass 4 – Building Unified CSR Topology Binaries...")

    epu_src, epu_dst = [], []
    for asin, uids in item_buyers.items():
        pid = product_to_id[asin]
        for uid in uids:
            u_global = user_to_id[uid]
            epu_src.append(u_global)
            epu_dst.append(pid)

    eps_src, eps_dst = [], []
    for asin, brand in item_to_seller.items():
        pid = product_to_id[asin]
        sid = seller_to_id[brand]
        eps_src.append(pid)
        eps_dst.append(sid)

    euu_src, euu_dst = [], []
    for asin, uids in item_buyers.items():
        users = sorted(list(set([user_to_id[u] for u in uids])))
        if len(users) > MAX_USERS_PER_PROD_EUU:
            users = random.sample(users, MAX_USERS_PER_PROD_EUU)
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                euu_src.extend([users[i], users[j]])
                euu_dst.extend([users[j], users[i]])

    edges_epu = _build_csr_with_self_loops(epu_src, epu_dst, N_u, src_offset=0, prefix="epu")
    edges_epu_T = _build_csr_with_self_loops(epu_dst, epu_src, N_p, src_offset=N_u, prefix="epu_T")

    edges_eps = _build_csr_with_self_loops(eps_src, eps_dst, N_p, src_offset=N_u, prefix="eps")
    edges_eps_T = _build_csr_with_self_loops(eps_dst, eps_src, N_s, src_offset=N_u+N_p, prefix="eps_T")

    edges_euu = _build_csr_with_self_loops(euu_src, euu_dst, N_u, src_offset=0, prefix="euu")

    # ─── FINAL SAVE ──────────────────────────────────────────────────────────
    _log("Saving Output Memmaps & Mappings...")
    V_p_16 = V_p.astype(np.float16)
    V_u_16 = V_u.astype(np.float16)
    V_s_16 = V_s.astype(np.float16)

    save_memmap(V_p_16, 'V_p_features.memmap')
    save_memmap(V_u_16, 'V_u_features.memmap')
    save_memmap(V_s_16, 'V_s_features.memmap')

    X = np.zeros((N_u + N_p + N_s, FEAT_DIM), dtype=np.float16)
    X[:N_u] = V_u_16
    X[N_u:N_u+N_p] = V_p_16
    X[N_u+N_p:] = V_s_16
    save_memmap(X, 'X_combined.memmap')

    with open('node_id_mappings.json', 'w') as f:
        json.dump({'product_map': product_to_id, 'user_map': user_to_id, 'seller_map': seller_to_id}, f)

    with open('node_counts.json', 'w') as f:
        json.dump({"users": N_u, "products": N_p, "sellers": N_s, "total": N_u+N_p+N_s}, f)

    gpu_gb = (V_p_16.nbytes + V_u_16.nbytes + V_s_16.nbytes) / 1e9

    text = '\n'.join([
        "═" * 62, "  GNN-EADD Stage 1 Preprocessing Statistics", "═" * 62, "",
        "  Node counts",
        f"    |V_u|  Users    : {N_u:>14,}",
        f"    |V_p|  Products : {N_p:>14,}",
        f"    |V_s|  Sellers  : {N_s:>14,}", "",
        "  Edge counts  (Includes GCN Self-Loops)",
        f"    |E_pu| User->Product  : {edges_epu:>14,}",
        f"    |E_up| Product->User  : {edges_epu_T:>14,}",
        f"    |E_ps| Product->Seller: {edges_eps:>14,}",
        f"    |E_sp| Seller->Product: {edges_eps_T:>14,}",
        f"    |E_uu| User->User     : {edges_euu:>14,}", "",
        "  Feature matrices  (float16, 128-D)",
        f"    V_p : shape {V_p_16.shape}  ->  {V_p_16.nbytes/1e6:.1f} MB",
        f"    V_u : shape {V_u_16.shape}  ->  {V_u_16.nbytes/1e6:.1f} MB",
        f"    V_s : shape {V_s_16.shape}  ->  {V_s_16.nbytes/1e6:.1f} MB", "",
        f"  Total Expected GPU VRAM Footprint ≈ {gpu_gb:.2f} GB", "═" * 62,
    ]) + '\n'

    print(text)
    _log(f"Pipeline complete. Total time: {(time.time() - t0)/60:.1f} min")

if __name__ == "__main__":
    main()
