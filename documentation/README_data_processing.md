# Data Preprocessing Pipeline

**File:** `data_preprocessing.py` — Converts raw Amazon JSON files into GPU-ready graph binaries for the GNN-EADD fraud detection system.

---

## Input Files

| File | Format | Contains |
|:---|:---|:---|
| `AMAZON_FASHION_5.json` (or another test data) | JSON Lines | Reviews — `reviewerID`, `asin`, `overall` (rating), `reviewText`, `unixReviewTime`, `vote` |
| `meta_AMAZON_FASHION.json` (or another test data) | JSON Lines | Product metadata — `asin`, `title`, `description`, `price`, `brand`, `categories` |

Both files support `.json` and `.json.gz` formats and are parsed line-by-line to keep RAM low.

---

## Output Files

| File | Format | Description |
|:---|:---|:---|
| `V_p_features.memmap` | float16, `(N_p, 128)` | Product feature vectors |
| `V_u_features.memmap` | float16, `(N_u, 128)` | User feature vectors |
| `V_s_features.memmap` | float16, `(N_s, 128)` | Seller feature vectors |
| `X_combined.memmap` | float16, `(N_total, 128)` | All nodes stacked: Users → Products → Sellers |
| `epu_*.bin`, `epu_T_*.bin` | int32 CSR | User ↔ Product edges (+ transpose) |
| `eps_*.bin`, `eps_T_*.bin` | int32 CSR | Product ↔ Seller edges (+ transpose) |
| `euu_*.bin` | int32 CSR | User ↔ User co-purchase edges |
| `node_id_mappings.json` | JSON | String ID → integer global ID maps |
| `node_counts.json` | JSON | `{ users, products, sellers, total }` |

All CSR files include GCN self-loops. Global ID layout: `[0, N_u)` Users → `[N_u, N_u+N_p)` Products → `[N_u+N_p, N_total)` Sellers.

---

## Pipeline — 4 Passes

### Pass 1 — Build Unified ID Space
- Scans metadata → collects valid ASINs and maps products to sellers (via cleaned `brand` field)
- Scans reviews → collects valid user IDs and builds `item_buyers[asin] → [users]`
- Assigns non-overlapping integer IDs to all Users, Products, and Sellers

### Pass 2 — Extract NLP Fraud Signatures
Streams reviews and computes **running aggregates** per product and per user (no raw text stored):

| Statistic | Description |
|:---|:---|
| Review count, rating mean & variance | Basic review profile |
| Temporal span (`t_max − t_min`) | Detects burst reviewing |
| Word count, lexical diversity | Short/templated review detection |
| VADER sentiment–rating mismatch | `abs(compound − (rating−3)/2)` — flags fake positivity |
| Helpful vote sum | Trust signal |
| Positive/negative ratio (users only) | Skewed rating behavior |

### Pass 3 — Build 128-D Feature Vectors

Every node in the graph gets a **128-dimensional feature vector**. The 128 dimensions are filled differently for each node type, but the process always follows the same pattern: **encode raw data → reduce dimensionality → normalize → concatenate into a single 128-D row**.

---

**Product Features (V_p) — 128-D = 96 + 24 + 8**

Construction steps:
1. **Text embedding (dims 0–95):** For each product, concatenate `title + " " + description` into one string. Feed it through the `all-MiniLM-L6-v2` sentence transformer → outputs a 384-D dense vector. This is done in **chunks of 10K texts** to avoid RAM overflow. After all products are encoded, apply `IncrementalPCA` to reduce all 384-D vectors down to **96-D**.
2. **Category encoding (dims 96–119):** Flatten each product's nested `categories` list (e.g. `[["Clothing", "Men", "Shirts"]]` → `["Clothing", "Men", "Shirts"]`). Use `MultiLabelBinarizer` to convert all products' category lists into a multi-hot matrix (one column per unique category across the dataset). Apply PCA to reduce this wide binary matrix to **24-D**.
3. **Behavioral stats (dims 120–127):** Take 8 statistics from Pass 2 — price, review count, avg rating, rating variance, temporal span, avg helpful votes, avg word count, avg sentiment–rating mismatch — and normalize them to `[0, 1]` using `MinMaxScaler`. These 8 values fill the last 8 dimensions.

Result: `V_p[i] = [text_96d[i] | cat_24d[i] | stats_8d[i]]` for each product `i`.

---

**User Features (V_u) — 128-D = 112 + 16**

Construction steps:
1. **Category preference (dims 0–111):** For each user, collect the **union of all categories** from every product they reviewed (e.g. if a user reviewed products in "Shoes", "Bags", and "Watches", their category set is `{"Shoes", "Bags", "Watches"}`). Convert this set into a multi-hot vector using the **same** `MultiLabelBinarizer` fitted on products. Apply PCA to reduce to **112-D**.
2. **Behavioral stats (dims 112–127):** Take 16 values from Pass 2 — review count, avg rating, rating variance, positive-review ratio, negative-review ratio, avg word count, avg lexical diversity, avg sentiment mismatch, avg helpful votes, temporal span per review, plus 6 reserved zeros — normalize with `MinMaxScaler`.

Result: `V_u[i] = [cat_pref_112d[i] | stats_16d[i]]` for each user `i`.

---

**Seller Features (V_s) — 128-D = 112 + 16**

Construction steps:
1. **Catalog embedding (dims 0–111):** For each seller (brand), find all their products. Take each product's already-built feature vector (first 120 dims = text + category parts) and **average them** across all the seller's products. This gives a 120-D "mean catalog profile". Apply PCA to reduce to **112-D**.
2. **Catalog stats (dims 112–127):** Compute 16 values — total product count, mean price, price variance, total review count, average reputation (mean rating across all products), plus 11 reserved zeros — normalize with `MinMaxScaler`.

Result: `V_s[i] = [mean_catalog_112d[i] | stats_16d[i]]` for each seller `i`.

---

**Final assembly:** All three matrices are cast from float32 → **float16** and saved as `.memmap` files. They are also stacked row-wise into `X_combined.memmap` in the order Users → Products → Sellers, matching the global ID layout.

### Pass 4 — Build CSR Graph Topology

All edges are stored in **Compressed Sparse Row (CSR)** format — a memory-efficient way to represent sparse graphs that GPUs can traverse directly. Instead of storing a full N×N adjacency matrix (which would be terabytes at 1.5M nodes), CSR only stores the actual connections using two arrays:

- **`row_ptr`** (size `N_src + 1`): `row_ptr[i]` to `row_ptr[i+1]` tells you *where* node `i`'s neighbors are in `col_idx`
- **`col_idx`** (size `|E|`): flat list of all destination node IDs, grouped by source node

**Example** — 3 nodes where node 0→{1,2}, node 1→{2}, node 2→{}:
```
row_ptr = [0, 2, 3, 3]     ← node 0 has 2 neighbors, node 1 has 1, node 2 has 0
col_idx = [1, 2, 2]        ← node 0's neighbors are 1,2; node 1's neighbor is 2
```

#### Edge Types Built

| Edge Type | Direction | How Edges Are Formed |
|:---|:---|:---|
| `E_pu` | User → Product | One edge for each (reviewer, product) pair from the reviews file |
| `E_pu_T` | Product → User | **Transpose** of E_pu — same edges but with source/destination swapped, so the GCN can pass messages in both directions |
| `E_ps` | Product → Seller | One edge for each product that has a valid brand |
| `E_ps_T` | Seller → Product | **Transpose** of E_ps |
| `E_uu` | User ↔ User | Co-purchase graph (see below) |

#### Co-Purchase Edges (E_uu)

For each product, all users who reviewed it are considered "co-purchasers". Edges are created between every pair:
1. Collect all unique user IDs who reviewed the product
2. If more than **25 users**, randomly sample 25 (prevents edge explosion on popular items — without this cap, a product with 1000 reviewers would create ~500K edges from that single product)
3. For every pair `(u_i, u_j)`, add **two directed edges**: `u_i → u_j` and `u_j → u_i`

This creates a dense social proximity graph that allows the GNN to detect **coordinated review rings** — groups of users who suspiciously all review the same products.


---

## Key Design Decisions

- **Seller nodes from brands** — Amazon data has no seller entity; cleaned `brand` field is used as a proxy
- **Chunked text encoding** — 10K texts encoded at a time to fit in 16 GB RAM
- **Co-purchase edge cap** — `MAX_USERS_PER_PROD_EUU = 25` prevents edge explosion on popular items
- **float16 output** — Halves GPU VRAM usage (~0.39 GB for all features)
- **IncrementalPCA** — Handles cases where samples < target dimensions by zero-padding

---

## Dependencies

`numpy`, `scikit-learn` (PCA, MinMaxScaler, MultiLabelBinarizer), `sentence-transformers` (all-MiniLM-L6-v2), `nltk` (VADER sentiment)
