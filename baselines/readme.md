# Baselines

This directory contains baseline methods used in the project for detecting anomalies in Amazon e-commerce data.

The baselines currently implemented are:

1. **Heuristic + K-Core labeling (`label_data.py`)**
2. **HeteroDOMINANT (`dominant.py`)**
3. **GraphSAGE Autoencoder (`sage_anomaly.py`)**

---

## 1) Heuristic + K-Core labeling (`label_data.py`)

### Overview
This script implements a multi-phase data processing pipeline to identify potentially fraudulent products, suspicious sellers, and anomalous user behavior. It combines traditional data filtering with **graph theory** (K-Core decomposition) to find dense subgraphs of suspicious activity.

### Graph construction
The algorithm treats the dataset as a **bipartite graph** $G = (U \cup P, E)$, where:
- **Nodes ($U, P$):** reviewers and products (ASINs)
- **Edges ($E$):** a review interaction between a user and a product

The core structural signal uses **K-Core Decomposition** (`networkx.core_number`).

### Product anomaly detection (heuristics)
A product is flagged as an anomaly (`is_anomaly`) if it meets **any** of the following criteria:

- **Fake Product Flag:** high ratings combined with suspiciously low prices.
  - *Logic:* Average Rating $\ge 4.8$ AND Price $< 15\%$ of the category median AND Brand name is missing from the Product Title.
- **Fake Seller Flag:** high engagement with low credibility.
  - *Logic:* "Also Bought" count $> 80$ AND Verified Purchase ratio $< 25\%$.
- **K-Core Anomaly:** the product belongs to the top $0.6\%$ of the most densely connected cores in the interaction graph.

### User anomaly detection (synchronized behavior)
A user is flagged if they exhibit **both** high connectivity and temporal synchronization:
- **Structural:** top $0.6\%$ of the K-Core distribution
- **Temporal:** "review bursts"—posting more than 35 reviews at the exact same Unix timestamp

### Key parameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **K-Core Quantile** | `0.994` | Only the top 0.6% of nodes by core number are considered "dense" anomalies. |
| **Price Threshold** | `0.15` | Products priced below 15% of their category median are flagged. |
| **Rating Floor** | `4.8` | Minimum rating to be considered for "Review Boosting" detection. |
| **Verified Ratio** | `< 0.25` | Flagged if fewer than 25% of reviews come from verified purchases. |
| **Burst Threshold** | `> 35` | Number of reviews posted at the same second to trigger a "Burst" flag. |
| **Related Count** | `> 80` | Minimum `also_buy` count to trigger high-volume seller checks. |

### Output files
- `labelling_meta.csv`: detailed metadata for products flagged as anomalies
- `labelling_5core.csv`: list of users flagged for high-density synchronized behavior
- `labelling_asin_meta.txt`: list of ASINs flagged via product heuristics
- `labelling_asin_5_core.txt`: ASINs associated with flagged anomalous users ("targeted" products)

---

## 2) HeteroDOMINANT (`dominant.py`)

### Overview
This script implements a **heterogeneous graph neural network (GNN)** inspired by DOMINANT (Deep Anomaly Detection on Attributed Networks). It models products, buyers, and sellers and flags nodes with high reconstruction error.

### Heterogeneous graph schema
- **Node types**
  - **Product:** features include normalized price and category hash
  - **Buyer:** nodes derived from the 5-core interaction dataset
  - **Seller:** nodes derived from brand metadata
- **Edge types**
  - `('buyer', 'reviews', 'product')`
  - `('seller', 'sells', 'product')`
  - inverse edges are created for bidirectional message passing

### Objective & scoring
The implementation focuses on **attribute reconstruction error**:

$$Loss = \text{MSE}(X_{product}, \hat{X}_{product})$$

Reconstruction score:

$$Score = \|X - \hat{X}\|_2$$

Thresholds used in the README:
- **Product threshold:** $Mean + 2.2\,\sigma$
- **Buyer threshold:** $Mean + 4.0\,\sigma$ (buyer score = average anomaly score of reviewed products)

### Parameters (as documented)
| Parameter | Value |
| :--- | :--- |
| `hidden_channels` | 64 |
| `out_channels` | 32 |
| `learning_rate` | 0.0005 |
| `weight_decay` | 1e-5 |
| `epochs` | 251 |
| `seed` | 42 |

---

## 3) GraphSAGE Autoencoder (`sage_anomaly.py`)

### Overview
This script implements an **unsupervised GraphSAGE autoencoder** for anomaly detection on a *product-to-product* graph.

### Graph construction (implementation-aligned)
- **Graph:** homogeneous **product-to-product** graph built from metadata relationships
- **Edges:** derived from `also_buy` and `also_viewed`
- **Node features:** a 2D vector
  1. **Price** (parsed from metadata and standardized)
  2. **Category ID** (hashed category, modulo 1000 in code)

> The script scales the 2D feature matrix using `StandardScaler`.

### Model
- 2-layer `SAGEConv` encoder
- linear decoder to reconstruct the original 2D input features
- training objective: MSE reconstruction loss
- anomaly score: $L_2$ distance between original and reconstructed feature vectors

### Notes
- The script additionally loads a 5-core interaction file to create a set of ASINs present in that dataset (used for filtering/reporting).

---

## Requirements
- `pandas`, `numpy`
- `networkx` (for k-core baseline)
- `torch`, `torch_geometric`
- `scikit-learn`
- `tqdm`
- `gzip`, `json` (standard library)
