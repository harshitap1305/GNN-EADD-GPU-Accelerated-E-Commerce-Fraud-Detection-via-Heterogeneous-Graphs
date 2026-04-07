This `README.md` provides a comprehensive technical overview of the `label_data.py` script, which is designed to detect anomalous behavior in e-Marketplace datasets (specifically Amazon review data) using structural graph analysis and heuristic-based flagging.

---

# E-Commerce Anomaly Detection Pipeline

This script implements a multi-phase data processing pipeline to identify potentially fraudulent products, suspicious sellers, and anomalous user behavior. It combines traditional data filtering with **Graph Theory** (K-Core decomposition) to find dense subgraphs of suspicious activity.

## Algorithm Overview

The detection logic is split into two primary entities: **Products** (Metadata) and **Users** (Reviews).

### 1. Graph Construction & Structural Analysis
The algorithm treats the dataset as a **Bipartite Graph** $G = (U \cup P, E)$, where:
* **Nodes ($U, P$):** Represent unique Reviewers and unique Products (ASINs).
* **Edges ($E$):** Represent a review interaction between a user and a product.

The core of the structural analysis uses **K-Core Decomposition**. A $k$-core is a maximal subgraph where every node has at least degree $k$. High core numbers often indicate "shill networks" or highly synchronized clusters of users and products.

### 2. Product Anomaly Detection (Heuristics)
A product is flagged as an anomaly (`is_anomaly`) if it meets **any** of the following criteria:

* **Fake Product Flag:** High ratings combined with suspiciously low prices.
    * *Logic:* Average Rating $\ge 4.8$ AND Price $< 15\%$ of the category median AND Brand name is missing from the Product Title.
* **Fake Seller Flag:** High engagement with low credibility.
    * *Logic:* "Also Bought" count $> 80$ AND Verified Purchase ratio $< 25\%$.
* **K-Core Anomaly:** The product belongs to the top $0.6\%$ of the most densely connected cores in the interaction graph.

### 3. User Anomaly Detection (Synchronized Behavior)
A user is flagged if they exhibit **both** high connectivity and temporal synchronization:
* **Structural:** The user is in the top $0.6\%$ of the K-Core distribution.
* **Temporal:** The user exhibits "Review Bursts"—defined as posting more than 35 reviews at the exact same Unix timestamp.

---

## Parameter Reference

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **K-Core Quantile** | `0.994` | Only the top 0.6% of nodes by core number are considered "dense" anomalies. |
| **Price Threshold** | `0.15` | Products priced below 15% of their category median are flagged. |
| **Rating Floor** | `4.8` | Minimum rating to be considered for "Review Boosting" detection. |
| **Verified Ratio** | `< 0.25` | Flagged if fewer than 25% of reviews come from verified purchases. |
| **Burst Threshold** | `> 35` | Number of reviews posted at the same second to trigger a "Burst" flag. |
| **Related Count** | `> 80` | Minimum "also_buy" count to trigger high-volume seller checks. |

---

## Pipeline Phases

1.  **Metadata Ingestion:** Loads JSON metadata, extracts categories, and cleans price strings into numeric floats.
2.  **Review Ingestion:** Loads interaction data (Reviewer ID, ASIN, Rating, Timestamps).
3.  **Graph Computation:** Builds the NetworkX graph and calculates `nx.core_number(G)`.
4.  **Product Labeling:** Groups by category to calculate medians and applies heuristic flags.
5.  **User Labeling:** Filters for synchronized bursts and intersects with high K-core users.
6.  **Export:** Generates CSV reports and TXT files containing anomalous ASINs for downstream modeling.

---

## Output Files

* `labelling_meta.csv`: Detailed metadata for products flagged as anomalies.
* `labelling_5core.csv`: List of users flagged for high-density synchronized behavior.
* `labelling_asin_meta.txt`: A simple list of ASINs flagged via product heuristics.
* `labelling_asin_5_core.txt`: ASINs associated with the flagged anomalous users (used to identify "Targeted" products).
<img width="719" height="418" alt="image" src="https://github.com/user-attachments/assets/732b79c3-10bf-4277-a85f-dee03edcfe21" />

---

## Requirements
* `pandas`, `numpy`
* `networkx` (for Graph algorithms)
* `tqdm` (for progress tracking)
* `gzip`, `json` (standard library)

This `README.md` provides a technical deep-dive into the `dominant.py` script, which implements a **Heterogeneous Deep Anomaly Detection** framework for e-commerce datasets.

---

This `README.md` provides a technical overview of the `dominant.py` script, which implements a **Heterogeneous Deep Anomaly Detection** framework for identifying suspicious products and buyers in e-commerce graphs.

---

# HeteroDOMINANT: Graph-Based Anomaly Detection

This script implements a **Heterogeneous Graph Neural Network (GNN)** inspired by the DOMINANT (Deep Anomaly Detection on Attributed Networks) architecture. It identifies anomalies by learning the normal relationships between products, buyers, and sellers, flagging nodes that deviate from these learned patterns.

## Algorithm Architecture

The model uses a multi-stage **Deep Autoencoder** specifically designed for heterogeneous graph data.

### 1. Heterogeneous Graph Construction
The script builds a complex graph structure representing the "Trinity" of e-commerce interactions:
* **Nodes:**
    * **Product:** Features include normalized price and category hash.
    * **Buyer:** Nodes derived from the 5-core interaction dataset.
    * **Seller:** Nodes derived from the 'brand' metadata.
* **Edges:**
    * `('buyer', 'reviews', 'product')`
    * `('seller', 'sells', 'product')`
    * Inverse edges are automatically created to allow bidirectional information flow during message passing.

### 2. Encoder-Decoder Mechanism
* **Hetero-Encoder:** Uses `HeteroConv` with `SAGEConv` operators to aggregate neighborhood information across different node and edge types.
* **Shared Latent Space:** Maps high-dimensional neighborhood data into a 32-dimensional embedding [z].
* **Attribute Decoder:** A linear layer that attempts to reconstruct the original product attributes (Price and Category).

---

## Mathematical Objective

The current implementation focuses on **Attribute Reconstruction Error**. The model is trained to minimize the difference between original features ($X$) and reconstructed features ($\hat{X}$):

$$Loss = \text{MSE}(X_{product}, \hat{X}_{product})$$

Anomalies are defined as nodes with a high **Reconstruction Score**, calculated using the $L_2$ norm:
$$Score = \|X - \hat{X}\|_2$$

---

## Parameter Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `hidden_channels` | 64 | Dimensionality of the internal GNN hidden layers. |
| `out_channels` | 32 | Size of the latent embedding used for reconstruction. |
| `learning_rate` | 0.0005 | Learning rate for the Adam optimizer. |
| `weight_decay` | 1e-5 | L2 regularization to prevent overfitting to common patterns. |
| `epochs` | 251 | Number of training iterations for convergence. |
| `seed` | 42 | Fixed random seed for reproducibility. |

---

## Scoring & Thresholding Logic

Once the model is trained, it performs inference to assign anomaly scores to both products and buyers:

### Product Scoring
Products are scored based on how poorly the model can reconstruct their price and category given their structural context.
* **Threshold:** $Mean + 2.2 \times \sigma$ (Standard Deviation).

### Buyer Scoring
Buyer anomalies are derived from their interaction history. A buyer's score is the **average anomaly score** of all products they have reviewed.
* **Threshold:** $Mean + 4.0 \times \sigma$ (Standard Deviation).

---

## Execution Workflow

1.  **Metadata Ingestion:** Extracts product price, brand (Seller), and category.
2.  **Interaction Ingestion:** Loads the 5-core review data to establish Buyer-Product links.
3.  **Normalization:** Scales product features using `MinMaxScaler` to a range of $[0, 1]$.
4.  **Unsupervised Training:** The model iterates 251 times to minimize reconstruction loss.
5.  **Anomalous Extraction:** Identifies and prints the total count of outlier products and buyers based on refined statistical thresholds.

---

## Requirements
* `torch` & `torch_geometric`
* `pandas`, `numpy`
* `scikit-learn`
* `tqdm`

  <img width="696" height="316" alt="image" src="https://github.com/user-attachments/assets/2af97f89-e7bb-4e20-87b1-c33e8b6879b6" />


This `README.md` provides a technical breakdown of the `sage_anomaly.py` script, which implements an **Unsupervised GraphSAGE Autoencoder** for identifying anomalous products and interactions in e-commerce graph data.

---

# GraphSAGE Anomaly Detection Pipeline

This repository contains an implementation of a structural anomaly detection system using **GraphSAGE** (SAGEPool/Graph Sample and Aggregate). Unlike standard heuristic models, this algorithm learns the "normal" structural and feature-based patterns of a product network and flags items that deviate significantly from these learned representations.

## Algorithm Architecture

The system uses an **Encoder-Decoder** architecture built on Graph Neural Networks (GNNs).

### 1. Graph Construction
The algorithm constructs a **Homogeneous Product-to-Product Graph**:
* **Nodes ($V$):** Represent unique products (ASINs).
* **Edges ($E$):** Represent "Related" relationships, specifically `also_buy` and `also_viewed` metadata connections.
* **Features ($X$):** A 2-dimensional feature vector for each node consisting of:
    1.  **Price:** Normalized numerical value.
    2.  **Category ID:** A hashed representation of the top-level product category.

### 2. GraphSAGE Encoder
The encoder consists of two `SAGEConv` layers. Instead of looking at a node in isolation, it aggregates features from a node’s local neighborhood to generate a latent embedding ($h$).

### 3. Linear Decoder & Reconstruction
The decoder is a linear layer that attempts to map the latent embeddings back to the original input features ($X$). 
* **Training Objective:** The model is trained to minimize **Mean Squared Error (MSE)** reconstruction loss: $Loss = \|X - \hat{X}\|^2$.
* **Anomaly Logic:** Nodes that are difficult to reconstruct (high reconstruction error) are considered anomalies because their feature-structure combination is statistically rare compared to the rest of the graph.


---

## Parameter Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `in_channels` | 2 | Input features (Price, Category ID). |
| `hidden_channels` | 64 | Dimension of the first hidden representation. |
| `out_channels` | 32 | Dimension of the latent embedding passed to the decoder. |
| `learning_rate` | 0.01 | Optimizer step size for Adam. |
| `epochs` | 31 | Number of full passes over the product graph. |
| `dropout` | 0.2 | Probability of dropping units to prevent overfitting. |
| `target_count` | 4500 | The specific number of anomalies the script attempts to extract. |

---

## Execution Workflow

1.  **Data Streaming:** Parses large `.json.gz` metadata files using a generator to minimize memory footprint.
2.  **Feature Scaling:** Applies `StandardScaler` to ensure price and category IDs are on a comparable scale for the GNN.
3.  **Unsupervised Training:** The model learns the manifold of "typical" product relationships and attributes.
4.  **Anomaly Scoring:** Calculates the $L_2$ norm (Euclidean distance) between the original feature $x$ and the reconstructed feature $\hat{x}$.
5.  **Targeted Extraction:** Automatically calculates a dynamic percentile threshold to identify exactly ~4,500 anomalous ASINs.

---

## Output Files

The script generates four distinct output files to separate raw product anomalies from those that also appear in the interaction (5-core) dataset:

* **`SAGE_meta_anomalies.csv`**: Full details (ASIN, score, price, category) of all flagged products.
* **`SAGE_meta_asins.txt`**: A simple list of ASINs for the flagged products.
* **`SAGE_five_core_anomalies.csv`**: A subset of anomalies that are present in the `Electronics_5.json.gz` interaction file.
* **`SAGE_five_core_asins.txt`**: A simple list of ASINs for the interaction-based anomalies.
<img width="718" height="401" alt="image" src="https://github.com/user-attachments/assets/f9ccafa1-c4d9-4a08-b5e4-83e6a66e3193" />

---

## Requirements
* `torch` & `torch_geometric` (PyG)
* `pandas` & `numpy`
* `scikit-learn` (for Scaling)
* `tqdm` (for progress visualization)
