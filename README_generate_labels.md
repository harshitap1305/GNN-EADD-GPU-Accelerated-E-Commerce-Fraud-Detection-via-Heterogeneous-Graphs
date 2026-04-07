# Anomaly Label Generator (Stage 2 Prep)

This script, `generate_labels.py`, processes the labeled anomaly data for e-commerce entities (users and products). It maps raw identifiers to the unified global graph space and creates a balanced dataset of anomalies and normal nodes.

## Prerequisites

Before running this script, ensure the following files are present in the `data/` directory:
* **`node_id_mappings.json`**: Generated during Stage 1; contains the `user_map` and `product_map`.
* **`labelling_asin_5_core.txt`**: A raw text file containing anomalous user IDs.
* **`labelling_asin_meta.txt`**: A raw text file containing anomalous product ASINs.

## Core Workflow

The script executes in four distinct phases:

### 1. Global ID Mapping
It loads the JSON mappings to ensure that any labeled anomaly actually exists in the current graph. If a user or product was filtered out during the 5-core preprocessing, it is safely ignored.

### 2. Anomaly Registration
It reads the user and product anomaly files, looks up their corresponding **Global IDs**, and compiles a master list of "Known Anomalies".

### 3. Balanced Negative Sampling
To prevent the model from becoming biased (since anomalies are rare), the script:
* Identifies all valid IDs in the graph (Users + Products).
* Subtracts the known anomalies to create a "Normal Pool".
* **Samples an equal number** of normal nodes to match the count of anomalies, creating a perfect 1:1 ratio for the Binary Cross Entropy (BCE) loss in Stage 2.

### 4. Compilation & Shuffling
The script assigns labels (**1 for Anomaly, 0 for Normal**), combines them with their indices, and shuffles the entire set. This ensures that when you split the data into training and validation sets later, both classes are well-represented.

## Output: `labels.npy`

The script produces a single NumPy file with the following structure:
* **Shape**: `[L, 2]`, where $L$ is the total number of samples (Anomalies + Normal Samples).
* **Column 0**: The **Global ID** of the node.
* **Column 1**: The **Binary Label** (0 or 1).

## Usage

Run the script from your terminal:
```bash
python generate_labels.py
```

> **Note**: A fixed random seed (`42`) is used to ensure that the "Normal" samples are reproducible across different runs.

---
*This file is a prerequisite for the semi-supervised GAT fine-tuning stage.*
