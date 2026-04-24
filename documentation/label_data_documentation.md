# `label_data.py` — Documentation

## Overview

Heuristic labelling pipeline that ingests raw Amazon product metadata and review data, then flags **anomalous products** and **anomalous users** using structural graph analysis and rule-based fraud profiles. The output is a set of labelled ground-truth files consumed by downstream GNN training stages.

---

## Entry Point

```python
run_pipeline(review_path: str, meta_path: str)
```

Default invocation (from `__main__`):

```python
run_pipeline('Electronics_5.json.gz', 'meta_Electronics.json.gz')
```

---

## Pipeline Phases

### Phase 1 — Load Product Metadata

- **Input**: gzipped JSONL (`meta_Electronics.json.gz`)
- **Extracts**: `asin`, `brand`, `title`, `price`, `category`, `also_buy_count`
- Price is cleaned to numeric; category is the deepest-level category.

### Phase 2 — Load Reviews

- **Input**: gzipped JSONL (`Electronics_5.json.gz`)
- **Extracts**: `reviewerID`, `asin`, `overall` (rating), `verified`, `unixReviewTime`

### Phase 3 — Structural Analysis

Builds a **bipartite graph** (user ↔ product) using NetworkX and computes the **k-core number** for every node.

> K-core number measures how deeply embedded a node is in a dense sub-graph — high values signal tightly interconnected clusters typical of fraud rings.

### Phase 4 — Product Anomaly Flagging

Three independent fraud signals, unified via logical OR:

| Flag | Condition | Rationale |
|---|---|---|
| `is_fake_product` | Rating ≥ 4.8 **AND** price < 15% of category median **AND** brand ≠ title | Suspiciously cheap, highly rated, brand-mismatched listings |
| `is_fake_seller` | `also_buy_count` > 80 **AND** verified ratio < 25% | Artificially inflated cross-sell networks with unverified purchases |
| `is_kcore_anomaly` | Core number ≥ 99.4th percentile | Structurally anomalous density |

**Unified flag**: `is_anomaly = max(is_fake_product, is_fake_seller, is_kcore_anomaly)`

### Phase 5 — User Anomaly Flagging

Two fraud profiles, unified via logical OR:

| Profile | Condition | Rationale |
|---|---|---|
| **Collusive Shill** | Core number ≥ 99.4th percentile **AND** verified ratio < 20% | Dense graph presence with almost no verified purchases — separates shills from legitimate power-users |
| **Spambot** | > 35 reviews at the exact same Unix timestamp | Physically impossible for a human; indicates automated posting |

### Phase 6 — Export

| Output File | Contents |
|---|---|
| `labelling_meta.csv` | Full rows of anomalous products |
| `labelling_5core.csv` | Full rows of anomalous users |
| `labelling_asin_meta.txt` | Anomalous product ASINs (one per line) |
| `labelling_asin_5_core.txt` | Anomalous user IDs (one per line) |

---

## Dependencies

`pandas`, `numpy`, `networkx`, `tqdm` (standard `gzip` / `json` from stdlib)

---

## Downstream Usage

The exported label files serve as **ground-truth supervision** for the Stage 2 GAT semi-supervised fine-tuning pipeline.
