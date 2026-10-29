# Split Boundary and Leakage Audit Report

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Document ID:** `reports/split_leakage_audit.md`  
**Phase:** Phase 2 — Data Pipeline, Labels, Windows, and Leakage-Safe Splits  
**Implementation Source:** [`src/splitting/leakage_safe_split.py`](file:///c:/Users/hp/Desktop/Research%20project/src/splitting/leakage_safe_split.py)  
**Scaler Source:** [`src/features/scaler.py`](file:///c:/Users/hp/Desktop/Research%20project/src/features/scaler.py)  
**Audit Status:** **100% PASSED (Zero Leakage Detected)**  

---

## 1. Splitting Strategy & Partitioning Architecture

To guarantee scientific validity and prevent both temporal lookahead and entity identity memorization, the dataset of $N = 10,970$ chronological sequence windows was partitioned using a strict **Entity-Grouped Partitioning Strategy**:

- **Partitioning Unit:** Task Entity (`entity_id = f"{job_id}_{task_index}"`).
- **Target Ratios:** 70% Train / 15% Validation / 15% Test.
- **Entity Disjointness Constraint:**
  $$\text{Entities}(D_{\text{train}}) \cap \text{Entities}(D_{\text{val}}) = \emptyset$$
  $$\text{Entities}(D_{\text{train}}) \cap \text{Entities}(D_{\text{test}}) = \emptyset$$
  $$\text{Entities}(D_{\text{val}}) \cap \text{Entities}(D_{\text{test}}) = \emptyset$$

### 1.1 Rationale for Entity-Grouped Partitioning

In cloud cluster operations, sequence models must generalize to **unseen task instances and jobs**, rather than predicting future steps of tasks whose historical behavior was seen in training. Entity-grouped splitting guarantees that:
1. No sliding window from any task in the test set shares history, rolling statistics, or context with the training set.
2. Even if a long-running task has overlapping windows, all of its windows reside entirely within exactly one partition.
3. The model's evaluation strictly reflects real-world generalization to new container workloads.

*(Note: The codebase also provides a pure `chronological` splitter with a 30-minute boundary embargo in [`src/splitting/leakage_safe_split.py`](file:///c:/Users/hp/Desktop/Research%20project/src/splitting/leakage_safe_split.py) for single-entity or continuous trace architectures).*

---

## 2. Partition Summary and Class Distributions

The partition yielded robust sample counts across all three classes in every partition:

| Partition | Entity Count | Window Count ($N$) | Percentage | Class 0 (NORMAL) | Class 1 (WARNING) | Class 2 (CRITICAL) | Total Pre-Failures |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** ($D_{\text{train}}$) | 1,911 | 7,851 | 71.57% | 5,804 (73.93%) | 367 (4.67%) | 1,680 (21.40%) | 2,047 (26.07%) |
| **Val** ($D_{\text{val}}$) | 409 | 1,536 | 14.00% | 1,260 (82.03%) | 74 (4.82%) | 202 (13.15%) | 276 (17.97%) |
| **Test** ($D_{\text{test}}$) | 410 | 1,583 | 14.43% | 1,333 (84.21%) | 81 (5.12%) | 169 (10.68%) | 250 (15.79%) |
| **Total** | **2,730** | **10,970** | **100.00%** | **8,397 (76.55%)** | **522 (4.76%)** | **2,051 (18.70%)** | **2,573 (23.45%)** |

---

## 3. Automated Leakage Audit Results

The automated audit suite in [`LeakageSafeSplitter.audit_leakage()`](file:///c:/Users/hp/Desktop/Research%20project/src/splitting/leakage_safe_split.py) ran four rigorous checks:

```python
audit_res = splitter.audit_leakage(meta_df, split_res, embargo_sec=1800.0)
```

| Audit Check | Verification Criterion | Result | Status |
| :--- | :--- | :--- | :--- |
| **1. Train $\cap$ Val Entity Overlap** | $\vert \text{Entities}(D_{\text{train}}) \cap \text{Entities}(D_{\text{val}})\vert == 0$ | 0 overlapping entities | **PASSED** |
| **2. Train $\cap$ Test Entity Overlap** | $\vert \text{Entities}(D_{\text{train}}) \cap \text{Entities}(D_{\text{test}})\vert == 0$ | 0 overlapping entities | **PASSED** |
| **3. Val $\cap$ Test Entity Overlap** | $\vert \text{Entities}(D_{\text{val}}) \cap \text{Entities}(D_{\text{test}})\vert == 0$ | 0 overlapping entities | **PASSED** |
| **4. Feature Lookahead Contamination** | All features computed strictly $\le t_{\text{pred}}$ | 0 future lookahead violations | **PASSED** |

---

## 4. Preprocessing Isolation & Scaler Provenance

In strict compliance with Master Engineering Prompt Rule 4 ("Fit imputers and scalers on training data only"):

1. **Training Split Fitting:**
   - Both the `StandardScaler` ($\mu_{\text{train}}, \sigma_{\text{train}}$) and missing-value median imputer ($\text{median}_{\text{train}}$) were fitted exclusively on the 7,851 training windows ($117,765$ time steps across 31 features).
   - Neither validation nor test observations contributed to parameter estimates.
2. **Artifact Serialization:**
   - Fitted parameters were serialized to [`configs/fitted_scaler.json`](file:///c:/Users/hp/Desktop/Research%20project/configs/fitted_scaler.json).
   - Zero NaN values remain in transformed tensors $X_{\text{train}}, X_{\text{val}}, X_{\text{test}}$.
3. **Artifact Shape Verification:**
   - $X_{\text{train}} \in \mathbb{R}^{7851 \times 15 \times 31}$, $y_{\text{train}} \in \{0, 1, 2\}^{7851}$
   - $X_{\text{val}} \in \mathbb{R}^{1536 \times 15 \times 31}$, $y_{\text{val}} \in \{0, 1, 2\}^{1536}$
   - $X_{\text{test}} \in \mathbb{R}^{1583 \times 15 \times 31}$, $y_{\text{test}} \in \{0, 1, 2\}^{1583}$
