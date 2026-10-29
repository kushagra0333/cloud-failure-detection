# Phase 6.3 — L=5 Model Training Under Ratified Protocol Amendment 2.1

**Execution State:** Controlled Model Training & Checkpoint Freezing (Pre-Evaluation Gate)  
**Governing Protocol:** Protocol Amendment 2.1 ([`results/phase6/protocol_l5_amendment_2_1.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json))  
**Immutable Historical Baseline:** Phase 6.2 Protocol ([`results/phase6/protocol_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5.json))  
**Model Checkpoints Directory:** [`models/checkpoints_l5/`](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_l5/)  
**Final Scientific Verdict:** `TRAINING_COMPLETE_READY_FOR_CONFIRMATORY_EVALUATION`

---

## 1. Executive Summary & Purpose

Following the formal ratification of **Protocol Amendment 2.1** in Phase 6.2.3, Phase 6.3 executes the clean, from-scratch training of all four pre-registered model architectures under the ratified $L=5$ sequence specification (5 consecutive 300s telemetry steps, 25-minute physical collection interval).

All four candidate architectures were trained strictly on the isolated development training partition ($N_{\text{train}} = 4,798$ windows across 367 entities). Preprocessing parameters were frozen exclusively from training data via [`configs/fitted_scaler.json`](file:///c:/Users/hp/Desktop/Research%20project/configs/fitted_scaler.json). Checkpoint selection and early stopping were governed solely by Validation Macro-F1 on the development validation partition ($N_{\text{val}} = 1,015$ windows across 78 entities).

**Strict Quarantine Invariant:** The 337 confirmatory holdout entities (137 failure entities, 200 normal entities) and their 5,511 primary evaluation windows remained sealed throughout this phase. **Zero holdout examples were loaded, evaluated, or scored, and zero decision thresholds were tuned.**

```
       ========================================================================
                         PHASE 6.3 TRAINING COMPLETION SUMMARY
       ========================================================================
       Governing Protocol Artifact              : protocol_l5_amendment_2_1.json
       Status                                   : RATIFIED_PRE_REGISTERED_FROZEN
       Original Protocol 2.0 (protocol_l5.json) : PRESERVED UNCHANGED (SHA-256 verified)
       Training Sequence Length L               : EXACTLY 5 STEPS (1,500s collection span)
       Input Dimensions                         : [batch_size, 5, 31] (Seq), [batch_size, 31] (Tabular)
       Checkpoints Directory                    : models/checkpoints_l5/
       Development Split Isolation              : 525 dev entities vs 337 holdout entities (0 overlap)
       Holdout Scoring Conducted                : ABSOLUTE ZERO (Sealed until Phase 6.4)
       Final Verdict                            : TRAINING_COMPLETE_READY_FOR_CONFIRMATORY_EVALUATION
       ========================================================================
```

---

## 2. Pre-Training Verification & Scientific Safeguards

Before training was initiated, all repository artifacts and boundary conditions were independently audited:

| Verification Item | Specification / Mandate | Empirical Verification | Audit Outcome |
| :--- | :--- | :--- | :---: |
| **Amended Protocol Status** | Status `RATIFIED_PRE_REGISTERED_FROZEN` | Loaded [`results/phase6/protocol_l5_amendment_2_1.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json); version verified `2.1` | **PASS** |
| **Original Protocol Preservation** | `protocol_l5.json` immutable | SHA-256 checksum: `ebf692598f2cb7115919c427127782f44ba054c655b3b76100aadb5955377bd1` | **PASS** |
| **Split Isolation** | Zero holdout entity overlap in dev | 525 dev entities $\cap$ 337 holdout entities $= \emptyset$ (0.0% overlap) | **PASS** |
| **Census Quarantine** | Zero 61-task census overlap in dev | 525 dev entities $\cap$ 61 quarantined tasks $= \emptyset$ (0.0% overlap) | **PASS** |
| **Tensor Input Dimensions** | Sequence models consume `[N, 5, 31]` | Verified: `X_train_l5`: `(4798, 5, 31)`, `X_val_l5`: `(1015, 5, 31)` | **PASS** |
| **Preprocessing Provenance** | Scaler fitted strictly on training data | Verified: [`configs/fitted_scaler.json`](file:///c:/Users/hp/Desktop/Research%20project/configs/fitted_scaler.json) fitted on dev train split | **PASS** |
| **Checkpoint Quarantine** | No reuse of legacy L=15 checkpoints | Checkpoints trained 100% from scratch and saved in `models/checkpoints_l5/` | **PASS** |
| **Holdout Quarantine** | Zero confirmatory holdout scoring | Zero holdout predictions computed; no holdout scoring performed | **PASS** |

---

## 3. Model Architectures & Training Specifications

Four models were trained from scratch under fixed random seed 42:

### Model 1: Class-Weighted Random Forest (Tabular Baseline)
- **Input Representation:** Latest-step feature vector $x_{\text{last}} \in \mathbb{R}^{31}$ at prediction anchor $t_{\text{pred}}$, capturing instantaneous and trailing causal aggregates.
- **Input Tensor Shape:** `[batch_size, 31]`
- **Hyperparameters:** `n_estimators=100`, `max_depth=15`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`.
- **Training Duration:** 0.30 seconds.
- **Checkpoint:** [`models/checkpoints_l5/random_forest_l5.joblib`](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_l5/random_forest_l5.joblib) (SHA-256: `7cba0f75bc11754b201d7961e3ee30968540caafe20deea93ac30372e9ce8481`).

### Model 2: Temporal LSTM (Recurrent Deep Architecture)
- **Architecture:** 2-layer causal LSTM consuming sequential observations without feature flattening.
- **Input Tensor Shape:** `[batch_size, 5, 31]`
- **Hidden Dimension:** 64 | **Layers:** 2 | **Dropout:** 0.2
- **Parameter Count:** **62,467 parameters** (LSTM layer 1: 24,832; layer 2: 33,280; classifier head: 4,355).
- **Loss Function:** Class-Weighted Cross-Entropy Loss with frozen weights:
  $$\mathcal{L}_{\text{LSTM}} = \text{CrossEntropyLoss}(\mathbf{w} = [1.0, 1.5, 2.0])$$
- **Optimizer & Scheduler:** `AdamW` ($\text{lr} = 1.0 \times 10^{-3}$, $\text{weight\_decay} = 1.0 \times 10^{-4}$); `ReduceLROnPlateau` ($\text{factor} = 0.5$, $\text{patience} = 5$, $\text{mode} = \text{'max'}$ on Val Macro-F1).
- **Early Stopping:** Monitored on Validation Macro-F1 with patience 10 (maximum 35 epochs).
- **Convergence:** Best validation Macro-F1 achieved at **Epoch 10** ($\text{Val Macro-F1} = 0.6365$); early stopping triggered at Epoch 20.
- **Training Duration:** 12.70 seconds.
- **Checkpoint:** [`models/checkpoints_l5/best_lstm_checkpoint_l5.pt`](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_l5/best_lstm_checkpoint_l5.pt) (SHA-256: `08f8cc36defe3771a1f5eda5a59ec434a3891105f11f6e32c6d59e7327a462d2`).

### Model 3: Compact Temporal Transformer (Attention Architecture)
- **Architecture:** Compact Transformer Encoder with multi-head self-attention and sinusoidal positional encodings over historical steps $L=5$.
- **Input Tensor Shape:** `[batch_size, 5, 31]`
- **Model Dimension ($d_{\text{model}}$):** 32 | **Attention Heads:** 4 ($\text{dim\_per\_head} = 8$) | **Layers:** 2
- **Feedforward Dimension ($d_{\text{ff}}$):** 64 | **Dropout:** 0.1 | **Max Seq Len:** 5 | **Pooling:** Last-step causal anchor
- **Parameter Count:** **19,331 parameters** (Input projection: 1,024; Encoder layers: 17,088; LayerNorm: 64; Classifier head: 1,155).
- **Loss Function:** Class-Weighted Cross-Entropy Loss with frozen weights:
  $$\mathcal{L}_{\text{Transformer}} = \text{CrossEntropyLoss}(\mathbf{w} = [1.0, 1.5, 2.0])$$
- **Optimizer & Scheduler:** `AdamW` ($\text{lr} = 5.0 \times 10^{-4}$, $\text{weight\_decay} = 1.0 \times 10^{-4}$); `ReduceLROnPlateau` ($\text{factor} = 0.5$, $\text{patience} = 5$, $\text{mode} = \text{'max'}$ on Val Macro-F1).
- **Early Stopping:** Monitored on Validation Macro-F1 with patience 10 (maximum 35 epochs).
- **Convergence:** Best validation Macro-F1 achieved at **Epoch 33** ($\text{Val Macro-F1} = 0.6401$).
- **Training Duration:** 41.44 seconds.
- **Checkpoint:** [`models/checkpoints_l5/best_transformer_checkpoint_l5.pt`](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_l5/best_transformer_checkpoint_l5.pt) (SHA-256: `a937a3c30d97dfa19ba048e47bf9095bb1203b5daec822d5dbd09f7a94ef937e`).

### Model 4: Representation-Fused Hybrid
- **Architecture:** Deep temporal backbone representation fused with instantaneous tabular features.
- **Temporal Backbone:** Frozen Compact Temporal Transformer encoder backbone yielding a 32-dimensional latent embedding $h_{\text{temp}} \in \mathbb{R}^{32}$.
- **Tabular Features:** Instantaneous tabular feature vector $x_{\text{last}} \in \mathbb{R}^{31}$ from Groups A–I.
- **Fused Input Dimension:** $\mathbf{z} = [h_{\text{temp}} \,\|\, x_{\text{last}}] \in \mathbb{R}^{63}$.
- **Classifier:** `RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42)`.
- **Training Procedure:** In-sample extraction on training split ($N_{\text{train}} = 4,798$). Zero validation or holdout data was used during feature fusion fitting.
- **Training Duration:** 0.51 seconds.
- **Checkpoint:** [`models/checkpoints_l5/representation_hybrid_l5.joblib`](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_l5/representation_hybrid_l5.joblib) (SHA-256: `3b83cce7a08b68832a4e21a73bf4b5894b910ca57a7bb6cff7f12e9dcfaec7f2`).

---

## 4. Development Validation Benchmark Results

> [!IMPORTANT]
> **Methodological Disclaimer:** The following metrics are computed strictly on the development validation partition ($N_{\text{val}} = 1,015$ windows across 78 entities). These are development-stage optimization results utilized strictly for model checkpoint selection. **They do not constitute confirmatory holdout performance.**

| Model Architecture | Input Shape | Param Count | Val Loss | Val Macro-F1 | Val Accuracy | Warning Recall | Critical Recall | Val Entity FAR | Val Window FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Baseline** | `[N, 31]` | N/A | N/A | **0.7174** | **0.9123** | 0.2917 | 0.6505 | **0.00%** (0/42) | **0.24%** (2/840) |
| **Temporal LSTM** | `[N, 5, 31]` | 62,467 | 0.5051 | 0.6365 | 0.8887 | 0.1806 | 0.6408 | 4.76% (2/42) | 2.02% (17/840) |
| **Compact Temporal Transformer** | `[N, 5, 31]` | 19,331 | 0.4894 | 0.6401 | 0.8887 | 0.2083 | 0.6214 | 2.38% (1/42) | 2.02% (17/840) |
| **Representation Hybrid** | `[N, 63]` | 19,331+ | N/A | 0.6460 | 0.8759 | 0.2500 | **0.6602** | 7.14% (3/42) | 4.40% (37/840) |

### Detailed Class-Level Performance on Validation Set

```
-------------------------------------------------------------------------------------------------------------------
Model                      Class        Precision    Recall       F1-Score     Support    PR-AUC
-------------------------------------------------------------------------------------------------------------------
Random Forest              NORMAL       0.9129       0.9976       0.9534       840        --
                           WARNING      0.8750       0.2917       0.4375       72         0.4696
                           CRITICAL     0.9178       0.6505       0.7614       103        0.7812

Temporal LSTM              NORMAL       0.9104       0.9798       0.9438       840        --
                           WARNING      0.5417       0.1806       0.2708       72         0.3477
                           CRITICAL     0.7586       0.6408       0.6947       103        0.6787

Compact Transformer        NORMAL       0.9113       0.9798       0.9443       840        --
                           WARNING      0.5357       0.2083       0.3000       72         0.3458
                           CRITICAL     0.7619       0.6214       0.6845       103        0.6908

Representation Hybrid      NORMAL       0.9234       0.9560       0.9394       840        --
                           WARNING      0.4091       0.2500       0.3103       72         0.3396
                           CRITICAL     0.6869       0.6602       0.6733       103        0.7011
-------------------------------------------------------------------------------------------------------------------
```

### Development Episode Detection Analysis

| Model | Total Val Episodes | Detected Episodes | Episode Recall | Mean Lead Time | Median Lead Time | Pure-Normal FAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 61 | 46 | 75.4% | 576.2s (9.6 min) | 331.1s (5.5 min) | 0.00% (0/42) |
| **Temporal LSTM** | 61 | 46 | 75.4% | 586.4s (9.8 min) | 331.1s (5.5 min) | 4.76% (2/42) |
| **Compact Transformer** | 61 | 47 | 77.0% | 617.9s (10.3 min) | 331.1s (5.5 min) | 2.38% (1/42) |
| **Representation Hybrid** | 61 | 48 | **78.7%** | 609.4s (10.2 min) | 331.1s (5.5 min) | 7.14% (3/42) |

---

## 5. Artifact Deliverables Summary

| Artifact File | Description | SHA-256 / Location |
| :--- | :--- | :--- |
| [`results/phase6/training_manifest_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/training_manifest_l5.json) | Complete training manifest with configs, durations, and environment details | Machine-readable JSON |
| [`results/phase6/validation_metrics_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/validation_metrics_l5.json) | Comprehensive development validation metrics and confusion matrices | Machine-readable JSON |
| [`models/checkpoints_l5/random_forest_l5.joblib`](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_l5/random_forest_l5.joblib) | Trained Random Forest tabular baseline checkpoint | `7cba0f75bc11754b201d7961e3ee30968540caafe20deea93ac30372e9ce8481` |
| [`models/checkpoints_l5/best_lstm_checkpoint_l5.pt`](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_l5/best_lstm_checkpoint_l5.pt) | Trained Temporal LSTM neural checkpoint (best epoch 10) | `08f8cc36defe3771a1f5eda5a59ec434a3891105f11f6e32c6d59e7327a462d2` |
| [`models/checkpoints_l5/best_transformer_checkpoint_l5.pt`](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_l5/best_transformer_checkpoint_l5.pt) | Trained Compact Temporal Transformer neural checkpoint (best epoch 33) | `a937a3c30d97dfa19ba048e47bf9095bb1203b5daec822d5dbd09f7a94ef937e` |
| [`models/checkpoints_l5/representation_hybrid_l5.joblib`](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_l5/representation_hybrid_l5.joblib) | Trained Representation Hybrid Random Forest classifier | `3b83cce7a08b68832a4e21a73bf4b5894b910ca57a7bb6cff7f12e9dcfaec7f2` |
| [`data/processed/l5/`](file:///c:/Users/hp/Desktop/Research%20project/data/processed/l5/) | Sliced L=5 development sequence arrays (`[N, 5, 31]`) and metadata CSVs | Directory (6 files) |
| [`tests/test_phase6_3_l5_training.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase6_3_l5_training.py) | Automated test suite verifying checkpoint shapes, isolation, and provenance | 8/8 Tests Passing |
| [`reports/phase6_3_l5_training.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/phase6_3_l5_training.md) | Comprehensive scientific audit report for Phase 6.3 training | Finalized |

---

## 6. Verification and Test Suite Execution

A dedicated automated test suite ([`tests/test_phase6_3_l5_training.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase6_3_l5_training.py)) was executed to verify checkpoint provenance, sequence tensor shapes, parameter counts, and split isolation:

```bash
.\.venv\Scripts\pytest -v tests/test_phase6_3_l5_training.py
```

### Execution Summary
- `test_checkpoint_existence_and_loading`: **PASSED** (All 4 checkpoints load cleanly).
- `test_sequence_dimensions_and_shapes`: **PASSED** (Sequence models consume `[N, 5, 31]`, tabular consumes `[N, 31]`, hybrid consumes `[N, 63]`).
- `test_parameter_counts_match_specification`: **PASSED** (LSTM: 62,467; Transformer: 19,331).
- `test_split_isolation_and_no_holdout_leakage`: **PASSED** (0 entity overlap between dev and holdout).
- `test_preprocessing_provenance`: **PASSED** (Fitted scaler transforms cleanly).
- `test_training_manifest_and_validation_metrics_schema`: **PASSED** (Manifest and validation metrics verified).
- `test_protocol_preservation_invariance`: **PASSED** (Original protocol SHA-256 and amended protocol status intact).
- `test_no_confirmatory_holdout_scoring`: **PASSED** (Zero confirmatory predictions computed).

**Result:** 8 of 8 Phase 6.3 tests passed.  
**Full Repository Test Suite:** `111 passed in 18.25s` (all 103 previous tests + 8 Phase 6.3 tests passing).

---

## 7. Final Phase 6.3 Verdict

```
################################################################################
#                                                                              #
#                          FINAL PHASE 6.3 VERDICT:                            #
#            TRAINING_COMPLETE_READY_FOR_CONFIRMATORY_EVALUATION               #
#                                                                              #
################################################################################
```

**Verdict Rationale:**  
All four pre-registered model architectures (Random Forest baseline, Temporal LSTM, Compact Temporal Transformer, and Representation Hybrid) have been trained strictly from scratch under Protocol Amendment 2.1 using $L=5$ sequence inputs. Preprocessing and checkpoint selection adhered strictly to development training and validation boundaries. Parameter counts match specifications exactly. Zero holdout data was scored, zero decision thresholds were tuned, and the sealed confirmatory holdout remains 100% unaccessed.

The codebase is officially ready to advance to **Phase 6.4: Frozen Confirmatory Holdout Evaluation**.
