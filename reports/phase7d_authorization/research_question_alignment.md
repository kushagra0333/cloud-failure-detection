# Phase 7D — Final Research-Question Alignment Report

**Date:** October 28, 2026  
**Analyst / Role:** Senior ML Research Engineer & Data-Pipeline Auditor (DeepMind Antigravity)  
**Governing Protocol:** Protocol v2.0 ([results/phase7b_1/phase7b_evaluation_protocol_v2.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b_1/phase7b_evaluation_protocol_v2.json))  
**Governing Frozen Split:** Candidate A ([results/phase7a_4/candidate_a_job_grouped_split.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/candidate_a_job_grouped_split.csv))  
**Input Reconstructed Dataset:** [`data/processed/phase7c/`](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7c/)  
**Machine-Readable Alignment Ledger:** [`results/phase7d_authorization/research_question_alignment.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7d_authorization/research_question_alignment.json)  

---

## 1. Executive Summary & Safeguard Confirmations

In accordance with strict instructions for this alignment check:
- **Zero Model Execution:** No models were trained, fine-tuned, evaluated, or benchmarked.
- **Quarantine Invariant:** The historical Phase 6 confirmatory holdout remains strictly sealed and unopened.
- **Immutable Artifacts:** The Phase 7C data arrays, scaler configurations, Candidate A split manifests, and Protocol v2.0 remain untouched.

This report resolves the critical methodological alignment questions between Protocol v2.0, the diagnostic motivation of Phase 7C/7D, the confirmatory status of Random Forest, and statistical bootstrapping claims.

---

## 2. Explicit Answers to Alignment Questions

### Question 1: What is Phase 7D's Primary Research Question?

> **Question:** Is Phase 7D's primary research question:  
> **A.** Whether the corrected pipeline improves the overall model-selection outcome, or  
> **B.** Whether removing the suspected legacy preprocessing contamination improves Random Forest specifically?

#### **Answer: Primary is (A); Nested Diagnostic is (B).**

1. **Protocol v2.0 Governing Scope (Option A):**  
   Under the ratified Protocol v2.0 Section 3, the experiment is architected as a **three-stage candidate-selection tournament** across four distinct inductive biases (Random Forest, Temporal LSTM, Compact Transformer, Representation Hybrid). Protocol v2.0 does *not* pre-select Random Forest as the designated confirmatory subject. Random Forest is an eligible contender among four models. The overarching confirmatory question governed by Protocol v2.0 is:
   $$\text{\textit{Does the clean, leakage-free pipeline enable the validation selection procedure to find a verifiable Operational Winner on unseen parent jobs?}}$$

2. **The Nested Diagnostic Question (Option B):**  
   The historical motivation for Phase 7C arose because Random Forest failed operational gates in Phase 7B under inherited preprocessing contamination (the $2.500000$ constant and overlapping parent-job scaling). Testing whether Random Forest specifically improves on clean data is a **vital comparative diagnostic**. However, within Protocol v2.0, this diagnostic question is structurally nested:
   - If Random Forest achieves the highest Validation Macro-F1 (as it did in Phase 7B), it becomes $M^*$ and its test results answer **both (A) and (B) confirmatory**.
   - If a sequential architecture (e.g. Transformer or Hybrid) outperforms Random Forest on clean validation data and becomes $M^*$, then (A) receives the official confirmatory evaluation, while (B) is evaluated as a descriptive scientific comparison.

---

### Question 2: Can Random Forest Receive a Confirmatory Interpretation if Another Architecture is Selected as $M^*$?

> **Question:** Under Protocol v2.0, can Random Forest's test result receive a confirmatory interpretation if another architecture is selected as $M^*$? Distinguish descriptive test metrics from the official confirmatory verdict.

#### **Answer: NO. Confirmatory interpretation is strictly reserved for $M^*$.**

Protocol v2.0 Section 3.3 (lines 40–55) establishes a strict firewall between confirmatory evaluation and descriptive benchmarks:

1. **The Confirmatory Subject ($M^*$ Only):**  
   - Only the single architecture that qualifies through Stage 2 validation gates (or achieves the highest Validation Macro-F1 in fallback) is crowned as $M^*$ and permanently frozen.
   - **Only $M^*$ is evaluated against Test Gate 1 and Test Gate 2.**
   - Only $M^*$ can yield the official confirmatory verdict: `OPERATIONAL_WINNER` or `NO_OPERATIONAL_WINNER`.
2. **Descriptive Scientific Benchmarks (The Non-Selected Models):**  
   - If an architecture other than Random Forest (e.g. Compact Transformer) is selected as $M^*$, Random Forest's test set evaluation is classified strictly as a **Descriptive Scientific Benchmark**.
   - Under Protocol v2.0 line 53: *"Under no circumstances can a non-selected model replace $M^*$ as the winner based on test performance."*
   - Therefore, in that scenario, Random Forest's test metrics would be reported with full transparency to diagnose the effect of clean data relative to Phase 7B, but its metrics **cannot receive an official confirmatory gate verdict**.

---

### Question 3: Does Validation Selection Directly Test the Random Forest Hypothesis?

> **Question:** Does the prescribed four-model validation selection procedure directly test the Random Forest-specific hypothesis, or does it primarily evaluate the overall candidate-selection procedure?

#### **Answer: It primarily evaluates the overall candidate-selection procedure.**

1. **Competitive Multi-Model Selection:**  
   Stage 2 validation selection evaluates whether a multi-architecture search over tabular snapshot, recurrent, self-attention, and hybrid representations can identify a qualified candidate.
2. **Indirect Test of Random Forest:**  
   The procedure does *not* directly isolate Random Forest as a fixed hypothesis. If clean raw data significantly benefits the temporal attention mechanism of the Transformer or the representation capacity of the Hybrid, those architectures may surpass Random Forest on validation metrics.
3. In that event, the validation selection procedure succeeds in its primary goal (identifying the strongest operational candidate on clean data), but it does not execute a single-model confirmatory test on Random Forest alone.

---

### Question 4: Recommended Strategic Path

> **Question:** Recommend one path:  
> - **Path A:** Preserve Protocol v2.0, run the four-model experiment exactly as specified, and limit the primary conclusion to the overall pipeline/model-selection outcome. Treat Random Forest-specific findings as descriptive evidence.  
> - **Path B:** If Random Forest-specific confirmation is essential, prepare a separately versioned protocol amendment defining Random Forest as the primary confirmatory candidate and specifying its validation selection, threshold freezing, test evaluation, metrics, and inference procedure. Do not evaluate the test set until that amendment is approved.

#### **Recommendation: `PATH A (Recommended)`**

```
========================================================================================
                          RECOMMENDED STRATEGIC PATH: PATH A
========================================================================================
Preserve Protocol v2.0 (Four-Model Tournament; Overall Pipeline Confirmation)
- Primary Confirmatory Scope : End-to-end pipeline and validation candidate selection
- Diagnostic Comparative Scope: Descriptive comparison of RF (and all models) vs Phase 7B
========================================================================================
```

#### Detailed Justification for Path A:
1. **Methodological Continuity & Exact 1-to-1 Comparability:**  
   Phase 7B ran the exact four-model tournament under Protocol v2.0. By executing the identical four-model tournament in Phase 7D, we maintain complete fidelity with Phase 7B. Every model's trajectory from contaminated to clean data can be audited side-by-side.
2. **Zero Protocol Churn:**  
   Path B would require halting the research to draft, debate, and approve a "Protocol v3.0 / Amendment 3" solely to force Random Forest to be the test subject. This would artificially constrain the research to a 1990s tabular baseline and discard the modern sequence architectures already implemented.
3. **Natural Resolution of the Random Forest Question:**  
   - In Phase 7B, Random Forest won validation selection ($M^*$) because neural models were overly conservative.
   - If Random Forest wins validation selection again in Phase 7D, it **naturally receives the full confirmatory test evaluation**, satisfying both Path A and Path B objectives simultaneously.
   - If another architecture (e.g. Transformer or Hybrid) wins validation selection on clean data, that is an **enormously valuable scientific discovery**: it proves that clean raw features unlocked sequence modeling capabilities that were previously suppressed by legacy imputation artifacts!
   - In either scenario, Random Forest's test metrics will still be fully scored, bootstrapped, and compared against Phase 7B RF in the diagnostic analysis.
4. **Preserves Research Rigor:**  
   Path A honors the core tenet of Protocol v2.0: candidate selection must occur on validation data, and the test set is reserved for one-shot confirmation of the best validated candidate.

---

### Question 5: Confirmation of Bootstrap Limitations & Scope

> **Question:** Confirm the bootstrap's 11-parent-job limitation will be explicitly reported and that the two bootstrap levels will not be presented as independent proof of generalization.

#### **Explicit Confirmation:**

1. **Parent-Job Cluster Limitation ($J=11$):**  
   We formally confirm that Phase 7D reporting will explicitly highlight the small-sample cluster limitation of the test set ($J=11$ parent jobs). Because only 3 parent jobs in the test set are pure-normal, empirical resampling draws 0 pure-normal parent jobs in $\sim 3.1\%$ of bootstrap replicates. As mandated by Protocol v2.0 line 105, these replicates will be tracked as undefined denominators, and the exact valid replicate fraction will be reported alongside wide percentile intervals.
2. **Hierarchical Intra-Cluster Nature:**  
   We formally confirm that the two bootstrap levels:
   - **Level 1:** Parent-Job Clustered Bootstrap ($J=11$ clusters)
   - **Level 2:** Entity Clustered Bootstrap ($N=59$ clusters)  
   will **NOT** be presented as independent proof of generalization or cross-cluster transfer.
3. **Single Physical Datacenter Invariant:**  
   Candidate A is derived from a **single Google Borg cluster**. The bootstrap procedures quantify internal sampling variability across parent jobs and entities within this single cluster. They do not demonstrate generalization to other physical datacenters or future calendar years. Both intervals will be reported as complementary internal stability diagnostics.

---

## 3. Final Recommendation & Readiness Status

### **Path Decision:** **`PROCEED WITH PATH A UNDER PROTOCOL V2.0`**

- **Primary Confirmatory Question:** Does the clean Phase 7C pipeline enable Protocol v2.0's candidate-selection procedure to qualify and confirm an `OPERATIONAL_WINNER` on unseen parent jobs?
- **Diagnostic Comparative Question:** How do Random Forest's metrics shift between Phase 7B (contaminated) and Phase 7D (clean)? (Evaluated confirmatory if RF is selected as $M^*$; evaluated descriptively if another model is selected as $M^*$).

### Execution Status:
- Pre-training alignment check is **COMPLETE**.
- Model training, checkpoint generation, and validation/test scoring remain **STRICTLY PAUSED**.
- The pipeline is fully prepared to instantiate [scripts/train_and_evaluate_phase7d.py](file:///c:/Users/hp/Desktop/Research%20project/scripts/train_and_evaluate_phase7d.py) upon your explicit authorization to begin model training.
