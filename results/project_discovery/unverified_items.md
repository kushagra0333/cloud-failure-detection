# Register of Unverified Items, Limitations & Required Evidence

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Auditor:** Independent Machine Learning Scientific Auditor  
**Date:** October 27, 2026  
**Status:** Permanent Audit Register of Epistemological Boundaries

---

## 1. Overview

In scientific machine learning, maintaining honesty about what *cannot* be proven is just as important as reporting verified findings. Observational trace data, secondary benchmarks, and historical logs impose inherent epistemological limits.

This document records all scientific claims, physical interpretations, and assumptions in this research project that **cannot be independently verified** from the available repository code, raw Google cluster trace tables, or trained model artifacts. For each item, we document:
- The exact unverified claim or assumption;
- Why it cannot be verified from available evidence;
- The specific external evidence that would be required to verify it;
- The mandatory scientific caveat required when discussing it in academic writing.

---

## 2. Unverified Items Register

### Item 1: Physical Root Cause of Container Failures
- **Unverified Claim:** "Tasks fail due to application memory exhaustion, CPU thrashing, or hardware degradation."
- **Why Unverifiable:** The Google Cluster Trace 2011 dataset records only a single categorical event enumeration (`event_type == 3` for `FAIL`). It does not provide exit codes, OS signal numbers (e.g., SIGSEGV, SIGKILL), application stack traces, or kernel diagnostic messages (`dmesg`).
- **Required Evidence:** Machine-level syslog dumps, Linux kernel audit logs, container runtime exit codes, or hardware error logs (EDAC, MCE).
- **Mandatory Caveat:** All failures must be described strictly as *abnormal container terminations recorded by the scheduler*, without attributing specific hardware or software failure mechanisms.

---

### Item 2: Causal Role of Missing CPI Telemetry (`cpi_missing_fraction_6`)
- **Unverified Claim:** "Missing CPI telemetry is a failure mechanism or causal precursor of container death."
- **Why Unverifiable:** While Phase 6.6 and 6.7 confirmed a large statistical association ($d = +1.2$ to $+3.0$) between missing CPI and false-positive alerts, observational data cannot determine why the counters were missing. In Google cluster tracing, hardware performance counters (like Cycles Per Instruction) drop out during container thread idling, Borg scheduler descheduling, or collection daemon timeouts.
- **Required Evidence:** Scheduler runqueue profiling, CPU context-switch traces, and collector heartbeat records showing that CPI missingness was caused by container corruption rather than benign scheduling pauses.
- **Mandatory Caveat:** Missing CPI must be reported strictly as an *observational feature correlation reflecting collection artifacts or workload inactivity*, never as a physical failure mechanism.

---

### Item 3: Nature of Pre-Failure Alerts ($30 - 48\text{ Minutes}$ Before Failure)
- **Unverified Claim:** "Elevated alert rates (75%–98% FPR) on normal windows 30 to 48 minutes before a failure represent early detection of latent degradation."
- **Why Unverifiable:** Sequence models triggered alerts on 110–139 of the 147 pre-failure normal windows. While these windows occur prior to a verified failure, the available telemetry cannot establish whether the model was detecting genuine pre-crash instability or merely exhibiting baseline false-alarm drift. In particular, Random Forest triggered on only 16 of these 147 windows (10.9%), casting doubt on whether a consistent degradation signal existed across the 31 features.
- **Required Evidence:** Application-level transaction latency logs, heartbeats, or memory leak profiling confirming that the container was experiencing performance degradation 40 minutes prior to failure.
- **Mandatory Caveat:** Under Protocol Amendment 2.1, alerts occurring $> 30\text{ min}$ prior to failure are officially classified as false alarms. Any discussion of early degradation detection must be explicitly labeled as an *unverified observational hypothesis*.

---

### Item 4: Generalization to Modern Cloud Microservices (2026 Datacenters)
- **Unverified Claim:** "The findings and model comparisons generalize to modern cloud architectures."
- **Why Unverifiable:** The dataset originates from Google Borg cluster traces collected over 29 days in May 2011. Workloads in 2011 were predominantly monolithic MapReduce batch jobs, C++ binaries, and Java services. Modern cloud computing in 2026 relies heavily on container orchestration (Kubernetes), serverless functions, microservices, gRPC communication, and virtualization hypervisors with fundamentally different resource dynamics.
- **Required Evidence:** Evaluation on contemporary cloud telemetry datasets (e.g., Alibaba Cluster Trace 2020+, Azure Public Dataset 2020+, or private Kubernetes production telemetry).
- **Mandatory Caveat:** The findings must be scoped strictly to *heterogeneous batch container workloads under Borg-style cluster scheduling*, with explicit disclaimers regarding modern cloud generalization.

---

### Item 5: Production Operational Failure Prevalence (Assumed 0.5%)
- **Unverified Claim:** "The production operational failure rate is exactly 0.5%, making false alarms the dominant operational risk."
- **Why Unverifiable:** The project documents frequently cite a production failure prevalence of $0.5\%$ to justify the stringent $\le 5.0\%$ FPR and $\le 12.0\%$ FAR operational gates. However, the evaluation cohort itself was constructed as a case-control sample with an enriched failure prevalence (~12.7% pre-failure windows). The true instantaneous failure rate of production jobs across the entire Google trace fluctuates between 0.1% and 4.0% depending on job priority and scheduling class.
- **Required Evidence:** Full cluster-wide census logs computing instantaneous task failure rates across all 670,000+ jobs over the entire 29-day trace.
- **Mandatory Caveat:** The 0.5% failure prevalence must be cited as an *illustrative operational scenario* representing typical low-failure batch production environments, not an absolute constant.

---

### Item 6: Optimality of the 31-Feature Engineered Representation
- **Unverified Claim:** "The 31 handcrafted features represent the optimal feature set for cloud failure prediction."
- **Why Unverifiable:** No ablation study was performed comparing the 31 features against raw telemetry or alternative representation spaces (e.g., Fourier spectral features, Wavelet transforms, or learned autoencoder embeddings).
- **Required Evidence:** Systematic feature selection and representation ablation experiments on development partitions.
- **Mandatory Caveat:** The 31 features must be presented as a *pre-registered baseline representation based on domain heuristics*, not an optimal feature space.

---

## 3. Summary Table of Epistemological Status

| Research Dimension | Current Status | Limiting Factor | Safe Academic Claim |
| :--- | :--- | :--- | :--- |
| **Model Comparison (RF vs LSTM vs TF)** | **VERIFIED** | None (Confirmatory data) | RF is competitive with or superior to deep sequence models on this dataset. |
| **Operational Gate Failure** | **VERIFIED** | None (Confirmatory data) | All models produce unacceptable false-alarm rates under uniform argmax. |
| **Post-Failure Censoring Violation** | **VERIFIED** | None (Raw event audit) | 149 post-failure windows were included due to a labeling generator bug. |
| **Verdict Invariance to Censoring** | **VERIFIED** | None (Counterfactual math) | Gate failures are robust across all censoring corrections. |
| **Physical Root Causes of Failures** | **UNVERIFIABLE** | Trace lacks kernel/OS logs | Categorical scheduler termination only; physical cause unknown. |
| **Causality of CPI Missingness** | **UNVERIFIABLE** | Trace lacks collector logs | Observational correlation only; not a proven causal mechanism. |
| **Early Degradation > 30 min** | **UNVERIFIABLE** | Trace lacks health signals | Unverified observational hypothesis; formally counted as false alarms. |
| **Modern Cloud Generalization** | **UNVERIFIABLE** | 2011 batch trace data | Scoped strictly to 2011 Borg-style multi-tenant batch workloads. |
