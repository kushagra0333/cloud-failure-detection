# Phase 7W Design: Governance Correction, External Dataset Sourcing & Compatibility Assessment

## 1. Scientific & Governance Context
Phase 7W addresses two critical responsibilities:
1. **Governance Correction**: Forensically investigates the Phase 7V execution history, documents the `pd.read_csv` metadata read discrepancy on Phase 6 confirmatory holdouts, and qualifies the prior governance status without altering sealed Phase 7V artifacts.
2. **External Dataset Assessment**: Evaluates four public external workload datasets ([Google Borg 2019](https://github.com/google/cluster-data), [ClusterWise Summit](https://huggingface.co/datasets/MachaParfait/ClusterWise), [Alibaba 2018](https://github.com/alibaba/clusterdata), and [LANL CFDR](https://www.usenix.org/cfdr-data)) for suitability across two distinct validation regimes:
   - **C1. Frozen-Model External Evaluation**: Testing existing frozen models without modification.
   - **C2. New External Benchmark**: Designing a newly trained or adapted failure-prediction task on a distinct cluster.

## 2. Hard Governance Restrictions
- **Zero Access to Prohibited Partitions**: Candidate A Test and Phase 6 Confirmatory Holdouts must never be opened, read, queried, or evaluated.
- **Zero Model Training / Tuning**: No neural network training, threshold optimization, or fuzzy rule changes.
- **No Conflation of Validation Types**: Frozen evaluation (C1) and adapted benchmark (C2) must remain strictly decoupled.

## 3. Decision Framework
- If no dataset can support C1, determine whether C2 is viable.
- If C1 is unviable and C2 is viable, declare `NEW_BENCHMARK_REQUIRED` and specify the exact data-sourcing protocol.
