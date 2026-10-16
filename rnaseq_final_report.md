# Integrated Bulk RNA-seq Analysis

## Executive Summary

This analysis integrates differential expression, enrichment, network prioritization, LLM biological triage, and literature review for **airway**.
The RNA-seq analysis included **8 matched samples**.
The differential-expression model tested **15,926 genes** and identified **635 significant genes**.
LLM triage produced **6 structured biological claims**.
The literature stage produced **100 evidence assessments**.

### PI-facing Biological Synthesis

- Dex activates a canonical glucocorticoid receptor–responsive transcriptional program in HASM at 18 h, consistent with primary GR engagement but requiring GR-dependency tests to confirm.
- Extracellular matrix and integrin remodeling signatures (collagens, laminins, integrins, THBS1) are enriched and are likely downstream/reactive to glucocorticoid exposure and/or TGFβ cross-talk rather than primary GR drive.
- Cytokine/cytokine-receptor interaction modules (e.g., TSLP, LIF, CXCL12, TGFB2, TNFSF family) indicate inflammatory pathway modulation downstream of GR, but directionality and mechanism remain to be resolved.
- FoxO/growth-arrest and apoptosis-linked transcripts (FOXO1/3, CDKN1A/2B, PLK2, TNFSF10) suggest a growth-suppressive state that could be downstream of dex or a stress/growth confounder.
- Circadian-associated transcripts (PER1/2/3, DBP, KLF9) suggest dex-driven clock re-entrainment in HASM, likely as a secondary GR effect.
- Angiogenesis-associated transcripts (VEGFA, THBS1, NRP2, ANG, ADM2) are present, suggesting paracrine vascular signaling changes in HASM, but the signal could reflect culture hypoxia or non-GR factors.

## Experimental Context

- **Dataset:** airway
- **Organism:** human
- **Assay:** bulk RNA-seq
- **Study context:** Dexamethasone-treated versus untreated human airway smooth muscle cells
- **Cell/tissue context:** airway smooth muscle cells
- **Perturbation/treatment:** dexamethasone treatment

## Differential Expression

| Gene | log2FC | adjusted p |
|---|---:|---:|
| SPARCL1 | 4.575 | 0.000191 |
| PER1 | 3.185 | 0.000191 |
| MT2A | 2.199 | 0.000191 |
| STOM | 1.440 | 0.000191 |
| PHC2 | 1.381 | 0.000191 |
| SAMHD1 | 3.780 | 0.000226 |
| MAOA | 3.334 | 0.000226 |
| DUSP1 | 2.951 | 0.000226 |
| KCTD12 | -2.531 | 0.000226 |
| SLC6A9 | -2.215 | 0.000226 |
| NEXN | 2.035 | 0.000226 |
| DNAJB4 | 1.509 | 0.000226 |
| ABAT | 1.149 | 0.000226 |
| ALOX15B | 8.025 | 0.000262 |
| FAM171B | -1.357 | 0.000262 |

## Enrichment / Biological Programs

| Term | adjusted p | Source |
|---|---:|---|
| cell periphery | 1.01e-18 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |
| multicellular organismal process | 2.52e-13 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |
| plasma membrane | 3.15e-13 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |
| regulation of multicellular organismal process | 5.02e-13 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |
| positive regulation of multicellular organismal process | 2.38e-12 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |
| cell communication | 2.8e-12 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |
| signaling | 9.13e-12 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |
| response to stimulus | 2.77e-11 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |
| signal transduction | 3.35e-10 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |
| anatomical structure morphogenesis | 5.81e-10 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |
| multicellular organism development | 7.53e-10 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |
| anatomical structure development | 1.64e-09 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |
| developmental process | 3.97e-09 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |
| positive regulation of developmental process | 4.84e-09 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |
| regulation of developmental process | 8.44e-09 | g:Profiler GO:BP,GO:MF,GO:CC,KEGG,REAC |

## LLM Triage

- Dex activates a canonical glucocorticoid receptor–responsive transcriptional program in HASM at 18 h, consistent with primary GR engagement but requiring GR-dependency tests to confirm.
- Extracellular matrix and integrin remodeling signatures (collagens, laminins, integrins, THBS1) are enriched and are likely downstream/reactive to glucocorticoid exposure and/or TGFβ cross-talk rather than primary GR drive.
- Cytokine/cytokine-receptor interaction modules (e.g., TSLP, LIF, CXCL12, TGFB2, TNFSF family) indicate inflammatory pathway modulation downstream of GR, but directionality and mechanism remain to be resolved.
- FoxO/growth-arrest and apoptosis-linked transcripts (FOXO1/3, CDKN1A/2B, PLK2, TNFSF10) suggest a growth-suppressive state that could be downstream of dex or a stress/growth confounder.
- Circadian-associated transcripts (PER1/2/3, DBP, KLF9) suggest dex-driven clock re-entrainment in HASM, likely as a secondary GR effect.
- Angiogenesis-associated transcripts (VEGFA, THBS1, NRP2, ANG, ADM2) are present, suggesting paracrine vascular signaling changes in HASM, but the signal could reflect culture hypoxia or non-GR factors.

## Network Prioritization

| Candidate | Score |
|---|---:|
| response to stimulus | 0.73008126 |
| biological regulation | 0.71314032 |
| regulation of biological process | 0.68852531 |
| multicellular organismal process | 0.67184238 |
| cellular response to stimulus | 0.65006541 |
| cellular_component | 0.62980705 |
| cellular anatomical structure | 0.60198629 |
| cell communication | 0.57533787 |
| signaling | 0.55679658 |
| anatomical structure development | 0.54469302 |

## Literature Evidence

### Dex activates a canonical glucocorticoid receptor–responsive transcriptional program in HASM at 18 h, consistent with primary GR engagement but requiring GR-dependency tests to confirm.

*Literature target: `finding_001`*

**Evidence screen:** 5 papers (**exclude**: 5)

- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **CPA3 drives the development of asthma by promoting the activation of the JAK/STAT6 pathway.** — 2025; PMID 41437357
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **PANoptosis, an indicator of COVID-19 severity and outcomes.** — 2024; PMID 38555477
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **Differential transcript usage analysis of bulk and single-cell RNA-seq data with DTUrtle** — 2021; Bioinformatics
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.

### Extracellular matrix and integrin remodeling signatures (collagens, laminins, integrins, THBS1) are enriched and are likely downstream/reactive to glucocorticoid exposure and/or TGFβ cross-talk rather than primary GR drive.

*Literature target: `finding_002`*

**Evidence screen:** 5 papers (**contextual_support**: 1, **exclude**: 4)

- **RNA-Seq transcriptome profiling identifies CRISPLD2 as a glucocorticoid responsive gene that modulates cytokine function in airway smooth muscle cells.** — 2014; PloS one; PMID 24926665
  - Evidence level: **contextual_support**; relevance score: **5.00**
  - Paper matches the assay and experimental context, but does not satisfy the stricter target/pathway rule for direct or mechanistic evidence.
- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **CPA3 drives the development of asthma by promoting the activation of the JAK/STAT6 pathway.** — 2025; PMID 41437357
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **PANoptosis, an indicator of COVID-19 severity and outcomes.** — 2024; PMID 38555477
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.

### Cytokine/cytokine-receptor interaction modules (e.g., TSLP, LIF, CXCL12, TGFB2, TNFSF family) indicate inflammatory pathway modulation downstream of GR, but directionality and mechanism remain to be resolved.

*Literature target: `finding_003`*

**Evidence screen:** 5 papers (**exclude**: 5)

- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **CPA3 drives the development of asthma by promoting the activation of the JAK/STAT6 pathway.** — 2025; PMID 41437357
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **PANoptosis, an indicator of COVID-19 severity and outcomes.** — 2024; PMID 38555477
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **Differential transcript usage analysis of bulk and single-cell RNA-seq data with DTUrtle** — 2021; Bioinformatics
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.

### FoxO/growth-arrest and apoptosis-linked transcripts (FOXO1/3, CDKN1A/2B, PLK2, TNFSF10) suggest a growth-suppressive state that could be downstream of dex or a stress/growth confounder.

*Literature target: `finding_004`*

**Evidence screen:** 5 papers (**contextual_support**: 1, **exclude**: 4)

- **RNA-Seq transcriptome profiling identifies CRISPLD2 as a glucocorticoid responsive gene that modulates cytokine function in airway smooth muscle cells.** — 2014; PloS one; PMID 24926665
  - Evidence level: **contextual_support**; relevance score: **5.00**
  - Paper matches the assay and experimental context, but does not satisfy the stricter target/pathway rule for direct or mechanistic evidence.
- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **CPA3 drives the development of asthma by promoting the activation of the JAK/STAT6 pathway.** — 2025; PMID 41437357
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **PANoptosis, an indicator of COVID-19 severity and outcomes.** — 2024; PMID 38555477
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.

### Circadian-associated transcripts (PER1/2/3, DBP, KLF9) suggest dex-driven clock re-entrainment in HASM, likely as a secondary GR effect.

*Literature target: `finding_005`*

**Evidence screen:** 5 papers (**contextual_support**: 1, **exclude**: 4)

- **RNA-Seq transcriptome profiling identifies CRISPLD2 as a glucocorticoid responsive gene that modulates cytokine function in airway smooth muscle cells.** — 2014; PloS one; PMID 24926665
  - Evidence level: **contextual_support**; relevance score: **5.00**
  - Paper matches the assay and experimental context, but does not satisfy the stricter target/pathway rule for direct or mechanistic evidence.
- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **CPA3 drives the development of asthma by promoting the activation of the JAK/STAT6 pathway.** — 2025; PMID 41437357
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **PANoptosis, an indicator of COVID-19 severity and outcomes.** — 2024; PMID 38555477
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.

### Angiogenesis-associated transcripts (VEGFA, THBS1, NRP2, ANG, ADM2) are present, suggesting paracrine vascular signaling changes in HASM, but the signal could reflect culture hypoxia or non-GR factors.

*Literature target: `finding_006`*

**Evidence screen:** 5 papers (**exclude**: 5)

- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **CPA3 drives the development of asthma by promoting the activation of the JAK/STAT6 pathway.** — 2025; PMID 41437357
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **PANoptosis, an indicator of COVID-19 severity and outcomes.** — 2024; PMID 38555477
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **Differential transcript usage analysis of bulk and single-cell RNA-seq data with DTUrtle** — 2021; Bioinformatics
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.

### Bulk RNA-seq result selected for literature review.

*Literature target: `finding_007`*

**Evidence screen:** 5 papers (**exclude**: 5)

- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **CPA3 drives the development of asthma by promoting the activation of the JAK/STAT6 pathway.** — 2025; PMID 41437357
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **PANoptosis, an indicator of COVID-19 severity and outcomes.** — 2024; PMID 38555477
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **Differential transcript usage analysis of bulk and single-cell RNA-seq data with DTUrtle** — 2021; Bioinformatics
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.

### Bulk RNA-seq result selected for literature review.

*Literature target: `finding_008`*

**Evidence screen:** 5 papers (**contextual_support**: 1, **exclude**: 4)

- **RNA-Seq transcriptome profiling identifies CRISPLD2 as a glucocorticoid responsive gene that modulates cytokine function in airway smooth muscle cells.** — 2014; PloS one; PMID 24926665
  - Evidence level: **contextual_support**; relevance score: **5.00**
  - Paper matches the assay and experimental context, but does not satisfy the stricter target/pathway rule for direct or mechanistic evidence.
- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **CPA3 drives the development of asthma by promoting the activation of the JAK/STAT6 pathway.** — 2025; PMID 41437357
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **PANoptosis, an indicator of COVID-19 severity and outcomes.** — 2024; PMID 38555477
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.

### Bulk RNA-seq result selected for literature review.

*Literature target: `finding_009`*

**Evidence screen:** 5 papers (**exclude**: 5)

- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **CPA3 drives the development of asthma by promoting the activation of the JAK/STAT6 pathway.** — 2025; PMID 41437357
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **PANoptosis, an indicator of COVID-19 severity and outcomes.** — 2024; PMID 38555477
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **Differential transcript usage analysis of bulk and single-cell RNA-seq data with DTUrtle** — 2021; Bioinformatics
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.

### Bulk RNA-seq result selected for literature review.

*Literature target: `finding_010`*

**Evidence screen:** 5 papers (**exclude**: 5)

- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **CPA3 drives the development of asthma by promoting the activation of the JAK/STAT6 pathway.** — 2025; PMID 41437357
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **PANoptosis, an indicator of COVID-19 severity and outcomes.** — 2024; PMID 38555477
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **Differential transcript usage analysis of bulk and single-cell RNA-seq data with DTUrtle** — 2021; Bioinformatics
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.

### SPARCL1 identified in bulk RNA-seq differential expression.

*Literature target: `finding_011`*

**Evidence screen:** 5 papers (**contextual_support**: 1, **direct**: 1, **exclude**: 3)

- **Metformin protects against myocardial ischemia/reperfusion injury and suppresses the Nur77-SPARCL1 inflammatory axis.** — 2026; PMID 42722063
  - Evidence level: **direct**; relevance score: **7.00**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **RNA-Seq transcriptome profiling identifies CRISPLD2 as a glucocorticoid responsive gene that modulates cytokine function in airway smooth muscle cells.** — 2014; PloS one; PMID 24926665
  - Evidence level: **contextual_support**; relevance score: **5.00**
  - Paper matches the assay and experimental context, but does not satisfy the stricter target/pathway rule for direct or mechanistic evidence.
- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Single-Cell Analysis of Human Heart Failure With Preserved Ejection Fraction.** — 2026; PMID 42137938
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.

### PER1 identified in bulk RNA-seq differential expression.

*Literature target: `finding_012`*

**Evidence screen:** 5 papers (**direct**: 3, **exclude**: 2)

- **WITHDRAWN: Circadian Clock-Gene Structure Couples with PD-L1 Across Six Cancer Types and Defines Survival-Associated Subtypes in Melanoma** — 2026
  - Evidence level: **direct**; relevance score: **7.50**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **Neuroinflammatory suppression of astrocytic BMAL1 defines a selective reactive astrocyte program.** — 2026; PMID 42421063
  - Evidence level: **direct**; relevance score: **7.00**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **Immune-stromal dysregulation and senescence in chronic spontaneous urticaria.** — 2026; PMID 42436760
  - Evidence level: **direct**; relevance score: **7.00**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.

### MT2A identified in bulk RNA-seq differential expression.

*Literature target: `finding_013`*

**Evidence screen:** 5 papers (**direct**: 2, **exclude**: 3)

- **"Eye-Conic" Spatial Transcriptomics Reveals the Layer-Specific Molecular Alterations in Corneas of Patients With Keratoconus.** — 2026; PMID 41931003
  - Evidence level: **direct**; relevance score: **7.50**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **Integrative multi-omics analysis reveals an 11-gene malignant-myeloid interaction signature and identifies TPST1 as a potential regulator of immunosuppressive microenvironment in glioma.** — 2026; PMID 41838327
  - Evidence level: **direct**; relevance score: **7.00**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **CPA3 drives the development of asthma by promoting the activation of the JAK/STAT6 pathway.** — 2025; PMID 41437357
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.

### STOM identified in bulk RNA-seq differential expression.

*Literature target: `finding_014`*

**Evidence screen:** 5 papers (**contextual_support**: 1, **exclude**: 4)

- **RNA-Seq transcriptome profiling identifies CRISPLD2 as a glucocorticoid responsive gene that modulates cytokine function in airway smooth muscle cells.** — 2014; PloS one; PMID 24926665
  - Evidence level: **contextual_support**; relevance score: **5.00**
  - Paper matches the assay and experimental context, but does not satisfy the stricter target/pathway rule for direct or mechanistic evidence.
- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **CPA3 drives the development of asthma by promoting the activation of the JAK/STAT6 pathway.** — 2025; PMID 41437357
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.
- **PANoptosis, an indicator of COVID-19 severity and outcomes.** — 2024; PMID 38555477
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.

### PHC2 identified in bulk RNA-seq differential expression.

*Literature target: `finding_015`*

**Evidence screen:** 5 papers (**contextual_support**: 1, **direct**: 2, **exclude**: 2)

- **PHC2 as a novel predictive biomarker and therapeutic target for cisplatin resistance in lung adenocarcinoma.** — 2026; PMID 41925952
  - Evidence level: **direct**; relevance score: **7.50**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **PHC2 promotes hepatocellular carcinoma progression and serves as a robust prognostic biomarker: A pancancer multiomics and clinical validation study.** — 2026; PMID 42025893
  - Evidence level: **direct**; relevance score: **7.00**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **RNA-Seq transcriptome profiling identifies CRISPLD2 as a glucocorticoid responsive gene that modulates cytokine function in airway smooth muscle cells.** — 2014; PloS one; PMID 24926665
  - Evidence level: **contextual_support**; relevance score: **5.00**
  - Paper matches the assay and experimental context, but does not satisfy the stricter target/pathway rule for direct or mechanistic evidence.
- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.

### SAMHD1 identified in bulk RNA-seq differential expression.

*Literature target: `finding_016`*

**Evidence screen:** 5 papers (**contextual_support**: 1, **direct**: 2, **exclude**: 2)

- **Inflammation-driven prognostic model and immune landscape profiling in osteosarcoma.** — 2025; PMID 41201557
  - Evidence level: **direct**; relevance score: **7.00**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **Entosis remodels the immune microenvironment of osteosarcoma by regulating macrophage polarization: An integrated study based on multi-omics data and in vitro experiments.** — 2026; PMID 42407378
  - Evidence level: **direct**; relevance score: **7.00**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **RNA-Seq transcriptome profiling identifies CRISPLD2 as a glucocorticoid responsive gene that modulates cytokine function in airway smooth muscle cells.** — 2014; PloS one; PMID 24926665
  - Evidence level: **contextual_support**; relevance score: **5.00**
  - Paper matches the assay and experimental context, but does not satisfy the stricter target/pathway rule for direct or mechanistic evidence.
- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.

### MAOA identified in bulk RNA-seq differential expression.

*Literature target: `finding_017`*

**Evidence screen:** 5 papers (**direct**: 1, **exclude**: 4)

- **Integrated Bulk and Single-Cell Transcriptomic Analysis Reveals Xenobiotic Metabolism Genes Drive Progression From Liver Cirrhosis to Hepatocellular Carcinoma.** — 2026; PMID 42136961
  - Evidence level: **direct**; relevance score: **7.00**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **A single-cell atlas revealing cellular heterogeneity across healthy and diseased human thymus.** — 2026; PMID 42091897
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **CPA3 drives the development of asthma by promoting the activation of the JAK/STAT6 pathway.** — 2025; PMID 41437357
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.

### DUSP1 identified in bulk RNA-seq differential expression.

*Literature target: `finding_018`*

**Evidence screen:** 5 papers (**contextual_support**: 1, **direct**: 2, **exclude**: 2)

- **Characterizing the Transcriptional Features that Underlie Established Injury and Disease Progression in Diabetic Kidney Disease.** — 2026; PMID 42747892
  - Evidence level: **direct**; relevance score: **7.50**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **DUSP1 is a key driver of disease persistence and potential therapeutic target in hairy cell leukemia.** — 2026; PMID 42200459
  - Evidence level: **direct**; relevance score: **7.00**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **RNA-Seq transcriptome profiling identifies CRISPLD2 as a glucocorticoid responsive gene that modulates cytokine function in airway smooth muscle cells.** — 2014; PloS one; PMID 24926665
  - Evidence level: **contextual_support**; relevance score: **5.00**
  - Paper matches the assay and experimental context, but does not satisfy the stricter target/pathway rule for direct or mechanistic evidence.
- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.

### KCTD12 identified in bulk RNA-seq differential expression.

*Literature target: `finding_019`*

**Evidence screen:** 5 papers (**contextual_support**: 1, **exclude**: 4)

- **RNA-Seq transcriptome profiling identifies CRISPLD2 as a glucocorticoid responsive gene that modulates cytokine function in airway smooth muscle cells.** — 2014; PloS one; PMID 24926665
  - Evidence level: **contextual_support**; relevance score: **5.00**
  - Paper matches the assay and experimental context, but does not satisfy the stricter target/pathway rule for direct or mechanistic evidence.
- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Depletion of P2X4 receptor alleviates prostate cancer bone metastasis through reduced cancer cell invasiveness and enhanced cell adhesion activities.** — 2025; PMID 40515881
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **CPA3 drives the development of asthma by promoting the activation of the JAK/STAT6 pathway.** — 2025; PMID 41437357
  - Evidence level: **exclude**; relevance score: **4.00**
  - Insufficient match to the current finding for evidence use.

### SLC6A9 identified in bulk RNA-seq differential expression.

*Literature target: `finding_020`*

**Evidence screen:** 5 papers (**contextual_support**: 1, **direct**: 2, **exclude**: 2)

- **GlyT1 inhibition by ALX-5407 attenuates allograft rejection through suppression of Th1 cell differentiation.** — 2025; PMID 41063979
  - Evidence level: **direct**; relevance score: **7.00**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **Slc6a9 is distributed in glial cells and neurons across several nervous system regions, whereas Slc6a5 is more restricted to neurons in the caudal brain.** — 2026; PMID 41703440
  - Evidence level: **direct**; relevance score: **7.00**
  - Same or equivalent assay detected.
  - Paper examines the same target/question or named gene(s).
  - Caution: Direct by assay/biology rule, but experimental context differs or is not explicit.
- **RNA-Seq transcriptome profiling identifies CRISPLD2 as a glucocorticoid responsive gene that modulates cytokine function in airway smooth muscle cells.** — 2014; PloS one; PMID 24926665
  - Evidence level: **contextual_support**; relevance score: **5.00**
  - Paper matches the assay and experimental context, but does not satisfy the stricter target/pathway rule for direct or mechanistic evidence.
- **Convection-enhanced delivery of dexamethasone in glioma models suppresses myeloid inflammation while avoiding systemic toxicities.** — 2026; PMID 42611562
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.
- **Disrupted glucocorticoid receptor cell signalling causes a ciliogenesis defect in the fetal mouse renal tubule.** — 2025; PMID 40247090
  - Evidence level: **exclude**; relevance score: **4.50**
  - Insufficient match to the current finding for evidence use.


## Interpretation Notes

- Differential-expression statistics are the primary quantitative evidence from the RNA-seq experiment.
- Enrichment and network outputs prioritize programs and candidates; they do not independently establish causality.
- LLM triage is an interpretation layer and should be checked against the numerical source tables.
- Literature retrieval contextualizes the experiment; primary papers should be reviewed directly before publication or experimental decisions.

## Supporting Material

- `01_rnaseq/` — bulk RNA-seq DE, enrichment, classifier, QC, and plots
- `02_network/` — network and candidate prioritization
- `03_llm_triage/` — LLM claims, programs, and report
- `04_literature_review/` — literature request and evidence response

Numerical source tables remain the authoritative record for individual statistics.
