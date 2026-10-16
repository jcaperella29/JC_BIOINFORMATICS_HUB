# Integrated scRNA-seq Analysis

## Executive Summary

This analysis integrates differential expression, enrichment, cell-state composition, network prioritization, LLM biological triage, and literature review for **Kang18_8vs8**.
The dataset contains **24,673 cells** and **35,635 genes**.
The largest cell-state composition differences were CD14+ Monocytes (-1.58 percentage points), NK cells (+1.28 percentage points), CD8 T cells (-1.02 percentage points).
LLM triage produced **6 structured biological claims**.
The literature stage produced **31 evidence assessments**.

### PI-facing Biological Synthesis

- Type I interferon/JAK–STAT antiviral transcriptional program is induced within CD14+ monocytes after ~6 h IFN‑β and is the most plausible primary driver of the observed RNA changes.
- NFκB/inflammatory signaling signatures are present but more consistent with a reactive or parallel response rather than the primary driver in IFN‑β–stimulated monocytes.
- ER stress/UPR and apoptosis/DNA‑damage response signatures are detectable at low–moderate levels and are more consistent with stress‑reactive biology than with causation.
- Enrichment for interferon‑β production/regulation terms may reflect induction of upstream regulators or feedback loops but does not by itself demonstrate endogenous IFN‑β production in monocytes under exogenous IFN‑β.
- A modest Interferon‑γ signaling signature is detected but is likely due to gene‑set overlap with type I IFN responses rather than true IFN‑γ pathway activation in this assay.
- Mild cell‑cycle/proliferation signals are likely artifacts/confounders in this acute immune‑stimulation context.

## Experimental Context

- **Dataset:** Kang18_8vs8
- **Organism:** Homo sapiens
- **Assay:** single-cell RNA-seq with donor-level pseudobulk differential expression
- **Study context:** PBMC interferon-beta stimulation dataset

## Cell-State / Lineage Context

Condition-aware composition is shown below so transcriptional changes can be interpreted alongside possible cell-state shifts.

| State | ctrl n | ctrl % | stim n | stim % | Δ percentage points |
|---|---:|---:|---:|---:|---:|
| CD4 T cells | 5161 | 41.9% | 5228 | 42.3% | +0.40 |
| CD14+ Monocytes | 2785 | 22.6% | 2600 | 21.0% | -1.58 |
| B cells | 1307 | 10.6% | 1266 | 10.2% | -0.37 |
| CD8 T cells | 1082 | 8.8% | 960 | 7.8% | -1.02 |
| NK cells | 916 | 7.4% | 1077 | 8.7% | +1.28 |
| FCGR3A+ Monocytes | 739 | 6.0% | 860 | 7.0% | +0.96 |
| Dendritic cells | 203 | 1.6% | 229 | 1.9% | +0.20 |
| Megakaryocytes | 122 | 1.0% | 138 | 1.1% | +0.13 |

## Differential Expression Across Conditions

| Gene | log2FC | adjusted p |
|---|---:|---:|
| ISG20 | 3.034 | 8.07e-18 |
| IFI6 | 5.663 | 9.59e-18 |
| MT2A | 4.258 | 2.36e-17 |
| IFI35 | 3.675 | 1.09e-16 |
| HERC5 | 4.194 | 1.09e-16 |
| IRF7 | 3.432 | 1.48e-16 |
| UBE2L6 | 2.623 | 3.22e-16 |
| TNFSF10 | 4.392 | 3.89e-16 |
| MX1 | 4.780 | 3.89e-16 |
| LY6E | 4.124 | 1.01e-15 |
| ISG15 | 5.121 | 1.7e-15 |
| NMI | 2.271 | 1.99e-15 |

## Cross-Condition Biological Themes

- **INNATE_IFN_ANTIVIRAL**
- **OTHER**
- **INFLAMMATION_NFKB**
- **ADAPTIVE_T_CELL**
- **MITO_OXPHOS**
- **MAPK_STRESS_KINASE_ATTENUATION**
- **DDR_P53_APOPTOSIS**
- **UPR_PROTEOSTASIS**
- **CELL_CYCLE_PROLIFERATION**

## Condition-Level Findings

### Condition A vs Condition B

#### Biological interpretation

- Type I interferon/JAK–STAT antiviral transcriptional program is induced within CD14+ monocytes after ~6 h IFN‑β and is the most plausible primary driver of the observed RNA changes.
- NFκB/inflammatory signaling signatures are present but more consistent with a reactive or parallel response rather than the primary driver in IFN‑β–stimulated monocytes.
- ER stress/UPR and apoptosis/DNA‑damage response signatures are detectable at low–moderate levels and are more consistent with stress‑reactive biology than with causation.
- Enrichment for interferon‑β production/regulation terms may reflect induction of upstream regulators or feedback loops but does not by itself demonstrate endogenous IFN‑β production in monocytes under exogenous IFN‑β.
- A modest Interferon‑γ signaling signature is detected but is likely due to gene‑set overlap with type I IFN responses rather than true IFN‑γ pathway activation in this assay.
- Mild cell‑cycle/proliferation signals are likely artifacts/confounders in this acute immune‑stimulation context.

#### Key supporting terms/genes

- **Leading DE genes:** ISG20, IFI6, MT2A, IFI35, HERC5, IRF7, UBE2L6, TNFSF10, MX1, LY6E
- **Programs/pathways:** INNATE_IFN_ANTIVIRAL, OTHER, INFLAMMATION_NFKB, ADAPTIVE_T_CELL, MITO_OXPHOS, MAPK_STRESS_KINASE_ATTENUATION, DDR_P53_APOPTOSIS, UPR_PROTEOSTASIS, CELL_CYCLE_PROLIFERATION

#### Top confounders to monitor

- Cell-state composition differences can contribute to apparent pathway changes.
- Cell-level DE is exploratory for biological inference; replicate-aware pseudobulk is preferable for formal inference.
- Enrichment and network ranking prioritize hypotheses rather than establish direct causality.

## Network Prioritization

| Candidate | Score |
|---|---:|
| response to stress (GO:BP) | 0.50103348 |
| positive regulation of biological process (GO:BP) | 0.47387139 |
| positive regulation of cellular process (GO:BP) | 0.40808601 |
| regulation of response to stress (GO:BP) | 0.35253519 |
| regulation of response to stimulus (GO:BP) | 0.33924285 |
| immune system process (GO:BP) | 0.25957656 |
| positive regulation of response to stimulus (GO:BP) | 0.24875328 |
| immune response (GO:BP) | 0.24566606 |
| cellular response to stimulus (GO:BP) | 0.14988712 |
| biological process involved in interspecies interaction between organisms (GO:BP) | 0.11856911 |

## Literature Evidence

### Type I interferon/JAK–STAT antiviral transcriptional program is induced within CD14+ monocytes after ~6 h IFN‑β and is the most plausible primary driver of the observed RNA changes.

*Literature target: `finding_001`*

**Evidence screen:** 1 paper (**exclude**: 1)

- **Early treatment with reverse transcriptase inhibitors significantly suppresses peak plasma IFNα in vivo during acute simian immunodeficiency virus infection.** — 2016; Cellular immunology; PMID 27622386
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.

### NFκB/inflammatory signaling signatures are present but more consistent with a reactive or parallel response rather than the primary driver in IFN‑β–stimulated monocytes.

*Literature target: `finding_002`*

**Evidence screen:** 2 papers (**exclude**: 2)

- **Effects of acute heat exposure on monocyte phenotype following exhaustive exercise: An ex vivo study.** — 2025; Cytokine; PMID 40543235
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **A BET Protein Inhibitor Targeting Mononuclear Myeloid Cells Affects Specific Inflammatory Mediators and Pathways in Crohn's Disease.** — 2022; Cells; PMID 36139421
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.

### A modest Interferon‑γ signaling signature is detected but is likely due to gene‑set overlap with type I IFN responses rather than true IFN‑γ pathway activation in this assay.

*Literature target: `finding_005`*

**Evidence screen:** 5 papers (**exclude**: 5)

- **Tumor-intrinsic IFNγ signaling and niche adaptation drive early colonization in ovarian cancer metastasis.** — 2026; Nature communications; PMID 42834039
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **The Polycystic Kidney Disease Cyst Transcriptome Defined by Integrating Spatial, Single Nuclear, and Bulk Transcriptomics.** — 2026; Kidney360; PMID 42804341
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **LncRNA Gas5, a Target of the Nonsense-Mediated Decay Pathway in the Brain, Regulates Neuroinflammation and Neurodegenerative Disease Pathways in a Tauopathy Mouse Model.** — 2026; International journal of molecular sciences; PMID 42737576
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **RSAD2-CMPK2 Signaling Mediates PVT1 Exon 9 Action in an AR-Independent Manner in PVT1 Exon 9 Overexpressing Neuroendocrine Prostate Cancers**
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **Transcriptomic Characterization of Hypermutated Microsatellite Stable Tumors Reveals Distinct Proliferative and Immune Signatures** — 2026
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.

### Mild cell‑cycle/proliferation signals are likely artifacts/confounders in this acute immune‑stimulation context.

*Literature target: `finding_006`*

**Evidence screen:** 5 papers (**exclude**: 5)

- **PDE4DIP-Derived MMG8 Supports Proliferation, Migration, and Tumor Growth in Hepatocellular Carcinoma Models.** — 2026; Frontiers in bioscience (Landmark edition); PMID 42689991
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **FUS/EWSR1::TFCP2-rearranged bone and soft tissue tumors: evidence of gene promiscuity.** — 2026; Virchows Archiv : an international journal of pathology; PMID 42678417
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **SPEN inactivation drives resistance to androgen receptor pathway inhibitors in metastatic prostate cancer.** — 2026; Clinical cancer research : an official journal of the American Association for Cancer Research; PMID 42594031
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **Integrated Bioinformatics Analysis Identifies NCAPG2 as an Immune-Related Prognostic Biomarker in Breast Cancer.** — 2026; World journal of oncology; PMID 42490819
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **Molecular insights into induction of apoptosis and regulation of cell cycle proliferation by cycloartane triterpenoids from Dysoxylum malabaricum in breast cancer cells.** — 2026; Journal of molecular histology; PMID 42165934
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.

### Literature-review target: INNATE_IFN_ANTIVIRAL

*Literature target: `finding_007`*

**Evidence screen:** 1 paper (**exclude**: 1)

- **Early treatment with reverse transcriptase inhibitors significantly suppresses peak plasma IFNα in vivo during acute simian immunodeficiency virus infection.** — 2016; Cellular immunology; PMID 27622386
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.

### Literature-review target: OTHER

*Literature target: `finding_008`*

**Evidence screen:** 5 papers (**contextual_support**: 5)

- **2026 iCARDIO Alliance Global Implementation Guidelines on Heart Failure.** — 2027; PMID 42822640
  - Evidence level: **contextual_support**; relevance score: **2.50**
  - Biologically related evidence was found, but it does not satisfy the direct-evidence assay/causal-distance rule.
- **Mortality and disability after hospitalisation for injuries in low-income and middle-income countries: a cohort study.** — 2027; PMID 42843391
  - Evidence level: **contextual_support**; relevance score: **2.50**
  - Biologically related evidence was found, but it does not satisfy the direct-evidence assay/causal-distance rule.
- **Effects of a 3-day simulated heatwave on physiological and functional outcomes in older adults in Canada: a single-site randomised crossover trial.** — 2027; PMID 42815509
  - Evidence level: **contextual_support**; relevance score: **2.50**
  - Biologically related evidence was found, but it does not satisfy the direct-evidence assay/causal-distance rule.
- **To Whom National Culture Matters: An Analysis of Mentions of Countries in Titles and Abstracts in Social and Personality Psychology and Related Fields From 1990 to 2020.** — 2026; PMID 42825664
  - Evidence level: **contextual_support**; relevance score: **2.50**
  - Biologically related evidence was found, but it does not satisfy the direct-evidence assay/causal-distance rule.
- **ICAM signaling rewires the inflammatory communication network in pediatric gout.** — 2026; PMID 42445190
  - Evidence level: **contextual_support**; relevance score: **2.50**
  - Biologically related evidence was found, but it does not satisfy the direct-evidence assay/causal-distance rule.

### Literature-review target: INFLAMMATION_NFKB

*Literature target: `finding_009`*

**Evidence screen:** 2 papers (**exclude**: 2)

- **Effects of acute heat exposure on monocyte phenotype following exhaustive exercise: An ex vivo study.** — 2025; Cytokine; PMID 40543235
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **A BET Protein Inhibitor Targeting Mononuclear Myeloid Cells Affects Specific Inflammatory Mediators and Pathways in Crohn's Disease.** — 2022; Cells; PMID 36139421
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.

### Literature-review target: ADAPTIVE_T_CELL

*Literature target: `finding_010`*

**Evidence screen:** 5 papers (**exclude**: 5)

- **Dynamic γδ T-cell Receptor remodeling and antigen-induced T-cell IFN-γ responses after SARS-CoV-2 infection during pregnancy: insights for γδ T-cell immunotherapy.** — 2026; Frontiers in immunology; PMID 42807541
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **Microenvironmental and transcriptional determinants of the robustness of spinal cord regeneration in zebrafish.** — 2026; bioRxiv : the preprint server for biology; PMID 42779586
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **Mucosal adjuvant activity of a PAR-2-activating peptide enhances lymph node immune cell recruitment and promotes immune cell activation in a prime-boost influenza vaccination model.** — 2026; Vaccine; PMID 42696827
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **The interplay between innate immune memory and adaptive immunity: Emerging mechanisms in alloimmunity.** — 2026; American journal of transplantation : official journal of the American Society of Transplantation and the American Society of Transplant Surgeons; PMID 42705464
  - Evidence level: **exclude**; relevance score: **-2.00**
  - Insufficient match to the current finding for evidence use.
- **Dendritic cell‑based vaccines for colorectal cancer: Mechanisms, clinical progress and future directions (Review).** — 2026; International journal of oncology; PMID 42695424
  - Evidence level: **exclude**; relevance score: **-2.00**
  - Insufficient match to the current finding for evidence use.

### Literature-review target: CELL_CYCLE_PROLIFERATION

*Literature target: `finding_015`*

**Evidence screen:** 5 papers (**exclude**: 5)

- **PDE4DIP-Derived MMG8 Supports Proliferation, Migration, and Tumor Growth in Hepatocellular Carcinoma Models.** — 2026; Frontiers in bioscience (Landmark edition); PMID 42689991
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **FUS/EWSR1::TFCP2-rearranged bone and soft tissue tumors: evidence of gene promiscuity.** — 2026; Virchows Archiv : an international journal of pathology; PMID 42678417
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **SPEN inactivation drives resistance to androgen receptor pathway inhibitors in metastatic prostate cancer.** — 2026; Clinical cancer research : an official journal of the American Association for Cancer Research; PMID 42594031
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **Integrated Bioinformatics Analysis Identifies NCAPG2 as an Immune-Related Prognostic Biomarker in Breast Cancer.** — 2026; World journal of oncology; PMID 42490819
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.
- **Molecular insights into induction of apoptosis and regulation of cell cycle proliferation by cycloartane triterpenoids from Dysoxylum malabaricum in breast cancer cells.** — 2026; Journal of molecular histology; PMID 42165934
  - Evidence level: **exclude**; relevance score: **0.00**
  - Insufficient match to the current finding for evidence use.


## Supporting Material

- `01_scrna/` — scRNA analysis, DE, enrichment, classifier, PCA/UMAP
- `02_network/` — network and candidate prioritization
- `03_llm_triage/` — LLM claims, programs, and report
- `04_literature_review/` — literature request and evidence response
- `cell_state/` — condition-aware state context

Numerical source tables remain the authoritative record for individual statistics.
