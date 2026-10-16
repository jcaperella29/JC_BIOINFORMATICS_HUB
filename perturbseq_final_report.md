# JCAP Perturb-seq Integrated Final Report

Targets summarized: 6

## Experiment Context

```json
{
  "phenotype": "CRISPR perturbation transcriptional response",
  "organism": "mouse",
  "assay": "Perturb-seq / scRNA-seq",
  "tissue": "",
  "cell_type": "",
  "perturbation": "CRISPR perturbation",
  "timepoint": "",
  "dataset": "Giladi Perturb-seq",
  "control_label": "CTRL",
  "cell_state_analysis": {
    "assay": "perturbseq",
    "status": "ok",
    "annotation_mode": "provided",
    "annotation_source": "metadata:lineage_state",
    "n_cells": 4500,
    "states": [
      "Car1",
      "Cd74",
      "Fcnb",
      "Fcrla",
      "Gstm1",
      "Hba-a2",
      "Ltf",
      "Ly86",
      "Pf4",
      "Prg2",
      "Prss34",
      "Vpreb1",
      "unassigned"
    ],
    "overall_state_summary": [
      {
        "state": "Car1",
        "n_cells": 1786,
        "fraction": 0.3968888888888889
      },
      {
        "state": "Hba-a2",
        "n_cells": 1409,
        "fraction": 0.3131111111111111
      },
      {
        "state": "Gstm1",
        "n_cells": 615,
        "fraction": 0.13666666666666666
      },
      {
        "state": "unassigned",
        "n_cells": 485,
        "fraction": 0.10777777777777778
      },
      {
        "state": "Vpreb1",
        "n_cells": 73,
        "fraction": 0.01622222222222222
      },
      {
        "state": "Fcnb",
        "n_cells": 64,
        "fraction": 0.014222222222222223
      },
      {
        "state": "Ly86",
        "n_cells": 23,
        "fraction": 0.005111111111111111
      },
      {
        "state": "Pf4",
        "n_cells": 12,
        "fraction": 0.0026666666666666666
      },
      {
        "state": "Fcrla",
        "n_cells": 12,
        "fraction": 0.0026666666666666666
      },
      {
        "state": "Prss34",
        "n_cells": 7,
        "fraction": 0.0015555555555555555
      },
      {
        "state": "Ltf",
        "n_cells": 6,
        "fraction": 0.0013333333333333333
      },
      {
        "state": "Prg2",
        "n_cells": 6,
        "fraction": 0.0013333333333333333
      },
      {
        "state": "Cd74",
        "n_cells": 2,
        "fraction": 0.00044444444444444447
      }
    ],
    "group_state_summary": [
      {
        "group": "CTRL",
        "state": "Car1",
        "n_cells": 655,
        "fraction_within_group": 0.43666666666666665
      },
      {
        "group": "CTRL",
        "state": "Hba-a2",
        "n_cells": 532,
        "fraction_within_group": 0.3546666666666667
      },
      {
        "group": "CTRL",
        "state": "Gstm1",
        "n_cells": 163,
        "fraction_within_group": 0.10866666666666666
      },
      {
        "group": "CTRL",
        "state": "unassigned",
        "n_cells": 107,
        "fraction_within_group": 0.07133333333333333
      },
      {
        "group": "CTRL",
        "state": "Fcnb",
        "n_cells": 22,
        "fraction_within_group": 0.014666666666666666
      },
      {
        "group": "CTRL",
        "state": "Vpreb1",
        "n_cells": 11,
        "fraction_within_group": 0.007333333333333333
      },
      {
        "group": "CTRL",
        "state": "Ly86",
        "n_cells": 4,
        "fraction_within_group": 0.0026666666666666666
      },
      {
        "group": "CTRL",
        "state": "Fcrla",
        "n_cells": 2,
        "fraction_within_group": 0.0013333333333333333
      },
      {
        "group": "CTRL",
        "state": "Prss34",
        "n_cells": 2,
        "fraction_within_group": 0.0013333333333333333
      },
      {
        "group": "CTRL",
        "state": "Cd74",
        "n_cells": 1,
        "fraction_within_group": 0.0006666666666666666
      },
      {
        "group": "CTRL",
        "state": "Pf4",
        "n_cells": 1,
        "fraction_within_group": 0.0006666666666666666
      },
      {
        "group": "Cebpa",
        "state": "Car1",
        "n_cells": 239,
        "fraction_within_group": 0.478
      },
      {
        "group": "Cebpa",
        "state": "Hba-a2",
        "n_cells": 232,
        "fraction_within_group": 0.464
      },
      {
        "group": "Cebpa",
        "state": "Gstm1",
        "n_cells": 15,
        "fraction_within_group": 0.03
      },
      {
        "group": "Cebpa",
        "state": "unassigned",
        "n_cells": 10,
        "fraction_within_group": 0.02
      },
      {
        "group": "Cebpa",
        "state": "Vpreb1",
        "n_cells": 2,
        "fraction_within_group": 0.004
      },
      {
        "group": "Cebpa",
        "state": "Ly86",
        "n_cells": 1,
        "fraction_within_group": 0.002
      },
      {
        "group": "Cebpa",
        "state": "Pf4",
        "n_cells": 1,
        "fraction_within_group": 0.002
      },
      {
        "group": "Csf1r",
        "state": "Car1",
        "n_cells": 192,
        "fraction_within_group": 0.384
      },
      {
        "group": "Csf1r",
        "state": "Hba-a2",
        "n_cells": 147,
        "fraction_within_group": 0.294
      },
      {
        "group": "Csf1r",
        "state": "Gstm1",
        "n_cells": 72,
        "fraction_within_group": 0.144
      },
      {
        "group": "Csf1r",
        "state": "unassigned",
        "n_cells": 51,
        "fraction_within_group": 0.102
      },
      {
        "group": "Csf1r",
        "state": "Vpreb1",
        "n_cells": 21,
        "fraction_within_group": 0.042
      },
      {
        "group": "Csf1r",
        "state": "Fcnb",
        "n_cells": 7,
        "fraction_within_group": 0.014
      },
      {
        "group": "Csf1r",
        "state": "Fcrla",
        "n_cells": 3,
        "fraction_within_group": 0.006
      },
      {
        "group": "Csf1r",
        "state": "Ly86",
        "n_cells": 3,
        "fraction_within_group": 0.006
      },
      {
        "group": "Csf1r",
        "state": "Pf4",
        "n_cells": 2,
        "fraction_within_group": 0.004
      },
      {
        "group": "Csf1r",
        "state": "Ltf",
        "n_cells": 1,
        "fraction_within_group": 0.002
      },
      {
        "group": "Csf1r",
        "state": "Prss34",
        "n_cells": 1,
        "fraction_within_group": 0.002
      },
      {
        "group": "Fcgr3",
        "state": "Car1",
        "n_cells": 159,
        "fraction_within_group": 0.318
      },
      {
        "group": "Fcgr3",
        "state": "unassigned",
        "n_cells": 117,
        "fraction_within_group": 0.234
      },
      {
        "group": "Fcgr3",
        "state": "Hba-a2",
        "n_cells": 111,
        "fraction_within_group": 0.222
      },
      {
        "group": "Fcgr3",
        "state": "Gstm1",
        "n_cells": 72,
        "fraction_within_group": 0.144
      },
      {
        "group": "Fcgr3",
        "state": "Vpreb1",
        "n_cells": 14,
        "fraction_within_group": 0.028
      },
      {
        "group": "Fcgr3",
        "state": "Fcnb",
        "n_cells": 9,
        "fraction_within_group": 0.018
      },
      {
        "group": "Fcgr3",
        "state": "Ly86",
        "n_cells": 7,
        "fraction_within_group": 0.014
      },
      {
        "group": "Fcgr3",
        "state": "Pf4",
        "n_cells": 6,
        "fraction_within_group": 0.012
      },
      {
        "group": "Fcgr3",
        "state": "Prg2",
        "n_cells": 1,
        "fraction_within_group": 0.002
      },
      {
        "group": "Fcgr3",
        "state": "Prss34",
        "n_cells": 1,
        "fraction_within_group": 0.002
      },
      {
        "group": "Fcgr3",
        "state": "Cd74",
        "n_cells": 1,
        "fraction_within_group": 0.002
      },
      {
        "group": "Fcgr3",
        "state": "Fcrla",
        "n_cells": 1,
        "fraction_within_group": 0.002
      },
      {
        "group": "Fcgr3",
        "state": "Ltf",
        "n_cells": 1,
        "fraction_within_group": 0.002
      },
      {
        "group": "Flt3",
        "state": "Car1",
        "n_cells": 169,
        "fraction_within_group": 0.338
      },
      {
        "group": "Flt3",
        "state": "Gstm1",
        "n_cells": 115,
        "fraction_within_group": 0.23
      },
      {
        "group": "Flt3",
        "state": "Hba-a2",
        "n_cells": 107,
        "fraction_within_group": 0.214
      },
      {
        "group": "Flt3",
        "state": "unassigned",
        "n_cells": 86,
        "fraction_within_group": 0.172
      },
      {
        "group": "Flt3",
        "state": "Fcnb",
        "n_cells": 7,
        "fraction_within_group": 0.014
      },
      {
        "group": "Flt3",
        "state": "Ly86",
        "n_cells": 5,
        "fraction_within_group": 0.01
      },
      {
        "group": "Flt3",
        "state": "Vpreb1",
        "n_cells": 3,
        "fraction_within_group": 0.006
      },
      {
        "group": "Flt3",
        "state": "Fcrla",
        "n_cells": 3,
        "fraction_within_group": 0.006
      },
      {
        "group": "Flt3",
        "state": "Prss34",
        "n_cells": 2,
        "fraction_within_group": 0.004
      },
      {
        "group": "Flt3",
        "state": "Prg2",
        "n_cells": 1,
        "fraction_within_group": 0.002
      },
      {
        "group": "Flt3",
        "state": "Ltf",
        "n_cells": 1,
        "fraction_within_group": 0.002
      },
      {
        "group": "Flt3",
        "state": "Pf4",
        "n_cells": 1,
        "fraction_within_group": 0.002
      },
      {
        "group": "Irf8",
        "state": "Car1",
        "n_cells": 197,
        "fraction_within_group": 0.394
      },
      {
        "group": "Irf8",
        "state": "Hba-a2",
        "n_cells": 137,
        "fraction_within_group": 0.274
      },
      {
        "group": "Irf8",
        "state": "Gstm1",
        "n_cells": 72,
        "fraction_within_group": 0.144
      },
      {
        "group": "Irf8",
        "state": "unassigned",
        "n_cells": 48,
        "fraction_within_group": 0.096
      },
      {
        "group": "Irf8",
        "state": "Vpreb1",
        "n_cells": 20,
        "fraction_within_group": 0.04
      },
      {
        "group": "Irf8",
        "state": "Fcnb",
        "n_cells": 15,
        "fraction_within_group": 0.03
      },
      {
        "group": "Irf8",
        "state": "Fcrla",
        "n_cells": 3,
        "fraction_within_group": 0.006
      },
      {
        "group": "Irf8",
        "state": "Ltf",
        "n_cells": 3,
        "fraction_within_group": 0.006
      },
      {
        "group": "Irf8",
        "state": "Prg2",
        "n_cells": 2,
        "fraction_within_group": 0.004
      },
      {
        "group": "Irf8",
        "state": "Ly86",
        "n_cells": 2,
        "fraction_within_group": 0.004
      },
      {
        "group": "Irf8",
        "state": "Prss34",
        "n_cells": 1,
        "fraction_within_group": 0.002
      },
      {
        "group": "Klf4",
        "state": "Car1",
        "n_cells": 175,
        "fraction_within_group": 0.35
      },
      {
        "group": "Klf4",
        "state": "Hba-a2",
        "n_cells": 143,
        "fraction_within_group": 0.286
      },
      {
        "group": "Klf4",
        "state": "Gstm1",
        "n_cells": 106,
        "fraction_within_group": 0.212
      },
      {
        "group": "Klf4",
        "state": "unassigned",
        "n_cells": 66,
        "fraction_within_group": 0.132
      },
      {
        "group": "Klf4",
        "state": "Fcnb",
        "n_cells": 4,
        "fraction_within_group": 0.008
      },
      {
        "group": "Klf4",
        "state": "Vpreb1",
        "n_cells": 2,
        "fraction_within_group": 0.004
      },
      {
        "group": "Klf4",
        "state": "Prg2",
        "n_cells": 2,
        "fraction_within_group": 0.004
      },
      {
        "group": "Klf4",
        "state": "Pf4",
        "n_cells": 1,
        "fraction_within_group": 0.002
      },
      {
        "group": "Klf4",
        "state": "Ly86",
        "n_cells": 1,
        "fraction_within_group": 0.002
      }
    ],
    "annotated_metadata_path": "/mnt/c/Users/jcape/Downloads/omics_pipeline/modules/jcap_perturbseq_integrated_pipeline/sample_data/giladi_mixscape_subset/cell_state_output/metadata_with_cell_state.csv",
    "handoff_path": "/mnt/c/Users/jcape/Downloads/omics_pipeline/modules/jcap_perturbseq_integrated_pipeline/sample_data/giladi_mixscape_subset/cell_state_output/cell_state_handoff.json",
    "warnings": [],
    "metadata": {}
  },
  "cell_state_reporting_instruction": "When summarizing perturbation biology, use the quantitative group_state_summary from cell_state_analysis. If discussing lineage or cell-state composition, report the relevant state names together with their cell counts and percentages where available. Do not merely say that composition shifted. Distinguish this perturbation's state distribution from cross-target comparisons, and do not call a pathway a direct target-gene mechanism when the same signal could be explained by state-composition differences."
}
```

## Cell-State Context

```json
{
  "assay": "perturbseq",
  "status": "ok",
  "annotation_mode": "provided",
  "annotation_source": "metadata:lineage_state",
  "n_cells": 4500,
  "states": [
    "Car1",
    "Cd74",
    "Fcnb",
    "Fcrla",
    "Gstm1",
    "Hba-a2",
    "Ltf",
    "Ly86",
    "Pf4",
    "Prg2",
    "Prss34",
    "Vpreb1",
    "unassigned"
  ],
  "overall_state_summary": [
    {
      "state": "Car1",
      "n_cells": 1786,
      "fraction": 0.3968888888888889
    },
    {
      "state": "Hba-a2",
      "n_cells": 1409,
      "fraction": 0.3131111111111111
    },
    {
      "state": "Gstm1",
      "n_cells": 615,
      "fraction": 0.13666666666666666
    },
    {
      "state": "unassigned",
      "n_cells": 485,
      "fraction": 0.10777777777777778
    },
    {
      "state": "Vpreb1",
      "n_cells": 73,
      "fraction": 0.01622222222222222
    },
    {
      "state": "Fcnb",
      "n_cells": 64,
      "fraction": 0.014222222222222223
    },
    {
      "state": "Ly86",
      "n_cells": 23,
      "fraction": 0.005111111111111111
    },
    {
      "state": "Pf4",
      "n_cells": 12,
      "fraction": 0.0026666666666666666
    },
    {
      "state": "Fcrla",
      "n_cells": 12,
      "fraction": 0.0026666666666666666
    },
    {
      "state": "Prss34",
      "n_cells": 7,
      "fraction": 0.0015555555555555555
    },
    {
      "state": "Ltf",
      "n_cells": 6,
      "fraction": 0.0013333333333333333
    },
    {
      "state": "Prg2",
      "n_cells": 6,
      "fraction": 0.0013333333333333333
    },
    {
      "state": "Cd74",
      "n_cells": 2,
      "fraction": 0.00044444444444444447
    }
  ],
  "group_state_summary": [
    {
      "group": "CTRL",
      "state": "Car1",
      "n_cells": 655,
      "fraction_within_group": 0.43666666666666665
    },
    {
      "group": "CTRL",
      "state": "Hba-a2",
      "n_cells": 532,
      "fraction_within_group": 0.3546666666666667
    },
    {
      "group": "CTRL",
      "state": "Gstm1",
      "n_cells": 163,
      "fraction_within_group": 0.10866666666666666
    },
    {
      "group": "CTRL",
      "state": "unassigned",
      "n_cells": 107,
      "fraction_within_group": 0.07133333333333333
    },
    {
      "group": "CTRL",
      "state": "Fcnb",
      "n_cells": 22,
      "fraction_within_group": 0.014666666666666666
    },
    {
      "group": "CTRL",
      "state": "Vpreb1",
      "n_cells": 11,
      "fraction_within_group": 0.007333333333333333
    },
    {
      "group": "CTRL",
      "state": "Ly86",
      "n_cells": 4,
      "fraction_within_group": 0.0026666666666666666
    },
    {
      "group": "CTRL",
      "state": "Fcrla",
      "n_cells": 2,
      "fraction_within_group": 0.0013333333333333333
    },
    {
      "group": "CTRL",
      "state": "Prss34",
      "n_cells": 2,
      "fraction_within_group": 0.0013333333333333333
    },
    {
      "group": "CTRL",
      "state": "Cd74",
      "n_cells": 1,
      "fraction_within_group": 0.0006666666666666666
    },
    {
      "group": "CTRL",
      "state": "Pf4",
      "n_cells": 1,
      "fraction_within_group": 0.0006666666666666666
    },
    {
      "group": "Cebpa",
      "state": "Car1",
      "n_cells": 239,
      "fraction_within_group": 0.478
    },
    {
      "group": "Cebpa",
      "state": "Hba-a2",
      "n_cells": 232,
      "fraction_within_group": 0.464
    },
    {
      "group": "Cebpa",
      "state": "Gstm1",
      "n_cells": 15,
      "fraction_within_group": 0.03
    },
    {
      "group": "Cebpa",
      "state": "unassigned",
      "n_cells": 10,
      "fraction_within_group": 0.02
    },
    {
      "group": "Cebpa",
      "state": "Vpreb1",
      "n_cells": 2,
      "fraction_within_group": 0.004
    },
    {
      "group": "Cebpa",
      "state": "Ly86",
      "n_cells": 1,
      "fraction_within_group": 0.002
    },
    {
      "group": "Cebpa",
      "state": "Pf4",
      "n_cells": 1,
      "fraction_within_group": 0.002
    },
    {
      "group": "Csf1r",
      "state": "Car1",
      "n_cells": 192,
      "fraction_within_group": 0.384
    },
    {
      "group": "Csf1r",
      "state": "Hba-a2",
      "n_cells": 147,
      "fraction_within_group": 0.294
    },
    {
      "group": "Csf1r",
      "state": "Gstm1",
      "n_cells": 72,
      "fraction_within_group": 0.144
    },
    {
      "group": "Csf1r",
      "state": "unassigned",
      "n_cells": 51,
      "fraction_within_group": 0.102
    },
    {
      "group": "Csf1r",
      "state": "Vpreb1",
      "n_cells": 21,
      "fraction_within_group": 0.042
    },
    {
      "group": "Csf1r",
      "state": "Fcnb",
      "n_cells": 7,
      "fraction_within_group": 0.014
    },
    {
      "group": "Csf1r",
      "state": "Fcrla",
      "n_cells": 3,
      "fraction_within_group": 0.006
    },
    {
      "group": "Csf1r",
      "state": "Ly86",
      "n_cells": 3,
      "fraction_within_group": 0.006
    },
    {
      "group": "Csf1r",
      "state": "Pf4",
      "n_cells": 2,
      "fraction_within_group": 0.004
    },
    {
      "group": "Csf1r",
      "state": "Ltf",
      "n_cells": 1,
      "fraction_within_group": 0.002
    },
    {
      "group": "Csf1r",
      "state": "Prss34",
      "n_cells": 1,
      "fraction_within_group": 0.002
    },
    {
      "group": "Fcgr3",
      "state": "Car1",
      "n_cells": 159,
      "fraction_within_group": 0.318
    },
    {
      "group": "Fcgr3",
      "state": "unassigned",
      "n_cells": 117,
      "fraction_within_group": 0.234
    },
    {
      "group": "Fcgr3",
      "state": "Hba-a2",
      "n_cells": 111,
      "fraction_within_group": 0.222
    },
    {
      "group": "Fcgr3",
      "state": "Gstm1",
      "n_cells": 72,
      "fraction_within_group": 0.144
    },
    {
      "group": "Fcgr3",
      "state": "Vpreb1",
      "n_cells": 14,
      "fraction_within_group": 0.028
    },
    {
      "group": "Fcgr3",
      "state": "Fcnb",
      "n_cells": 9,
      "fraction_within_group": 0.018
    },
    {
      "group": "Fcgr3",
      "state": "Ly86",
      "n_cells": 7,
      "fraction_within_group": 0.014
    },
    {
      "group": "Fcgr3",
      "state": "Pf4",
      "n_cells": 6,
      "fraction_within_group": 0.012
    },
    {
      "group": "Fcgr3",
      "state": "Prg2",
      "n_cells": 1,
      "fraction_within_group": 0.002
    },
    {
      "group": "Fcgr3",
      "state": "Prss34",
      "n_cells": 1,
      "fraction_within_group": 0.002
    },
    {
      "group": "Fcgr3",
      "state": "Cd74",
      "n_cells": 1,
      "fraction_within_group": 0.002
    },
    {
      "group": "Fcgr3",
      "state": "Fcrla",
      "n_cells": 1,
      "fraction_within_group": 0.002
    },
    {
      "group": "Fcgr3",
      "state": "Ltf",
      "n_cells": 1,
      "fraction_within_group": 0.002
    },
    {
      "group": "Flt3",
      "state": "Car1",
      "n_cells": 169,
      "fraction_within_group": 0.338
    },
    {
      "group": "Flt3",
      "state": "Gstm1",
      "n_cells": 115,
      "fraction_within_group": 0.23
    },
    {
      "group": "Flt3",
      "state": "Hba-a2",
      "n_cells": 107,
      "fraction_within_group": 0.214
    },
    {
      "group": "Flt3",
      "state": "unassigned",
      "n_cells": 86,
      "fraction_within_group": 0.172
    },
    {
      "group": "Flt3",
      "state": "Fcnb",
      "n_cells": 7,
      "fraction_within_group": 0.014
    },
    {
      "group": "Flt3",
      "state": "Ly86",
      "n_cells": 5,
      "fraction_within_group": 0.01
    },
    {
      "group": "Flt3",
      "state": "Vpreb1",
      "n_cells": 3,
      "fraction_within_group": 0.006
    },
    {
      "group": "Flt3",
      "state": "Fcrla",
      "n_cells": 3,
      "fraction_within_group": 0.006
    },
    {
      "group": "Flt3",
      "state": "Prss34",
      "n_cells": 2,
      "fraction_within_group": 0.004
    },
    {
      "group": "Flt3",
      "state": "Prg2",
      "n_cells": 1,
      "fraction_within_group": 0.002
    },
    {
      "group": "Flt3",
      "state": "Ltf",
      "n_cells": 1,
      "fraction_within_group": 0.002
    },
    {
      "group": "Flt3",
      "state": "Pf4",
      "n_cells": 1,
      "fraction_within_group": 0.002
    },
    {
      "group": "Irf8",
      "state": "Car1",
      "n_cells": 197,
      "fraction_within_group": 0.394
    },
    {
      "group": "Irf8",
      "state": "Hba-a2",
      "n_cells": 137,
      "fraction_within_group": 0.274
    },
    {
      "group": "Irf8",
      "state": "Gstm1",
      "n_cells": 72,
      "fraction_within_group": 0.144
    },
    {
      "group": "Irf8",
      "state": "unassigned",
      "n_cells": 48,
      "fraction_within_group": 0.096
    },
    {
      "group": "Irf8",
      "state": "Vpreb1",
      "n_cells": 20,
      "fraction_within_group": 0.04
    },
    {
      "group": "Irf8",
      "state": "Fcnb",
      "n_cells": 15,
      "fraction_within_group": 0.03
    },
    {
      "group": "Irf8",
      "state": "Fcrla",
      "n_cells": 3,
      "fraction_within_group": 0.006
    },
    {
      "group": "Irf8",
      "state": "Ltf",
      "n_cells": 3,
      "fraction_within_group": 0.006
    },
    {
      "group": "Irf8",
      "state": "Prg2",
      "n_cells": 2,
      "fraction_within_group": 0.004
    },
    {
      "group": "Irf8",
      "state": "Ly86",
      "n_cells": 2,
      "fraction_within_group": 0.004
    },
    {
      "group": "Irf8",
      "state": "Prss34",
      "n_cells": 1,
      "fraction_within_group": 0.002
    },
    {
      "group": "Klf4",
      "state": "Car1",
      "n_cells": 175,
      "fraction_within_group": 0.35
    },
    {
      "group": "Klf4",
      "state": "Hba-a2",
      "n_cells": 143,
      "fraction_within_group": 0.286
    },
    {
      "group": "Klf4",
      "state": "Gstm1",
      "n_cells": 106,
      "fraction_within_group": 0.212
    },
    {
      "group": "Klf4",
      "state": "unassigned",
      "n_cells": 66,
      "fraction_within_group": 0.132
    },
    {
      "group": "Klf4",
      "state": "Fcnb",
      "n_cells": 4,
      "fraction_within_group": 0.008
    },
    {
      "group": "Klf4",
      "state": "Vpreb1",
      "n_cells": 2,
      "fraction_within_group": 0.004
    },
    {
      "group": "Klf4",
      "state": "Prg2",
      "n_cells": 2,
      "fraction_within_group": 0.004
    },
    {
      "group": "Klf4",
      "state": "Pf4",
      "n_cells": 1,
      "fraction_within_group": 0.002
    },
    {
      "group": "Klf4",
      "state": "Ly86",
      "n_cells": 1,
      "fraction_within_group": 0.002
    }
  ],
  "annotated_metadata_path": "/mnt/c/Users/jcape/Downloads/omics_pipeline/modules/jcap_perturbseq_integrated_pipeline/sample_data/giladi_mixscape_subset/cell_state_output/metadata_with_cell_state.csv",
  "handoff_path": "/mnt/c/Users/jcape/Downloads/omics_pipeline/modules/jcap_perturbseq_integrated_pipeline/sample_data/giladi_mixscape_subset/cell_state_output/cell_state_handoff.json",
  "warnings": [],
  "metadata": {}
}
```

## Cross-target Summary

| Target | Literature assessments |
|---|---:|
| CEBPA | 420 |
| CSF1R | 420 |
| FCGR3 | 420 |
| FLT3 | 420 |
| IRF8 | 420 |
| KLF4 | 420 |

## CEBPA

**Mixscape:** KO fraction: 0.1281; NP fraction: 0.8719

**Top DE genes:** Npm1, Fth1, Mcm2, Rpl8, AK158095;Serbp1

**Top network candidates:** intracellular anatomical structure (GO:CC), metabolic process (GO:BP), binding (GO:MF), intracellular organelle (GO:CC), organelle (GO:CC)

**Triage highlights:**
- Proteasome/proteostasis components (Psma*/Psmb*/Psmd*/Psmc*) are broadly enriched, most consistent with a reactive stress/turnover program rather than a specific causal pathway.
- Checkpoint/DNA‑damage (DDR/p53) terms are present but largely tracked by proteasome and cell‑cycle genes, consistent with a reactive stress or fitness effect of CRISPR editing.
- Cell‑cycle/proliferation signatures (DNA replication and mitotic modules) are detectable and likely reflect reactive fitness/expansion differences across perturbations rather than a shared causal driver.

**Literature evidence:** 420 assessments
- Evidence levels: background: 6, contextual_support: 411, exclude: 3
- Hereditary Predisposition to Acute Myeloid Leukemia: A Novel Germline CEBPA Mutation in a Multigenerational Family. [contextual_support; relevance=3.0]
- More precise risk stratification for de novo acute myeloid leukemia with double-mutation CEBPA. [contextual_support; relevance=3.0]
- Clinical outcomes and molecular features of CEBPA-mutated MPAL: a multi-center case series. [contextual_support; relevance=3.0]

## CSF1R

**Mixscape:** KO fraction: 0.458; NP fraction: 0.542

**Top DE genes:** Pla2g12a, Mthfd1, Fam132a, Klf1, Ermap

**Top network candidates:** intracellular anatomical structure (GO:CC), cytoplasm (GO:CC), intracellular organelle (GO:CC), organelle (GO:CC), metabolic process (GO:BP)

**Triage highlights:**
- A proteasome/UPR-proteostasis program is activated and is most consistent with a reactive stress response to CRISPR perturbation rather than a specific target-driven mechanism.
- p53/G1–S DNA damage checkpoint gene enrichment likely reflects a reactive response to genome editing and growth stress rather than a primary, target-specific program.
- Non-canonical NF-κB pathway terms are weakly enriched and are most consistent with a reactive inflammatory/stress response rather than target-specific regulation.

**Literature evidence:** 420 assessments
- Evidence levels: background: 20, contextual_support: 370, exclude: 30
- Inhibition of adipocyte-macrophage interaction via CSF1-CSF1R by curcumin. [contextual_support; relevance=3.0]
- Virtual single-cell perturbation and genetic causal inference reveal &lt;i&gt;CSF1R&lt;/i&gt;-dependent immunometabolic communication in iron metabolism-associated osteoarthritis. [contextual_support; relevance=3.0]
- Targeting IGF2BP3-driven CSF1/CSF1R signaling disrupts the positive feedback loop between tumor cells and macrophages and inhibits HCC progression. [contextual_support; relevance=3.0]

## FCGR3

**Mixscape:** KO fraction: 0.4013; NP fraction: 0.5987

**Top DE genes:** Isg20, Hba-a2, Cd24a, Beta-s, Glrx5

**Top network candidates:** metabolic process (GO:BP), intracellular anatomical structure (GO:CC), intracellular organelle (GO:CC), organelle (GO:CC), membrane-bounded organelle (GO:CC)

**Triage highlights:**
- Mitochondrial oxidative phosphorylation and respiratory chain signatures are strongly enriched, consistent with elevated OXPHOS, but the magnitude can be explained by lineage state composition differences across perturbations rather than within‑state regulation.
- Proteasome/UPR–proteostasis components are enriched, consistent with increased proteolytic and protein homeostasis activity, but likely reflect proliferation/stress coupling rather than a primary driver.
- Low‑level NF‑κB/innate antiviral signatures are present but are uncertain given low scores and potential dissociation/vector triggers.

**Literature evidence:** 420 assessments
- Evidence levels: contextual_support: 295, exclude: 125
- NK Cytotoxicity Mediated by NK-92 Cell Lines Expressing Combinations of Two Allelic Variants for FCGR3. [contextual_support; relevance=3.0]
- Quantitative analysis of myeloid cell patterns and immunosuppressive enzyme (IDO, ARG1) expression in colorectal cancer pulmonary metastases and corresponding primary tumours. [contextual_support; relevance=3.0]
- Integrative multi-dataset analysis identifies immune-inflammatory hub genes as diagnostic biomarkers and therapeutic targets for age-related hearing loss. [contextual_support; relevance=3.0]

## FLT3

**Mixscape:** KO fraction: 0.4327; NP fraction: 0.5673

**Top DE genes:** Hba-a2, Hars, Beta-s, Cd24a, Tmem14c

**Top network candidates:** metabolic process (GO:BP), intracellular anatomical structure (GO:CC), binding (GO:MF), protein binding (GO:MF), regulation of metabolic process (GO:BP)

**Triage highlights:**
- Proteasome/UPR and S-phase/replication programs are likely reactive stress and proliferation responses rather than primary drivers.
- DNA-damage/p53-associated responses are present and are likely reactive to CRISPR editing/transduction rather than target-specific regulation.
- Mitochondria/OXPHOS transcriptional enrichment is observed but remains uncertain due to composition differences and growth/stress confounding.

**Literature evidence:** 420 assessments
- Evidence levels: background: 24, contextual_support: 390, exclude: 6
- Novel juxtamembrane FLT3 in-frame insertion in a child with refractory AML: a case report of durable remission following allogeneic HSCT and FLT3 inhibitor therapy [contextual_support; relevance=3.0]
- Raf-MEK-ERK signaling mediates FLT3-induced mechanical allodynia and reduced morphine analgesic efficacy in male rodents [contextual_support; relevance=3.0]
- Divergent Clonal Trajectories of FLT3-ITD in Acute Myeloid Leukemia and Their Clinical and Transcriptomic Associations. [contextual_support; relevance=3.0]

## IRF8

**Mixscape:** KO fraction: 0.3474; NP fraction: 0.6526

**Top DE genes:** 1100001G20Rik, S100a9, S100a8, Minpp1, Nfia

**Top network candidates:** intracellular anatomical structure (GO:CC), metabolic process (GO:BP), positive regulation of cellular process (GO:BP), binding (GO:MF), organelle (GO:CC)

**Triage highlights:**
- Proteasome and protein-folding chaperone programs are prominently enriched and are most consistent with a reactive proteostasis/UPR response to perturbation stress rather than a primary driver mechanism.
- Mitochondrial inner-membrane/OXPHOS enrichment is present and is more consistent with a reactive metabolic shift or composition effect than a primary driver.
- DNA-damage/p53 checkpoint and S-phase replication signatures likely reflect CRISPR cutting–induced viability effects rather than on-target regulatory programs.

**Literature evidence:** 420 assessments
- Evidence levels: background: 33, contextual_support: 367, exclude: 20
- IRF8 Suppresses Hepatocellular Carcinoma Progression Through LIAS-Dependent IFN Production and Cuproptosis Activation. [contextual_support; relevance=3.0]
- IRF8 transcriptionally drives NAIP to promote NLRP3/NLRC4 inflammasome-mediated renal tubular pyroptosis in diabetic kidney disease. [contextual_support; relevance=3.0]
- Cone photoreceptor ablation in microglia-deficient larval zebrafish retina elicits a regenerative response alongside a compensatory immune cell response. [contextual_support; relevance=3.0]

## KLF4

**Mixscape:** KO fraction: 0.4943; NP fraction: 0.5057

**Top DE genes:** Fth1, Slc43a3, Hbb-b1, Car2, Fech

**Top network candidates:** metabolic process (GO:BP), intracellular anatomical structure (GO:CC), positive regulation of biological process (GO:BP), positive regulation of cellular process (GO:BP), organelle (GO:CC)

**Triage highlights:**
- Broad induction of proteasome and chaperone genes is consistent with a proteostasis/UPR-like reactive program rather than a primary driver.
- Mitochondrial OXPHOS and inner-membrane gene enrichment suggests a reactive metabolic shift rather than a primary causal mechanism.
- Enrichment of p53/DNA damage checkpoint terms is consistent with a CRISPR-induced DNA damage response and is likely reactive.

**Literature evidence:** 420 assessments
- Evidence levels: contextual_support: 411, exclude: 9
- Decreased expression of Krüppel-like factor 4 is associated with colorectal cancer progression. [contextual_support; relevance=3.0]
- Novel netrin-1-Unc5b-Klf4 axis balances AT2 survival in experimental bronchopulmonary dysplasia. [contextual_support; relevance=3.0]
- Rosiglitazone alleviates white matter injury by inhibiting NLRP3/capspase-1 mediated microglial pyroptosis via downregulating KLF4. [contextual_support; relevance=3.0]

## Interpretation Notes

- Mixscape classifications estimate perturbation-responsive versus non-perturbed cells.
- Differential expression and enrichment summarize downstream transcriptional consequences and do not by themselves establish direct molecular interaction.
- Network and LLM outputs are prioritization aids rather than independent experimental validation.
- Literature evidence is ranked contextual support and should be interpreted with assay, organism, and biological-context matching in mind.
