# Interpretability Analysis of Self-Supervised Heterogeneous Graph Deep Learning for Mineral Exploration Targeting

This package is the code, model archive, plotting tables, and rendered figures for the manuscript workflow. It is organized for review and reproducibility without distributing the southeastern Hubei input layers. No original or shifted vector/raster layers and no machine-readable array inputs (`.npz`, `.npy`) are included.

The workflow combines geochemical and geophysical observations with a heterogeneous graph of anchor cells and geological entities. It applies a per-anchor centered log-ratio transform to the 39 positive geochemical channels, channel-wise population Z-scaling, and Z-scaling of the two geophysical channels. The resulting 41-channel node representation is used by GCN, HGCN, GT, SSL-GT, and SSL-GGT comparison models. The later stages rank anchor cells and calculate post hoc Integrated Gradients and entity-mask summaries.

## Package contents

```text
project-root/
|-- README.md
|-- requirements.txt
|-- notebooks/
|   `-- SSL_GGL_paper_workflow.ipynb
|-- code/
|   |-- model/
|   |   |-- run_experiment.py
|   |-- run_knowledge_injection_comparison.py
|   `-- plot_figure_07_legacy.py      # Reproduces the retained Figure 7
|-- plot_data/
|   `-- figure_03/ ... figure_09/    # CSV/JSON tables used by plotted outputs
|-- figures/
|   `-- figure_03/ ... figure_09/    # Current code-generated TIFF outputs
|-- checkpoints/
|   `-- interpretation/              # Copied .pt checkpoints
|-- results/
|   |-- interpretation_archive/      # CSV/JSON interpretation and provenance
|   |-- training_archive/            # CSV/JSON/PDF/PNG historical experiments
|   |-- figures/                     # Archived non-spatial figures
|   `-- tables/                      # Summary tables
|-- metadata/
|   `-- input_data_description.md
`-- docs/assets/paper_workflow_reference.png
```

The `figures/figure_03` through `figures/figure_09` folders contain rendered figure outputs where an output is available. Figure 7 uses the retained legacy dependence analysis: 556 target-union anchors with complete per-anchor Integrated Gradients, ten selected features, and 31 known deposits.

## Input data deliberately omitted

The omitted data are described in [metadata/input_data_description.md](metadata/input_data_description.md). They include the southeastern Hubei geochemical and geophysical grids, geological vector layers, graph arrays, labels, target memberships, processed feature arrays, coordinate-shifted derivatives, and every `.npz`/`.npy` array. This keeps location-bearing source data outside the review package while documenting the exact schemas needed to run the workflow.

## Running the workflow with external inputs

Use Python 3.12 in the prepared `PytorchGDAL` environment. Install the pinned packages with:

```powershell
python -m pip install -r requirements.txt
```

The notebook is intentionally delivered without its input bundle. To execute it, provide an external data root containing the schemas listed in `metadata/input_data_description.md`, set the environment variable `SSL_GGL_DATA_ROOT` to that root, and start the notebook from this project or one of its child directories. The notebook uses project-relative code discovery and does not require a machine-specific absolute path in the package.

Run `SSL_GGL_paper_workflow.ipynb` from top to bottom. The paper-aligned comparison contains five model groups, six positive-label fractions, ten random repeats, and the corresponding pretraining/fine-tuning stages. Training can require substantial GPU memory and time. The `.pt` files in `checkpoints/` and the CSV/JSON archives in `results/` are historical outputs and provenance products; they are not a substitute for rerunning the experiment with the complete external input bundle.

## Interpretation and figure status

Figures 3 through 9 have rendered TIFF outputs in this package. Figures 8 and 9 are supplemental analyses in the supplied manuscript version, which has numbered captions only through Figure 7. Figure 7 is the legacy local-IG dependence analysis and is reproducible from `plot_data/figure_07/` with `code/plot_figure_07_legacy.py`. See [FIGURE_AUDIT.md](FIGURE_AUDIT.md) for the figure-by-figure audit and known limitations.

All included spatially derived outputs are rendered images only. No output SHP or coordinate-bearing raster is distributed in this package.
