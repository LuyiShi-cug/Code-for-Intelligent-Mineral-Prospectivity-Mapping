# Figure Audit

This audit records the status of the code-generated figure outputs included in this package. It follows the supplied manuscript revision and the plotting tables retained under `plot_data/`.

| Figure | Status | Finding |
|---|---|---|
| 3 | Corrected | The current TIFF contains the six captioned observations (Au, Cu, Cr, Pb, MAG, and GRA) with available geology overlays. The separate other-gold-deposit layer named in the caption was unavailable, so it is not invented. |
| 4 | Passed numeric check | At the 10% positive ratio, the stored means and standard deviations match the manuscript for GCN, HGCN, GT, SSL-GT, and SSL-GGT. |
| 5 | Passed numeric check | The stored ten-seed summaries and bootstrap intervals reproduce the reported 10% mean Jaccard values. |
| 6 | Data check passed | The stored rank raster and anchor scores agree, and the known positives fall in the top-ranked area. The rendered TIFF is retained; coordinate-bearing source rasters and SHP layers are omitted here. |
| 7 | Retained legacy analysis | Uses 556 target-union anchors with complete per-anchor Integrated Gradients, ten selected features, and 31 known deposits. The reproducible TIFF is `figures/figure_07/figure_7_legacy_local_ig.tif`; source plotting tables are `plot_data/figure_07/plot_points.csv` and `feature_selection.csv`. |
| 8 | Supplemental | No Figure 8 caption appears in the supplied manuscript version. The TIFF is an entity-mask/input contrast analysis. |
| 9 | Supplemental | No Figure 9 caption appears in the supplied manuscript version. The TIFF is a four-target interpretation summary. |

The historical checkpoint archive is included for provenance. It contains reconstructed runs and should not be described as a fresh execution of the complete paper-aligned schedule unless the external input bundle is supplied and the notebook is rerun.
