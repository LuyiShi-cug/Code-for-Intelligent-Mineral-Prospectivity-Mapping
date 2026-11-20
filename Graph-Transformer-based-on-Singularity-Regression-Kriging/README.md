Data Preparation
---------------------
Before training, make sure the required inputs are prepared. Three folders are involved:

- `input_data/` — input data for the SRK and SRK\_residual modes.
- `idw_data/` — input data for the spatial and spectral branches of the IDW mode.
- `data2/` — the SRK outputs for the 39 geochemical elements, i.e., the residual term, the trend term, and the predicted values.
How to use it?
---------------------
- `model.py` defines the model architecture.  
- `dual_model.py` combines SRK, IDW, and SRK\_residual inputs for dual-branch training, while  `branch_model.py` runs single-branch training on either spatial or spectrum input.  
- `training_utils.py` contains the functions shared by both scripts.

Before running, update `DATA_ROOT`, `SAMPLE_INDEX_PATH`, `REFERENCE_TIF`, and `OUTPUT_DIR`. 

In `dual_model.py`, set `METHOD` to `"srk"`, `"idw"`, or `"srk_residual"`; in `branch_model.py`, set `BRANCH` to `"spatial"` or `"spectrum"`.  
Then run the corresponding file directly. The script saves the geochemical anomaly `.tif`, the loss logs, and the loss curves.  
 
SRK\_residual mode requires `input_data2/SpatialInput.npy`, `input_data2/TransformerInput.npy`, `data2/srk_krige_resid.csv`, and `data2/srk_trend.csv`.  
SRK mode requires `input_data2/SRK_pred_SpatialInput.npy` and `input_data2/SRK_pred_TransformerInput.npy`.  
IDW mode requires `idw_data/SpatialInput.npy` and `idw_data/TransformerInput.npy`.  
Spatial mode requires `input_data2/SpatialInput.npy` and Spectrum mode requires `input_data2/TransformerInput.npy`.

All modes use `input_data2/PointXY.npy`, `input_data2/label1.npy`, and the sample index file. The dual-branch of the SRK\_residual mode and the patial branch additionally use `input_data2/GeologyData.npy` as supplementary input. The sample index file can be generated on the first run of the SRK\_residual dual-branch setup.

System-specific notes
---------------------

Dependencies include PyTorch, PyTorch Geometric, NumPy, Pandas, SciPy, scikit-learn, and Matplotlib; generating GeoTIFF output additionally requires GDAL.

Reference(singularity regression kriging):
--------------------
[Ren, K., Song, Y., Chen, M., Yu, Q., 2026. A singularity regression kriging for spatial prediction. GIScience & Remote Sensing, 63(1), 2690341.](https://doi.org/10.1080/15481603.2026.2690341)

Contact Information:
--------------------

Ying Xu: Xvying1021@gmail.com<br>
