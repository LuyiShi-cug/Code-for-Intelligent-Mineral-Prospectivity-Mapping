# Input Data Description

The workflow expects an external southeastern Hubei mineral-exploration data bundle. The bundle is intentionally not included here. The following description records the required schema without recording coordinates, CRS values, coordinate offsets, vector geometries, or machine-readable arrays.

## Observations and features

- 15,550 valid 1-km anchor cells.
- 39 positive geochemical concentration channels: `Ag`, `Al2O3`, `As`, `Au`, `B`, `Ba`, `Be`, `Bi`, `CaO`, `Cd`, `Co`, `Cr`, `Cu`, `F`, `Fe2O3`, `Hg`, `K2O`, `La`, `Li`, `MgO`, `Mn`, `Mo`, `Na2O`, `Nb`, `Ni`, `P`, `Pb`, `Sb`, `SiO2`, `Sn`, `Sr`, `Th`, `Ti`, `U`, `V`, `W`, `Yt`, `Zn`, and `Zr`.
- Two geophysical channels: `Mag0_F` and `Grav0_F`.
- The model input has 41 channels after per-anchor CLR processing of the 39 positive concentrations and channel-wise population Z-scaling. The geophysical channels use channel-wise anchor-wide Z-scaling.

## Heterogeneous graph

- 16,148 total graph nodes: anchor nodes plus geological entity nodes.
- 598 geological entity nodes.
- 165,260 directed edge rows.
- Required graph information: node types, node-to-node edges, relation types, distances or edge weights, and sparse entity-to-anchor membership.
- The graph uses four target-membership columns corresponding to the four paper target areas.

## Labels and target areas

- 36 known model-positive deposits.
- Four target-area membership columns.
- Target-anchor counts are 293, 33, 116, and 114 for target areas 1–4, respectively.
- The training and evaluation bundle also requires positive-label flags, anchor identifiers, grid indices, and any stored score arrays used for validation.

## Geological layers

The original vector layers are external inputs. The workflow expects faults, contact zones, Yanshanian intrusions, carbonate rocks, and the selected skarn gold-polymetallic deposit points. These layers are used to construct entity nodes, constraints, target memberships, or map overlays. Their geometries and attributes are deliberately excluded from this package.

## Coordinate and grid policy

The source project used a local metric grid and previously created shifted-coordinate derivatives by subtracting minimum x and y values. Neither original coordinates nor shifted coordinates are included here. Reproduction requires the researcher to supply the external spatial reference and apply the project-specific anonymization policy before exporting any review artifact.

## Expected external files

The original workflow used compressed array files for chemistry, geophysics, graph structure, sparse aggregation, labels/targets, target membership, and processed features, together with CSV metadata and JSON configuration. Those files are intentionally omitted, including every `.npz` and `.npy`. Use the schemas above and the notebook cells as the contract for reconstructing them in a separate private data root.
