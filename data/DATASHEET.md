# Dataset Datasheet — Urea in Milk Droplet Images

**Status:** Phase 2 draft; verify the experimental-procedure details against the original lab report/notebook before treating this as final.

## Dataset purpose

The dataset is intended for a calibration task: estimate the amount of **added urea (% w/v)** in milk from photographs of milk droplets after the p-DMAB reaction. It is being used for an ES 335 machine-learning project. The result is an experimental estimate, not a certified food-safety test.

## Experimental procedure and measurement context

The current project roadmap describes the measurement procedure as mixing 40 µL milk with 20 µL p-DMAB on a PDMS slide, then photographing the droplet from above at approximately two minutes after mixing. It describes Amul Taaza milk, two imaging devices (Dino-Lite and Samsung A31), and two lamp settings (on/off). **Confirm these details and any preparation steps against the original lab report or notebook before final submission.**

## Unit of observation and identifiers

One metadata row represents one original photograph. The active `image_path` points to the original BMP/JPG photograph; the pre-cut PNGs are not used as model inputs in the current plan. `conc` is the target concentration of added urea in % w/v. `device` and `lamp` describe the acquisition condition.

The preparation hierarchy used for leakage control is:

- `instance_id`: preparation instance (1, 2, or 3).
- `tube_id`: one physical tube, identified by `(instance_id, concentration)`. There is one tube per concentration within each instance.
- `droplet_id` and `view_id`: intentionally unpopulated. Individual droplet/photo pairings have not been reconstructed, and droplet-level grouping is not required for the planned tube-level split.

All photographs belonging to one tube share one `tube_id`, regardless of device, lamp setting, or droplet. The planned model evaluation must split by `tube_id`, not by image.

## Dataset composition

The current master table contains **1,064 photographs** and **51 physical tubes**:

| Instance | Source folder(s) | Images | Concentration levels | Tubes |
|---:|---|---:|---:|---:|
| 1 | `data/raw/Dinolite/` excluding `4thnov_validationdata`, and `data/raw/SamsungA31/` | 576 | 24 | 24 |
| 2 | `data/raw/new_data/` | 456 | 19 | 19 |
| 3 | `data/raw/Dinolite/4thnov_validationdata/` | 32 | 8 | 8 |
| **Total** | | **1,064** | | **51** |

Instance 1 includes both Dino-Lite and Samsung A31 images with lamp-on and lamp-off conditions. Instance 2 includes both devices and both lamp conditions. Instance 3 contains Dino-Lite images with lamp-on and lamp-off conditions.

The master metadata is `data/processed/metadata.csv`. It retains source workbook and row provenance, original path text from the Excel file, a relative path to the original photograph, filename/EXIF timestamp provenance, matching method, cohort, instance and tube IDs. The raw image and workbook files remain under `data/raw/` and are not intended to be committed to Git.

## Data preparation and matching

The Excel workbooks have no header row: the first column contains an image path and the second contains the concentration label. Rows are mapped to original photographs by exact filename or, where the workbook points to a processed PNG, by a unique matching filename stem. For the latter, the match establishes a filename correspondence; it is not a pixel-level verification that the processed PNG was generated from that original photo.

The metadata builder assigns lamp conditions for the added cohorts using the recorded workbook row ordering: for each concentration, the first six of twelve rows are lamp-on and the remaining six are lamp-off in instance 2; for each concentration in instance 3, the first two of four rows are lamp-on and the remaining two are lamp-off. These rules are based on the experimenter's clarification and are recorded in `lamp_assignment_source`.

Original photographs are used rather than pre-cut PNGs. ROI masks and colour features will be derived from the original photos in later phases. The project will compare ROI-only features with ROI-plus-background features rather than assume background information is useful.

## Known data-quality issues

- **47 images have no usable filename or EXIF timestamp.** These are in the original Samsung A31 lamp-off condition. Their timestamps remain missing rather than being inferred.
- The original cohort has irregular image counts at some concentration/condition combinations: Dino-Lite off at 0.20% (4 images), Dino-Lite off at 0.40% (4), Dino-Lite on at 0.45% (5), phone off at 0.20% (8), and phone on at 0.20% (9). These counts are retained and must be described; no images are to be invented or silently removed.
- Dino-Lite sequence number A056 appears twice in the original lamp-off 0.40% group, with different timestamps and file paths. The image IDs and paths are unique, but the experimental explanation for the repeated sequence number should be checked against available records or explicitly left unresolved.
- Timestamp precision must be preserved to seconds in the generated CSV. Do not open and re-save the machine-generated CSV in spreadsheet software if it changes timestamp formatting; regenerate it from `src/build_metadata.py` instead.
- Individual droplet and view identities are unavailable in the current metadata. This limits droplet-level repeatability analysis and multi-view aggregation, but does not prevent the planned tube-level split.
- The original experimental design has concentration/order and session limitations described in the project roadmap. These should be discussed in the final limitations section, with exact session details checked against the lab records.

## Intended use and limitations

The dataset supports a course-project investigation of whether droplet colour features can estimate added urea concentration under the represented laboratory conditions. It does not establish performance on other milk brands, reagent batches, slide types, devices, lighting environments, or field photographs. It is not a validated food-safety screening method and must not be presented as one.

## Provenance and outstanding confirmation

The raw files and source spreadsheets are the primary data sources. The metadata builder and integrity-check report provide a reproducible record of image matching, instance/tube assignments, and data-quality checks. Before finalizing this datasheet, verify the procedural details above against the original lab report and document any explanation available for the irregular counts and repeated A056 sequence number.