# Decision Log

Record each important decision, its reason, and its observed impact.

| Date | Decision | Reason | Evidence or impact |
|---|---|---|---|

2026-10-23 — Expanded metadata to include the original four-condition
experiment, the 456-image new-tube cohort, and the 32-image November 4
validation cohort. Reason: the additional images represent distinct tube
preparations and are useful for studying generalization. Keep cohort identity
explicit and do not mix validation data into development/model selection.

2026-10-23 — Use original photographs and derive an ROI mask during feature
extraction instead of requiring pre-cut transparent PNGs. Reason: preserve
the complete scene so ROI-only features can be compared against ROI plus
background/illumination features. The active image_path points to the original
photo; orig_path preserves the exact path recorded by the source spreadsheet.

2026-10-23 — Documented the current understanding of the three
experimental cohorts and their unresolved tube/droplet identities.
Reason: the original, new-tube and November 4 collections have
different acquisition histories and must not be combined blindly.
Exact tube and droplet IDs will only be assigned where supported
by experimental records or sufficiently strong pairing evidence.
Unrecoverable identities will be flagged and handled using the
roadmap's conservative fallback.

2026-10-23 — Retained all 1,064 original photographs as rows in
the master metadata, with a cohort identifier. The November 4
validation cohort will remain separate from development and model
selection.

## Phase 2 decision-log addendum — 16 Nov 2026

1. **Use original photographs.** The active `image_path` points to original BMP/JPG files. Pre-cut PNGs are not used as model inputs. ROI masks and colour features will be derived from original photographs; ROI-only and ROI-plus-background feature sets will be compared.
2. **Preparation instances.** `original` maps to `instance_id=1`; `new_tubes` maps to `instance_id=2`; `nov4_validation` maps to `instance_id=3`.
3. **Physical tube identity.** There is exactly one physical tube for each concentration in each preparation instance. Therefore `tube_id` is deterministic from `(instance_id, concentration)`, using IDs such as `I1_C0.20`. All images of the same tube share this ID.
4. **Droplet/view identities.** `droplet_id` and `view_id` remain blank because reconstructing each droplet/photo pairing is difficult and is not required for the planned tube-level evaluation. The primary split unit remains `tube_id`.
5. **Lamp assignment for added cohorts.** Instance 2 uses the experimenter-confirmed ordering of six lamp-on then six lamp-off rows per concentration. Instance 3 uses two lamp-on then two lamp-off rows per concentration.
6. **Timestamp handling.** Missing timestamps are not inferred. Machine-generated timestamps must retain seconds precision; rebuild metadata from the source workbooks and image filenames rather than editing/re-saving the CSV in spreadsheet software.
7. **Metadata source of truth.** `src/build_metadata.py` is updated to generate `instance_id` and `tube_id` directly in `data/processed/metadata.csv`. The separate enriched CSV is an intermediate check, not the canonical rebuild target.
8. **Known integrity issues retained for documentation.** The original cohort has uneven counts at specified concentration/condition combinations, and Dino-Lite sequence A056 occurs twice with different file paths/timestamps. These are not silently corrected or deleted.


