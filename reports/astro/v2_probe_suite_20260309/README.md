# Astro `v2 final` Probe Suite Summary

Date: 2026-03-09

## Overview
- Backbone checkpoint: `astro_sdss_v2_20gb_curated/experiments/full_20260306_160018_tty/checkpoint-final.pth`
- Evaluation data: SDSS curated split, `train=743`, `val=83`
- Probe setting: frozen backbone, metadata disabled for redshift tasks
- Main goal: validate whether `v2 final` learned a usable unified astronomical representation

## Included artifacts
- `redshift_probe_summary.csv`: baseline frozen redshift probe using `z_shared`
- `feature_ablation_summary.csv`: `z_shared / z_img / z_spec` comparison
- `linear_vs_mlp_summary.csv`: linear vs MLP probe comparison
- `concat_summary.csv`: `shared + private` feature concatenation results
- `retrieval_summary.csv`: image↔spectrum retrieval metrics
- Representative PNGs for probe and retrieval visualization

## Key findings
1. `z_shared` is useful and supports cross-modal retrieval.
2. `z_spec` carries stronger redshift-specific detail than `z_shared`.
3. `z_shared` is more linear/readable; `z_img` and `z_spec` are stronger but more nonlinear.
4. `z_shared + z_spec` is the best current downstream interface for redshift.
5. Cross-modal retrieval with `z_shared` is clearly above random (`R@1 ~ 0.19-0.22` on 83 candidates).

## Recommended interpretation
- `z_shared`: unified semantic alignment space
- `z_spec`: modality-private science-detail carrier
- `z_shared + z_spec`: best current task interface for `photo-z`

## Repro paths
The full non-tracked experiment outputs remain under ignored local paths:
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_feature_ablation/`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_linear_ablation/`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_concat_ablation/`
- `astro_sdss_v2_20gb_curated/probes/cross_modal_retrieval/`

This folder contains only the lightweight summaries and representative figures intended for version control.
