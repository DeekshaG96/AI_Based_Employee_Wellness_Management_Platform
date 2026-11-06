# Run Milestone 2 in Google Colab

I cannot launch an external Colab runtime from this local session. The
repository includes a Colab-ready notebook at
`notebooks/milestone_2_goemotions_colab.ipynb`.

Before opening it, create two local zip files without changing the Downloads
reference folder:

1. A repository zip containing the project source.
2. `archive (1)` as a zip containing its `data` directory and the
   `goemotions-distilbert` checkpoint as a separate zip.

Upload those archives when the notebook asks. Do not upload `.env` files,
databases, caches, virtual environments, `node_modules`, or generated bundles.
The notebook runs the repository's existing trainer, evaluator, calibration,
and comparison scripts; it does not fabricate data or metrics.

Recommended Colab runtime:

- Runtime > Change runtime type > T4 GPU
- Python 3.10+ runtime
- Keep the test split untouched until validation checkpoint selection and
  validation-only threshold calibration are complete

The notebook saves aggregate JSON reports under `/content/moodmentor-results`.
Download those reports and place them in the worktree's `docs/` directory only
after reviewing that the reports contain no source text or predictions.

The current local evidence remains the source of truth until a fresh Colab run
produces complete, same-policy validation and held-out test artifacts.
