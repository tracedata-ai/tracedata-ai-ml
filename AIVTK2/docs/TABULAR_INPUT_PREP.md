# Prepare tabular inputs for AIVT2

This guide prepares a tabular bundle (regression) in AI Verify style under `AIVTK2/artifacts/tabular/`.

Reference:

- [Prepare Tabular Datasets and Models](https://aiverify-foundation.github.io/aiverify/detailed-guide/input-preparation/prepare-tabular/)

## What gets generated

For each bundle:

- `data/testing_dataset.sav` (pickle DataFrame)
- `data/ground_truth_dataset.sav` (pickle DataFrame, target column only)
- `data/background_dataset.sav` (pickle DataFrame, representative feature rows)
- `model/smoothness_model.joblib` (model file copy)
- `bundle_metadata.json`

## Run (PowerShell)

From repo root:

```powershell
./AIVTK2/scripts/prepare-tabular.ps1
```

Optional custom bundle name:

```powershell
./AIVTK2/scripts/prepare-tabular.ps1 -BundleName tracedata-regression-v1
```

## Direct Python command

```bash
uv run python AIVTK2/scripts/prepare_tabular_inputs.py --n-samples 500 --background-rows 96 --model-path models/smoothness_model.joblib
```

## Column mapping for this model

- Feature columns:
  - `accel_fluidity`
  - `driving_consistency`
  - `comfort_zone_percent`
- Target column:
  - `smoothness_label`
- Sensitive/context columns included in test data:
  - `age`
  - `years_experience`

## Notes

- If `models/smoothness_model.joblib` is missing, train first:
  - `uv run python -m src.mlops.production_window_training`
- Files are serialized using pickle/joblib, matching tabular examples in the AI Verify guide.

