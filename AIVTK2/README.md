# AIVTK2

Central folder for AI Verify Toolkit 2.0 integration in this repository.

## Structure

- `AIVTK2/docker/` - compose and Dockerfile
- `AIVTK2/scripts/` - helper scripts to run/stop stack
- `AIVTK2/docs/` - setup and usage docs

## Data preparation

Prepare AI Verify tabular input bundle:

```powershell
./AIVTK2/scripts/prepare-tabular.ps1
```

See:

- `AIVTK2/docs/TABULAR_INPUT_PREP.md`

## Quick run (PowerShell)

```powershell
./AIVTK2/scripts/run-aivt2.ps1 -Mode portal
```

Then open:

- `http://localhost:3000`

