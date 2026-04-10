# AIVTK2 setup (single folder)

This folder centralizes all AI Verify Toolkit 2.0 integration assets for this repo:

- Docker Compose: `AIVTK2/docker/docker-compose.yml`
- Dockerfile (helper): `AIVTK2/docker/Dockerfile`
- Env template: `AIVTK2/docker/.env.example`
- Auto-run script (PowerShell): `AIVTK2/scripts/run-aivt2.ps1`

Reference:

- [AI Verify Docker setup](https://aiverify-foundation.github.io/aiverify/getting-started/docker-setup/)

## One-command startup (Windows PowerShell)

From repo root:

```powershell
./AIVTK2/scripts/run-aivt2.ps1 -Mode portal
```

Portal + automated tests (venv workers):

```powershell
./AIVTK2/scripts/run-aivt2.ps1 -Mode automated-venv
```

Portal + automated tests (docker workers):

```powershell
./AIVTK2/scripts/run-aivt2.ps1 -Mode automated-docker
```

Run TraceData compliance helper container:

```powershell
./AIVTK2/scripts/run-aivt2.ps1 -Mode helper -Build
```

Stop all services:

```powershell
./AIVTK2/scripts/run-aivt2.ps1 -Down
```

## Notes

- The script auto-creates `AIVTK2/docker/.env` from `.env.example` on first run.
- Default portal URL: `http://localhost:3000`.
- If port conflicts occur, update `AIVTK2/docker/.env`.

