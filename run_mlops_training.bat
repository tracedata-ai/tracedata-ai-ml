@echo off
REM Quick Start Script for MLOps Pipeline (Windows)

setlocal enabledelayedexpansion

echo.
echo ================================================================
echo.          MLOps Training Pipeline - Quick Start
echo.
echo ================================================================
echo.

REM Step 1: Check Python
echo [1/6] Checking Python...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Install Python and add to PATH.
    exit /b 1
)

REM Step 2: Install dependencies
echo.
echo [2/6] Checking dependencies...
python -m pip list | find "mlflow" >nul
if %ERRORLEVEL% NEQ 0 (
    echo   Installing MLFlow...
    python -m pip install mlflow pyyaml -q
)

python -m pip list | find "xgboost" >nul
if %ERRORLEVEL% NEQ 0 (
    echo   Installing XGBoost...
    python -m pip install xgboost -q
)

REM Step 3: Create directories
echo.
echo [3/6] Creating directories...
if not exist models mkdir models
if not exist reports mkdir reports
if not exist data mkdir data
if not exist logs mkdir logs
if not exist mlruns mkdir mlruns

REM Step 4: Show configuration
echo.
echo [4/6] MLOps Configuration loaded from mlops_config.yaml

REM Step 5: Run training
echo.
echo ================================================================
echo.                   Starting Training...
echo.
echo ================================================================
echo.

python -m src.mlops.training_pipeline

REM Step 6: Show results
echo.
echo ================================================================
echo.                   Training Complete!
echo.
echo ================================================================
echo.
echo RESULTS:
echo   * Model saved to: models\smoothness_model.joblib
echo   * Training data: data\train.csv
echo   * Feature importance: reports\feature_importance.json
echo   * MLFlow runs: mlruns\
echo.
echo NEXT STEPS:
echo   1. View results in MLFlow:
echo      mlflow ui
echo.
echo   2. Open in browser:
echo      http://localhost:5000
echo.
echo   3. Compare experiments:
echo      mlflow experiments list
echo.
echo SETUP COMPLETE!
echo.

pause
