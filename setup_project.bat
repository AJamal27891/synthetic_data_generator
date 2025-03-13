@echo off
REM =============================================
REM Create Project Directory Structure
REM =============================================
echo Creating project directories...
mkdir data
mkdir metadata
mkdir src
mkdir notebooks
mkdir docs

REM Create placeholder files if they don't exist
if not exist README.md echo # Synthetic Data Project > README.md
if not exist requirements.txt echo pandas^&echo numpy^&echo faker^&echo PyYAML^&echo sdv > requirements.txt
if not exist config.yaml type nul > config.yaml

echo.
echo Project directories and placeholder files created.
echo.

REM =============================================
REM Create Conda Environment with Python 3.11
REM =============================================
echo Creating Conda environment "synthetic_data_project" with Python 3.11...
conda create -y -n synthetic_data_project python=3.11

REM =============================================
REM Activate the Environment and Install Dependencies
REM =============================================
echo Activating environment "synthetic_data_project"...
call conda activate synthetic_data_project

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo Conda environment setup complete.
echo.
pause
