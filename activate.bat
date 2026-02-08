@echo off
REM Activate the medical-mtl virtual environment
REM Usage: activate.bat

echo Activating medical-mtl environment...
call medical-mtl\Scripts\activate.bat

echo.
echo ========================================
echo Medical MTL Environment Activated!
echo ========================================
echo.
echo Python: %VIRTUAL_ENV%\Scripts\python.exe
echo.
echo Next steps:
echo   1. Download datasets: python scripts\download_datasets_hf.py --all
echo   2. Run validation: python validate_setup.py
echo   3. Start experiments!
echo.
