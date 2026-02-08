@echo off
REM Convert METHODOLOGY_BERT_MTL.md to DOCX
REM Requires Pandoc: https://pandoc.org/installing.html

echo Converting METHODOLOGY_BERT_MTL.md to DOCX...
echo.

REM Check if pandoc is installed
where pandoc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Pandoc not found!
    echo.
    echo Please install Pandoc:
    echo   1. Download from: https://pandoc.org/installing.html
    echo   2. Or use chocolatey: choco install pandoc
    echo.
    echo Alternatively, open METHODOLOGY_BERT_MTL.md directly in Microsoft Word!
    pause
    exit /b 1
)

REM Convert to DOCX with table of contents
pandoc METHODOLOGY_BERT_MTL.md -o METHODOLOGY_BERT_MTL.docx --toc --toc-depth=3 --number-sections

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ SUCCESS! Created METHODOLOGY_BERT_MTL.docx
    echo.
    echo You can now:
    echo   - Open in Microsoft Word
    echo   - Copy sections to your research paper
    echo   - Use as supplementary material
    echo.
) else (
    echo.
    echo ❌ Conversion failed!
    echo.
    echo Try opening METHODOLOGY_BERT_MTL.md directly in Microsoft Word instead.
    echo.
)

pause
