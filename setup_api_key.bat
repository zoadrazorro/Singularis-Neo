@echo off
REM Setup OpenAI API Key for Beta v3

echo ================================================================================
echo OPENAI API KEY SETUP
echo ================================================================================
echo.
echo This will set your OpenAI API key for the current session.
echo.
echo To make it permanent:
echo   1. Search for "Environment Variables" in Windows
echo   2. Click "Environment Variables" button
echo   3. Under "User variables", click "New"
echo   4. Variable name: OPENAI_API_KEY
echo   5. Variable value: your-api-key-here
echo.
echo ================================================================================
echo.

set /p API_KEY="Enter your OpenAI API key: "

if "%API_KEY%"=="" (
    echo.
    echo ERROR: No API key entered
    pause
    exit /b 1
)

REM Set for current session
set OPENAI_API_KEY=%API_KEY%

echo.
echo ✓ API key set for current session
echo.
echo You can now run:
echo   python run_beta_v3.py
echo.
echo Or test it:
echo   python run_beta_v3.py --test-mode --duration 30
echo.
pause
