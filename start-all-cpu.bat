@echo off
REM ACE-Step UI - CPU-only startup for Windows
REM For AMD Ryzen with integrated graphics / no dedicated GPU.
setlocal

echo ==================================
echo   ACE-Step Startup (CPU mode)
echo ==================================
echo.

REM --- Force CPU mode (no CUDA/ROCm acceleration) ---
REM Integrated Radeon (Vega) in Ryzen APUs is NOT supported by ROCm/PyTorch,
REM so generation runs on the CPU. The variables below hide the GPU and save memory.
set CUDA_VISIBLE_DEVICES=-1
set HIP_VISIBLE_DEVICES=-1
set ACESTEP_LM_BACKEND=pt
REM DiT-only: disables the heavy LLM (Thinking / AI Enhance) so it fits in ~16 GB RAM.
set ACESTEP_INIT_LLM=false
REM Run the VAE on CPU too (we have no usable GPU).
set ACESTEP_VAE_ON_CPU=1

if not exist "node_modules" (
    echo Error: UI dependencies not installed! Run setup.bat first.
    pause
    exit /b 1
)
if not exist "server\node_modules" (
    echo Error: Server dependencies not installed! Run setup.bat first.
    pause
    exit /b 1
)

if "%ACESTEP_PATH%"=="" (
    set ACESTEP_PATH=..\ACE-Step-1.5
)
if not exist "%ACESTEP_PATH%" (
    echo.
    echo Warning: ACE-Step not found at %ACESTEP_PATH%
    echo Set ACESTEP_PATH or put ACE-Step-1.5 next to ace-step-ui.
    echo Example: set ACESTEP_PATH=C:\ACE-Step-1.5
    pause
    exit /b 1
)

set API_COMMAND=
if exist "%ACESTEP_PATH%\python_embeded\python.exe" (
    echo [+] Detected Windows Portable Package
    set API_COMMAND=python_embeded\python acestep\api_server.py
) else (
    echo [+] Detected Standard Installation
    set API_COMMAND=uv run acestep-api --port 8001
)

echo.
echo   NOTE: CPU generation is slow - a single track may take several minutes.
echo   This is expected.
echo.

REM Environment variables set above are inherited by child windows.
echo [1/3] Starting ACE-Step API server (CPU)...
start "ACE-Step API Server (CPU)" cmd /k "cd /d "%ACESTEP_PATH%" && %API_COMMAND%"

echo Waiting for API to initialize...
timeout /t 5 /nobreak >nul

echo [2/3] Starting backend server...
start "ACE-Step UI Backend" cmd /k "cd /d "%~dp0server" && npm run dev"

timeout /t 3 /nobreak >nul

echo [3/3] Starting frontend...
start "ACE-Step UI Frontend" cmd /k "cd /d "%~dp0" && npm run dev"

timeout /t 2 /nobreak >nul

echo.
echo ==================================
echo   All Services Running! (CPU mode)
echo ==================================
echo.
echo   ACE-Step API: http://localhost:8001
echo   Backend:      http://localhost:3001
echo   Frontend:     http://localhost:3000
echo.
echo   Close the terminal windows to stop the services.
echo.
timeout /t 3 /nobreak >nul
start http://localhost:3000
pause >nul
