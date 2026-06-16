@echo off
echo Starting Dark Trace AI...
echo =========================================

:: Start the backend API in a new window
echo Starting Backend API on port 8000...
start "Dark Trace Backend" cmd /k "call .venv\Scripts\activate.bat 2>nul || echo Virtual env not found, using global python && python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000"

:: Wait a few seconds for the backend to initialize
timeout /t 3 /nobreak > nul

:: Start the frontend HTTP server in a new window
echo Starting Frontend Server on port 8501...
start "Dark Trace Frontend" cmd /k "cd frontend && python -m http.server 8501"

echo =========================================
echo Both servers are starting in separate windows.
echo Once they load, open your web browser to:
echo http://localhost:8501
echo.
echo NOTE: Make sure to HARD REFRESH your browser (Ctrl + F5) 
echo so it loads the new index.html!
pause
