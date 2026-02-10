@echo off
echo Starting Forest Fire Warning System...

echo Starting Backend Server...
start "Backend API" cmd /k "cd backend && python -m uvicorn main:app --reload --port 8000 || pause"

echo Starting Frontend Server...
start "Frontend Website" cmd /k "cd frontend && python -m http.server 3000 || pause"

echo Waiting for servers to initialize...
timeout /t 5

echo Opening Browser...
start http://localhost:3000

echo Done! Servers are running in background windows.
echo You can close this window, but keep the other two server windows open.
pause
