@echo off
echo Starting FastAPI Server...
echo Installing dependencies...
pip install fastapi uvicorn python-dotenv google-generativeai -q

echo.
echo Starting server on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

uvicorn main:app --reload --port 8000 --host 0.0.0.0
pause
