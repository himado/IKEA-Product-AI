@echo off
REM DoHa: connect the phone over USB and start the backend server (step 8: room scan upload).
REM Double-click this file. Keep the window open while testing; close it to stop the server.
REM The adb reverse rule is lost whenever adb restarts or the USB cable is re-plugged - run this again then.

set ADB=%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe
set PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
cd /d "%~dp0"

echo [1/2] Phone over USB:
"%ADB%" devices
"%ADB%" reverse tcp:8000 tcp:8000
"%ADB%" reverse --list

echo.
echo [2/2] Starting server on http://127.0.0.1:8000 (phone uses the same address via USB) ...
"%PY%" backend\server\app.py
pause
