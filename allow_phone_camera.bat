@echo off
REM One-time setup: allow the phone camera (browser mode) through Windows Firewall.
REM Just double-click this file and click "Yes" on the security prompt.
REM (Only needed once. Port must match config.yaml -> camera.browser.port, default 8443.)

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator permission...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

set PORT=8443
netsh advfirewall firewall delete rule name="PelakYab phone camera" >nul 2>&1
netsh advfirewall firewall add rule name="PelakYab phone camera" dir=in action=allow protocol=TCP localport=%PORT%

echo.
echo Done - port %PORT% is now open for the PelakYab phone camera.
echo You can close this window.
pause
